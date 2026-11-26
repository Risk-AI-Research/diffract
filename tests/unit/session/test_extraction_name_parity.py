"""Stored names for modules, state dictionaries, and reloaded safetensors weights."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.core.storage.ram_manager import RAMStorageManager
from diffract.session import EmptyExtractionError, Session, SessionError

pytestmark = pytest.mark.unit

if not import_utils.is_available("torch"):
    pytest.skip("torch not installed", allow_module_level=True)

torch = import_utils.require("torch")
nn = torch.nn


class ReferenceNet(nn.Module):
    """Divergence-probing reference: embeddings, buffers, ties, int paths."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)  # 2D but not nn.Linear
        self.encoder = nn.Sequential(  # integer child paths
            nn.Linear(8, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 8, bias=False),
        )
        self.norm = nn.BatchNorm1d(8)  # buffers live in state_dict only
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight  # tied pair


LINEAR_WEIGHT_PATHS = {"encoder.0.weight", "encoder.2.weight", "head.weight"}


@pytest.mark.parametrize("view", ["module", "state_dict"])
@pytest.mark.parametrize("mixed", [False, True], ids=["bf16", "mixed"])
def test_checkpoint_coverage_and_metrics_match_explicit_fp32(tmp_path, view, mixed):
    from diffract import analyze

    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
    state = model.state_dict()
    state["0.weight"] = torch.tensor([[1.011, -2.034, 0.0], [0.0, 4.07, -7.2]])
    state["1.weight"] = torch.tensor([[1e-30, -3.01], [4.06, 1e5]])
    saved = {
        name: value if mixed and name == "0.weight" else value.bfloat16()
        for name, value in state.items()
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    source = model if view == "module" else checkpoint
    analysis, report = analyze(
        source, fields=("frob_norm", "l2_norm"), return_coverage=True
    )
    assert report.counts == dict(
        candidates=2, extracted=2, excluded=2, unsupported=0, failed=0
    )
    entries = {entry.name: entry for entry in report.items}
    assert set(entries) == set(checkpoint)
    metrics = {
        (r["param_name"], r["field"]): r["value_scalar"] for r in analysis.details
    }
    names = {"0.weight", "1.weight"}
    assert set(metrics) == {
        (name, field) for name in names for field in ("frob_norm", "l2_norm")
    }
    with Session(profile="ram") as session:
        session.models.add(source, model_id="net")
        rows = session.results.export_metrics("weights", export_format="dict")
        stored = {r["metadata"]["name"]: r["fields"]["weights"] for r in rows.values()}
        meta = {
            r["name"]: r["other_meta"]
            for r in session.models.parameters.list(verbose=True)
        }
    assert set(stored) == names
    for name in names:
        expected_dtype = "float32" if mixed and name == "0.weight" else "bfloat16"
        assert (
            entries[name].status,
            entries[name].source_dtype,
            entries[name].storage_dtype,
        ) == ("extracted", expected_dtype, "float32")
        assert (meta[name]["source_dtype"], meta[name]["storage_dtype"]) == (
            expected_dtype,
            "float32",
        )
        ref = checkpoint[name].float().numpy()
        assert stored[name].dtype == np.float32
        np.testing.assert_array_equal(stored[name], ref)
        assert metrics[name, "frob_norm"] == pytest.approx(
            np.linalg.norm(ref.astype("float64")), rel=1e-6
        )
        assert metrics[name, "l2_norm"] == pytest.approx(
            np.linalg.norm(ref.astype("float64"), ord=2), rel=1e-6
        )
    for name in ("0.bias", "1.bias"):
        assert (entries[name].status, entries[name].reason) == ("excluded", "bias")


@pytest.mark.parametrize("view", ["module", "state_dict"])
def test_coverage_distinguishes_embeddings_bias_and_unreadable_weights(view):
    model = nn.ModuleDict(
        {
            "embed": nn.Embedding(5, 2),
            "linear": nn.Linear(2, 2),
            "unreadable": nn.Linear(2, 2, bias=False, device="meta"),
            "activation": nn.ReLU(),
        }
    )
    with Session(profile="ram") as session:
        coverage = session.models.add(model if view == "module" else model.state_dict())
        session.compute.apply("frob_norm")
        rows = session.results.export_metrics("frob_norm", export_format="dict")
    entries = {row.name: row for row in coverage.items}
    assert set(entries) == {
        "embed.weight",
        "linear.weight",
        "linear.bias",
        "unreadable.weight",
    }
    assert coverage.counts == {
        "candidates": 3,
        "extracted": 1 if view == "module" else 2,
        "excluded": 1,
        "unsupported": 1 if view == "module" else 0,
        "failed": 1,
    }
    assert (entries["linear.bias"].status, entries["linear.bias"].reason) == (
        "excluded",
        "bias",
    )
    assert entries["unreadable.weight"].status == "failed"
    assert "meta" in entries["unreadable.weight"].reason.lower()
    names = {r["metadata"]["name"] for r in rows.values()}
    if view == "module":
        assert names == {"linear.weight"}
        assert entries["embed.weight"].status == "unsupported"
        assert "Embedding" in entries["embed.weight"].reason
    else:
        assert names == {"linear.weight", "embed.weight"}
        assert entries["embed.weight"].status == "extracted"


@pytest.mark.parametrize("profile", ["ram", "local"])
def test_required_paths_leave_coordinate_reusable(tmp_path, monkeypatch, profile):
    monkeypatch.chdir(tmp_path)
    model = nn.ModuleDict({"head": nn.Linear(2, 2), "other": nn.Linear(2, 2)})
    with torch.no_grad():
        model["other"].weight.copy_(torch.tensor([[1.0, -2.0], [3.0, 5.0]]))
    with Session(profile=profile) as session:
        session.models.add(model, model_id="net", step=0)
        broken = model.state_dict()
        broken["other.weight"] = torch.empty((2, 2), device="meta")
        required = ["head.weight", "other.weight", "new.weight", "head.bias"]
        with (
            patch.object(
                RAMStorageManager,
                "put_blob",
                autospec=True,
                side_effect=RAMStorageManager.put_blob,
            ) as write_blob,
            pytest.raises(SessionError) as err,
        ):
            session.models.add(broken, model_id="net", step=1, required_names=required)
        write_blob.assert_not_called()
        for text in (
            "other.weight",
            "meta",
            "new.weight",
            "absent",
            "head.bias",
            "bias",
        ):
            assert text in str(err.value)
        assert session.models.list(layout="coordinates") == [("net", 0)]
        if profile == "local":
            from diffract.core.storage.maintenance import check_store

            assert not check_store(tmp_path / ".diffract/sqlite/metadata_index.db")[
                "orphan_blobs"
            ]
        coverage = session.models.add(
            model, model_id="net", step=1, required_names=["other.weight"]
        )
        assert {
            row.stored_name for row in coverage.items if row.status == "extracted"
        } == {"other.weight", "head.weight"}
        exported = session.filter(steps=[1]).results.export_metrics(
            "weights", export_format="dict"
        )
        for row in exported.values():
            name = row["metadata"]["name"]
            np.testing.assert_array_equal(
                row["fields"]["weights"], model.state_dict()[name].numpy()
            )
        session.compute.apply("frob_norm")
        assert (
            len(session.results.export_metrics("frob_norm", export_format="list")) == 4
        )
    if profile == "local":
        with Session(profile=profile) as reopened:
            assert reopened.models.list(layout="coordinates") == [
                ("net", 0),
                ("net", 1),
            ]


def test_empty_extraction_carries_per_parameter_reasons():
    with Session(profile="ram") as session, pytest.raises(EmptyExtractionError) as err:
        session.models.add(nn.Embedding(4, 3))
    (item,) = err.value.coverage.items
    assert (item.name, item.status) == ("weight", "unsupported")
    assert "Embedding" in item.reason


def test_required_custom_handler_can_reuse_its_output_buffer():
    from diffract.core.data.nn.extractors.handlers.base import ParameterHandler
    from diffract.core.data.nn.params.schema import ParameterType

    class EmbeddingHandler(ParameterHandler):
        parameter_type = ParameterType.DENSE

        def __init__(self):
            self.buffer = np.empty((2, 2), dtype=np.float32)

        def can_handle(self, module, _name):
            return isinstance(module, nn.Embedding)

        def process(self, module, _name):
            self.buffer[:] = module.weight.detach().numpy()
            return self.buffer

    first = torch.tensor([[2.0, -3.0], [1.0, 4.0]])
    second = torch.tensor([[-5.0, 0.0], [7.0, -1.0]])
    model = nn.Sequential(
        nn.Embedding.from_pretrained(first), nn.Embedding.from_pretrained(second)
    )
    with Session(profile="ram") as session:
        session.models.add(
            model,
            required_names=("0.weight", "1.weight"),
            custom_handlers=[EmbeddingHandler()],
        )
        rows = session.results.export_metrics("weights", export_format="dict")
    stored = {r["metadata"]["name"]: r["fields"]["weights"] for r in rows.values()}
    np.testing.assert_array_equal(stored["0.weight"], first.numpy())
    np.testing.assert_array_equal(stored["1.weight"], second.numpy())


def _stored_names(model_object) -> dict[str, tuple[int, ...]]:
    """(name -> weights shape) as stored by a fresh ram session."""
    with Session(profile="ram") as session:
        session.models.add(model_object, model_id="ref")
        exported = session.results.export("weights", export_format="dict")
    return {
        entry["metadata"]["name"]: tuple(entry["fields"]["weights"].shape)
        for entry in exported.values()
    }


def _reference_state_dict() -> dict:
    torch.manual_seed(0)
    return ReferenceNet().state_dict()


def test_module_view_names_are_state_dict_paths() -> None:
    torch.manual_seed(0)
    names = _stored_names(ReferenceNet())
    # Only nn.Linear modules are handled in the module view; their names are
    # exactly the state_dict paths of the extracted weight tensors.
    assert set(names) == LINEAR_WEIGHT_PATHS


def test_bare_root_module_is_named_weight() -> None:
    assert set(_stored_names(nn.Linear(4, 4, bias=False))) == {"weight"}


def test_module_and_state_dict_views_agree_on_linear_weights() -> None:
    torch.manual_seed(0)
    module_view = _stored_names(ReferenceNet())
    state_view = _stored_names(_reference_state_dict())

    for path in LINEAR_WEIGHT_PATHS:
        assert module_view[path] == state_view[path]

    # Only the state dict includes the embedding weight.
    assert set(state_view) - set(module_view) == {"embed.weight"}
    assert set(module_view) <= set(state_view)


def test_state_dict_and_safetensors_views_agree(tmp_path) -> None:
    st = pytest.importorskip(
        "safetensors.torch",
        reason="safetensors not installed (dev extra); the safetensors "
        "parity leg needs it",
    )
    st_numpy = pytest.importorskip("safetensors.numpy")

    state = _reference_state_dict()

    with pytest.raises(RuntimeError, match="share memory"):
        st.save_file(state, str(tmp_path / "tied.safetensors"))

    untied = {key: tensor.clone() for key, tensor in state.items()}
    st.save_file(untied, str(tmp_path / "ref.safetensors"))
    reloaded = st_numpy.load_file(str(tmp_path / "ref.safetensors"))

    assert _stored_names(reloaded) == _stored_names(state)


def test_numpy_view_admits_2d_tensors_the_torch_view_excludes() -> None:
    tensors = {"pos_table": torch.randn(4, 4), "fc.weight": torch.randn(4, 4)}

    torch_view = _stored_names(tensors)
    numpy_view = _stored_names({k: v.numpy() for k, v in tensors.items()})

    assert set(torch_view) == {"fc.weight"}
    assert set(numpy_view) == {"fc.weight", "pos_table"}
    assert torch_view["fc.weight"] == numpy_view["fc.weight"]


def test_overrides_key_on_weight_tensor_paths() -> None:
    # The public override contract: keys are weight-tensor state_dict paths,
    # identical for the module and state_dict views of the same network.
    from diffract import ParameterOverrides

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

    overrides = {"q_proj.weight": ParameterOverrides(name="q", ptype="attn")}

    with Session(profile="ram") as session:
        session.models.add(Toy(), model_id="m", parameter_overrides=overrides)
        exported = session.results.export("weights", export_format="dict")

    (entry,) = exported.values()
    assert entry["metadata"]["name"] == "q"
    assert entry["metadata"]["parameter_type"] == "ATTN"
