"""Check spectral arithmetic through the public session and persistent cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import linalg

from diffract import Session
from diffract.containers import compute_singleton_container
from diffract.core.compute import InvalidConfiguration, KernelResult
from diffract.core.compute.registry import KernelRegistry
from diffract.session import SessionError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_kernel_registry():
    with compute_singleton_container.kernel_registry.override(KernelRegistry()):
        yield


def test_mixed_svd_run_records_each_input_and_the_custom_kernel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with Session(profile="local") as session:

        @session.compute.kernel()
        def inverse_moment(weights_svals):
            working = weights_svals.astype(np.float64)
            return KernelResult(
                float(np.sum(working**-2)),
                input_dtype=str(weights_svals.dtype),
                working_dtype=str(working.dtype),
                output_dtype="float64",
                backend="numpy",
            )

        session.compute.configure_kernel(
            "weights_svd", allow_cuda=False, precision="preserve"
        )
        inputs = {
            "half": np.diag([2, 4]).astype(np.float16),
            "double": np.diag([1.0 + 2**-32, 2.0]),
        }
        session.models.add(inputs, "design")
        with patch.object(linalg, "svd", wraps=linalg.svd) as solver:
            session.compute.apply("inverse_moment")
        actual = {
            tuple(call.args[0].diagonal()): (
                call.args[0].dtype.name,
                call.kwargs["lapack_driver"],
            )
            for call in solver.call_args_list
        }
        rows = session.results.export(
            "weights_svals", "inverse_moment", shape="long", export_format="list"
        )
        spectra = {r["param_name"]: r for r in rows if r["field"] == "weights_svals"}
        run = spectra["half"]["run_id"]
        assert spectra["double"]["run_id"] == run
        assert spectra["double"]["value_array"] == [1.0 + 2**-32, 2.0]
        assert spectra["half"]["value_array"] == [2.0, 4.0]

    with Session(profile="local") as session:
        header = session.results.describe_run(run)
        assert header["requested"] == {"allow_cuda": False, "precision": "preserve"}
        assert header["arithmetic_status"] == "recorded"
        facts = {f["parameter_name"]: f for f in header["arithmetic"]}
        for name, original, result in [
            ("half", "float16", "float32"),
            ("double", "float64", "float64"),
        ]:
            entry = facts[name]
            assert actual[tuple(inputs[name].diagonal())] == (
                entry["working_dtype"],
                entry["driver"],
            )
            assert (
                entry["origin"],
                entry["checkpoint_dtype"],
                entry["input_dtype"],
                entry["working_dtype"],
                entry["output_dtype"],
                entry["device"],
                entry["backend"],
                entry["driver"],
            ) == (
                "observed",
                original,
                original,
                result,
                result,
                "cpu",
                "scipy",
                "gesdd",
            )
        selected = session.filter(param_names=["half"]).results.describe_run(run)
        assert selected["arithmetic"] == [facts["half"]]
        with pytest.raises(SessionError, match="No observation"):
            session.results.describe_run(run, parameter_uid="absent")
        (moment,) = [
            r
            for r in rows
            if r["field"] == "inverse_moment" and r["param_name"] == "half"
        ]
        assert moment["value_scalar"] == pytest.approx(5 / 16)
        record = session.results.describe_run(
            moment["run_id"], parameter_uid=moment["parameter_uid"]
        )
        (entry,) = record["arithmetic"]
        assert (
            entry["origin"],
            entry["input_dtype"],
            entry["working_dtype"],
            entry["backend"],
            entry["driver"],
        ) == ("self-reported", "float32", "float64", "numpy", None)
        session.compute.apply("frob_norm")
        ordinary = session.results.export(
            "frob_norm", shape="long", export_format="list"
        )[0]
        selected = session.filter(param_names=[ordinary["param_name"]]).results
        assert (
            selected.describe_run(ordinary["run_id"])["arithmetic_status"] == "unknown"
        )
        with pytest.raises(SessionError, match="scope"):
            session.filter(model_ids=["absent"]).results.describe_run(
                ordinary["run_id"]
            )
        unknown = session.results.describe_run(None)
        assert (
            unknown["requested"],
            unknown["arithmetic_status"],
            unknown["arithmetic"],
        ) == (None, "unknown", [])


def test_bfloat_admission_and_random_svd_provenance():
    torch = pytest.importorskip("torch")
    state = {
        "layer.weight": torch.tensor([[1, 2, 0], [3, -1, 2]], dtype=torch.bfloat16)
    }
    with Session(profile="ram") as session:
        session.models.add(state, "model")
        for kernel in ("weights_rand_svd", "weights_rand_vectors"):
            session.compute.configure_kernel(
                kernel, allow_cuda=False, precision="preserve"
            )
        fields = (
            "weights_rand",
            "weights_rand_svals",
            "weights_rand_lsvs",
            "weights_rand_rsvs",
        )
        session.compute.apply(*fields)
        (values,) = session.results.export_metrics(*fields, export_format="list")
        u, s, v = (
            values[f]
            for f in ("weights_rand_lsvs", "weights_rand_svals", "weights_rand_rsvs")
        )
        np.testing.assert_allclose(
            (u * s) @ v.T, values["weights_rand"], rtol=2e-6, atol=2e-6
        )
        rows = session.results.export(*fields[1:], shape="long", export_format="list")
        assert len(rows) == 3
        for row in rows:
            (fact,) = session.results.describe_run(row["run_id"])["arithmetic"]
            assert fact["checkpoint_dtype"].endswith("bfloat16")
            assert (
                fact["input_dtype"],
                fact["working_dtype"],
                fact["output_dtype"],
                np.dtype(row["value_dtype"]).name,
            ) == ("float32", "float32", "float32", "float32")
        with pytest.raises(InvalidConfiguration, match="precision"):
            session.compute.configure_kernel(
                "weights_rand_vectors", precision="float16"
            )


def test_svd_cache_status_tracks_the_resolved_device(monkeypatch):
    torch = pytest.importorskip("torch")
    with Session(profile="ram") as session:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        session.compute.configure_kernel(
            "weights_svd", allow_cuda=True, precision="preserve"
        )
        session.models.add({"w": np.eye(3)}, model_id="net")
        session.compute.apply("weights_svals")
        (cpu,) = session.results.status("weights_svals")
        assert cpu.status == "fresh"
        monkeypatch.setattr(torch.version, "cuda", "12.8")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        (gpu,) = session.results.status("weights_svals")
        assert gpu.status == "stale"


def test_precision_override_promotes_half_and_rejects_float32_overflow():
    with Session(profile="ram") as session:
        session.models.add({"half": np.diag([2, 3]).astype(np.float16)}, "half_model")
        session.compute.configure_kernel(
            "weights_svd", allow_cuda=False, precision="float64"
        )
        session.compute.apply("weights_svals")
        (row,) = session.results.export_metrics("weights_svals", export_format="list")
        np.testing.assert_array_equal(
            row["weights_svals"], np.array([2.0, 3.0]), strict=True
        )
        session.models.add({"huge": np.diag([1e100, 1.0])}, "large_model")
        session.compute.configure_kernel("weights_svd", precision="float32")
        with pytest.raises(SessionError, match="range of float32"):
            session.filter(model_ids=["large_model"]).compute.apply("weights_svals")


def test_torch_solver_transport_preserves_double_for_single_and_tuple_outputs():
    torch = pytest.importorskip("torch")
    from diffract.core.compute.extensions.utils import torch_cuda_wrapper

    matrix = np.array([[1 + 2**-32, 2], [3, 4]], dtype=np.float64)

    def calculate(tensor):
        assert tensor.dtype == torch.float64
        return tensor.T, tensor + 1

    left, right = torch_cuda_wrapper(matrix, calculate, device="cpu")
    np.testing.assert_array_equal(left, matrix.T, strict=True)
    np.testing.assert_array_equal(right, matrix + 1, strict=True)
    single = torch_cuda_wrapper(matrix, lambda tensor: tensor, device="cpu")
    np.testing.assert_array_equal(single, matrix, strict=True)


_PROCESS = """
import json
import sys

import numpy as np
from diffract import Session

out = []
for policy in sys.argv[1:]:
    with Session(profile="local") as session:
        if not session.models.list():
            session.models.add({"w": np.diag([1.0 + 2**-25, 3.0])}, "net")
        if policy != "inherit":
            session.compute.configure_kernel(
                "weights_svd", precision=policy, allow_cuda=False
            )
        session.compute.apply("min_weights_sval")
        rows = session.results.export(
            "min_weights_sval", "weights_svals", shape="long", export_format="list"
        )
        by_field = {row["field"]: row for row in rows}
        spectrum = by_field["weights_svals"]
        tail = by_field["min_weights_sval"]
        out.append([
            spectrum["run_id"], tail["run_id"],
            np.dtype(spectrum["value_dtype"]).name, tail["value_scalar"]
        ])
print(json.dumps(out))
"""


@pytest.mark.parametrize(
    ("first", "second"), [("float32", "float64"), ("float64", "float32")]
)
def test_precision_rebuilds_dependencies_in_fresh_processes(tmp_path, first, second):
    def run(*policies):
        source = Path(__file__).resolve().parents[3] / "src"
        result = subprocess.run(  # noqa: S603 - fixed interpreter and test program
            [sys.executable, "-c", _PROCESS, *policies],
            cwd=tmp_path,
            env=os.environ | {"PYTHONPATH": str(source)},
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    initial, inherited = run(first, "inherit")
    cached, changed, inherited_again = run(first, second, "inherit")
    (restored,) = run(first)
    assert initial == inherited == cached
    assert changed == inherited_again
    assert [r[2] for r in (initial, changed, restored)] == [first, second, first]
    assert len({r[0] for r in (initial, changed, restored)}) == 3
    assert len({r[1] for r in (initial, changed, restored)}) == 3
    for row in (initial, changed, restored):
        assert row[3] == (1.0 if row[2] == "float32" else 1.0 + 2**-25)
