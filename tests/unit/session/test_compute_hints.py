"""Unit tests for compute namespace metric listing and error hints."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from diffract.containers import compute_singleton_container
from diffract.core.compute.registry import KernelRegistry
from diffract.session import (
    KernelNotFoundError,
    ScopeValidationError,
    Session,
    SessionError,
)

pytestmark = pytest.mark.unit

_COMPUTE_LOGGER = "diffract.session.namespaces.compute"


@contextmanager
def _capture_logs(logger_name: str) -> Iterator[list[str]]:
    """Capture INFO messages from a named diffract logger.

    The package loggers set ``propagate = false`` with their own handler, so
    pytest's ``caplog`` (rooted at the root logger) never sees them; attach a
    handler to the exact module logger instead.
    """
    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _ListHandler(level=logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)


def _seed_models(container, model_ids: tuple[str, ...]) -> Session:
    """Seed a session with one single-parameter numpy model per id."""
    session = Session(container=container)
    rng = np.random.default_rng(0)
    for model_id in model_ids:
        session.models.add({"w.weight": rng.random((4, 4))}, model_id=model_id)
    return session


def test_list_available_metrics_returns_fields(ram_container) -> None:
    session = Session(container=ram_container)

    metrics = session.compute.list_available_metrics()

    assert "frob_norm" in metrics
    assert "stable_rank" in metrics


def test_list_available_metrics_verbose_names_producing_kernel(ram_container) -> None:
    session = Session(container=ram_container)

    verbose = session.compute.list_available_metrics(verbose=True)

    assert any(entry.startswith("frob_norm <- ") for entry in verbose)


def test_apply_unknown_field_suggests_close_match(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*frob_norm"):
        session.compute.apply("frobnorm")


def test_apply_unknown_field_points_to_metric_listing(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match="list_available_metrics"):
        session.compute.apply("definitely_not_a_metric")


def test_configure_unknown_kernel_suggests_close_match(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*hard_rank"):
        session.compute.configure_kernel("hard_rankk", threshold=1e-6)


# ---------------- CROSS_MODEL BINARY scope validation ----------------


def test_apply_binary_cross_model_three_models_raises_scope_error(
    ram_container,
) -> None:
    """l_overlap is a binary cross-model kernel: applying it with three models
    in scope must raise an actionable error, not silently write nothing."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with pytest.raises(ScopeValidationError) as excinfo:
        session.compute.apply("l_overlap")

    message = str(excinfo.value)
    assert "l_overlap" in message
    for model_id in ("a", "b", "c"):
        assert model_id in message
    # The recommendation is filter(model_ids=[...]); an unshipped pairs=
    # parameter must not be advertised.
    assert "filter(model_ids=" in message
    assert "pairs=" not in message
    # filter() returns a SessionContext, so the runnable form goes through
    # the compute namespace, never a bare .apply on the context.
    assert ".compute.apply(" in message
    assert ").apply(" not in message


def test_apply_transitive_binary_cross_model_raises_same_scope_error(
    ram_container,
) -> None:
    """A dependent metric (l_agreement -> l_overlap) must raise the same typed,
    actionable error rather than a raw KeyError from deep in storage."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with pytest.raises(ScopeValidationError) as excinfo:
        session.compute.apply("l_agreement")

    assert isinstance(excinfo.value, SessionError)
    message = str(excinfo.value)
    assert "l_overlap" in message  # names the offending dependency
    assert "l_agreement" in message  # names the requested field
    assert "filter(model_ids=" in message
    assert "pairs=" not in message


def test_apply_binary_cross_model_two_models_computes_end_to_end(
    ram_container,
) -> None:
    """A valid model pair is not over-blocked: the full chain computes and both
    l_overlap and its dependent l_agreement are stored."""
    session = _seed_models(ram_container, ("a", "b"))

    session.compute.apply("l_agreement")

    overlaps = session.results.export_aggregates("l_overlap", export_format="list")
    agreements = session.results.export_aggregates("l_agreement", export_format="list")
    assert {(row["model_source"], row["model_target"]) for row in overlaps} == {
        ("a", "b"),
        ("b", "a"),
    }
    assert {(row["model_source"], row["model_target"]) for row in agreements} == {
        ("a", "b"),
        ("b", "a"),
    }


def test_apply_does_not_log_success_when_scope_invalid(ram_container) -> None:
    """The false 'Successfully produced fields' log must not fire when the scope
    is rejected."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with (
        _capture_logs(_COMPUTE_LOGGER) as messages,
        pytest.raises(ScopeValidationError),
    ):
        session.compute.apply("l_overlap")

    assert not any("Successfully produced fields" in m for m in messages)


def test_apply_success_log_reports_only_written_fields(ram_container) -> None:
    """Re-applying an already-computed field writes nothing; the log must say so
    rather than falsely claim success."""
    session = _seed_models(ram_container, ("a", "b"))
    session.compute.apply("frob_norm")

    with _capture_logs(_COMPUTE_LOGGER) as messages:
        session.compute.apply("frob_norm")

    assert not any("Successfully produced fields" in m for m in messages)
    assert any("No new fields produced" in m for m in messages)


def test_session_kernel_version_tracks_effective_configuration(ram_container) -> None:
    session = Session(container=ram_container)

    @session.compute.kernel(version="2.1")
    def versioned_weight_sum(weights: np.ndarray, scale: int = 1) -> float:
        return float(weights.sum()) * scale

    registry = ram_container.compute.kernel_registry()
    initial = registry.get_kernel_config_identity("versioned_weight_sum")
    session.compute.configure_kernel("versioned_weight_sum", scale=2)
    updated = registry.get_kernel_config_identity("versioned_weight_sum")

    assert initial.kernel_version == updated.kernel_version == "2.1"
    assert initial.config_json == '{"scale":1}'
    assert updated.config_json == '{"scale":2}'
    assert updated.config_hash != initial.config_hash


@pytest.mark.parametrize("resident_limit", [3, 2048])
def test_parameter_kernel_resolves_only_scoped_inputs_in_bulk(
    ram_container, monkeypatch: pytest.MonkeyPatch, resident_limit: int
) -> None:
    from diffract.core.compute.execution import parameter_runner
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    ram_container.cache.config.backend.from_value("none")
    session = Session(container=ram_container)
    for model_id, count in (("inside", 2), ("outside", 9)):
        session.models.add(
            {
                f"layer{i}.weight": np.full((2, 2), count + i, dtype=np.float32)
                for i in range(3)
            },
            model_id=model_id,
        )

    @session.compute.kernel(require_fields=("weights",), produce_fields=("weight_sum",))
    def weight_sum(weights: np.ndarray) -> float:
        return float(weights.sum())

    selections = []
    sizes = []
    original = CatalogStorageManager.live_pointers
    original_sizes = CatalogStorageManager.read_sizes

    def capture(storage, uids, fields, *, table):
        selections.append((set(uids), tuple(fields), table))
        return original(storage, uids, fields, table=table)

    def capture_sizes(storage, required, *, table):
        sizes.append(set(required))
        return original_sizes(storage, required, table=table)

    scoped = session.filter(model_ids=["inside"])
    with monkeypatch.context() as apply_patch:
        apply_patch.setattr(CatalogStorageManager, "live_pointers", capture)
        apply_patch.setattr(CatalogStorageManager, "read_sizes", capture_sizes)
        apply_patch.setattr(parameter_runner, "MAX_RESIDENT_ARGUMENTS", resident_limit)
        scoped.compute.apply("weight_sum")
    metrics = scoped.results.export_metrics("weight_sum", export_format="dict")
    assert selections == [(set(metrics), ("weights",), "parameters")]
    assert sizes == ([set(metrics)] if resident_limit == 3 else [])
    assert sorted(row["fields"]["weight_sum"] for row in metrics.values()) == [
        8,
        12,
        16,
    ]


@pytest.mark.parametrize("resident_limit", [3, 2048])
def test_in_model_run_resolves_exact_batch_and_keeps_member_values(
    ram_container, monkeypatch: pytest.MonkeyPatch, resident_limit: int
) -> None:
    from diffract.core.compute.execution import aggregation_runner
    from diffract.core.compute.execution.enums import KernelApplyLevel
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    ram_container.cache.config.backend.from_value("none")
    session = Session(container=ram_container)
    for model_id, counts in (("left", (2, 3)), ("right", (11, 17))):
        session.models.add(
            {
                f"layer{index}.weight": np.full((2, 2), count, dtype=np.float32)
                for index, count in enumerate(counts)
            },
            model_id=model_id,
        )

    @session.compute.kernel(
        require_fields=("weights",),
        produce_fields=("total_weight",),
        apply_level=KernelApplyLevel.IN_MODEL,
    )
    def total_weight(weights: list[np.ndarray]) -> float:
        return float(sum(weight.sum() for weight in weights))

    scopes = []
    sizes = []
    original = CatalogStorageManager.live_pointers
    original_sizes = CatalogStorageManager.read_sizes

    def capture(storage, uids, fields, *, table):
        scopes.append((set(uids), tuple(fields), table))
        return original(storage, uids, fields, table=table)

    def capture_sizes(storage, required, *, table):
        sizes.append(set(required))
        return original_sizes(storage, required, table=table)

    with monkeypatch.context() as apply_patch:
        apply_patch.setattr(CatalogStorageManager, "live_pointers", capture)
        apply_patch.setattr(CatalogStorageManager, "read_sizes", capture_sizes)
        apply_patch.setattr(
            aggregation_runner, "_MAX_RESIDENT_ARGUMENTS", resident_limit
        )
        session.compute.apply("total_weight")
    exported = session.results.export_aggregates("total_weight", export_format="dict")
    assert len(scopes) == 1
    assert len(scopes[0][0]) == 4
    assert scopes[0][1:] == (("weights",), "parameters")
    assert sizes == ([scopes[0][0]] if resident_limit == 3 else [])
    assert sorted(row["value"] for row in exported) == [20, 112]


@pytest.fixture
def custom_kernel_session():
    with (
        compute_singleton_container.kernel_registry.override(KernelRegistry()),
        Session(profile="ram") as session,
    ):
        yield session


def test_missing_kernel_input_is_reported_before_any_value_read(
    custom_kernel_session, monkeypatch
):
    from diffract.core.data.nn.params.proxy import ParameterDataProxy
    from diffract.session.errors import FieldNotFoundError

    session = custom_kernel_session
    for model_id in ("a", "b"):
        session.models.add({"w": np.diag([3.0, 5.0])}, model_id=model_id)
    second = session.filter(model_ids=["b"])
    (other,) = second.models.parameters.list()
    second.results.ingest_metrics({other["uid"]: {"sample_count": 24}})
    first = session.filter(model_ids=["a"])
    (owner,) = first.models.parameters.list()
    first.results.ingest_metrics({owner["uid"]: {"matrix": np.arange(24)}})

    @session.compute.kernel()
    def matrix_mean(matrix):
        return float(matrix.mean())

    @session.compute.kernel()
    def combined(matrix_mean, weights_svals):
        return matrix_mean + weights_svals.max()

    with monkeypatch.context() as patch:

        def reject_read(*args, **kwargs):
            pytest.fail("A value was read during input validation")

        patch.setattr(ParameterDataProxy, "read_resolved_field", reject_read)
        with pytest.raises(FieldNotFoundError) as error:
            session.compute.apply("frob_norm", "combined")
    for part in (
        "matrix_mean",
        "'matrix'",
        "weights",
        "ingest_metrics",
        "missing this input: 1",
        "sample_count, weights",
    ):
        assert part in str(error.value)
    assert all(
        s.status == "never computed" for s in session.results.status("frob_norm")
    )

    first.compute.apply("matrix_mean")
    (result,) = first.results.export_metrics("matrix_mean", export_format="list")
    assert result["matrix_mean"] == 11.5

    @session.compute.kernel()
    def matrix(weights):
        return weights + 3.0

    second.compute.apply("matrix_mean")
    (result,) = second.results.export_metrics(
        "weights", "matrix_mean", export_format="list"
    )
    assert result["matrix_mean"] == pytest.approx(result["weights"].mean() + 3.0)


@pytest.mark.parametrize("coverage", [False, True])
def test_pair_input_is_required_only_on_joined_parameters(
    custom_kernel_session, coverage
):
    from diffract.core.compute.execution import KernelApplyLevel
    from diffract.session.errors import FieldNotFoundError

    session = custom_kernel_session
    for model in ("a", "b", "c"):
        session.models.add({"shared": np.eye(2), model: np.eye(2)}, model_id=model)
    shared = session.filter(param_names=["shared"])
    owners = {p["model_id"]: p["uid"] for p in shared.models.parameters.list()}
    shared.results.ingest_metrics(
        {owners["a"]: {"score": 2.0}, owners["b"]: {"score": 9.0}}
    )

    @session.compute.kernel(apply_level=KernelApplyLevel.CROSS_MODEL)
    def delta(score):
        return score[1] - score[0]

    pairs = [("a", "b"), ("a", "c")]
    with pytest.raises(FieldNotFoundError, match="'delta' requires input 'score'"):
        session.compute.apply("delta", pairs=pairs, return_coverage=coverage)
    assert session.results.export_aggregates("delta", export_format="list") == []

    shared.results.ingest_metrics({owners["c"]: {"score": 15.0}})
    session.compute.apply("delta", pairs=pairs, return_coverage=coverage)
    rows = session.results.export_aggregates("delta", export_format="list")
    assert sorted(row["value"] for row in rows) == [7.0, 13.0]
