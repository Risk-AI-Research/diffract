"""Unit tests for Session public API."""

from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
from threading import Barrier
from unittest import TestCase

import numpy as np
import pytest

from diffract.containers import WiringConfiguration, create_main_container
from diffract.core import WEIGHTS_FIELD
from diffract.core.compute import KernelConfig
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import (
    DuplicateParameterError,
    InvalidIdentifierError,
    KernelNotFoundError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    Session,
    SessionError,
)

pytestmark = pytest.mark.unit


def _write_ram_config(path: Path) -> None:
    # Use a unique metadata path based on the config file name
    metadata_path = path.parent / f"{path.stem}_metadata.db"
    path.write_text(
        f"""
[storage]
backend = ram

[metadata]
backend = sqlite

[metadata.sqlite]
path = {metadata_path}

[cache]
backend = simple

[cache.simple]
max_memory_mb = 16
ttl_seconds = 3600
key_prefix = "test:cache:"

[parallel.thread_pool]
max_workers = 1

[parallel.process_pool]
max_workers = 1

[export]
default_export_format = dict

[nn.extractor]
skip_not_implemented_types = true
""".strip()
        + "\n"
    )


def _session_with_sum_metric(temp_dir: Path) -> tuple[Session, ParameterMetadata]:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p0", name="layer.0.weight", ptype=ParameterType.DENSE, model_id="m1"
    )
    proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    proxy.set_field(WEIGHTS_FIELD, np.arange(6, dtype=np.float32).reshape(2, 3))

    registry = container.compute_singleton.kernel_registry()
    if not registry.can_produce_field("sum_metric"):

        def sum_metric(w: np.ndarray) -> float:
            return float(np.sum(w))

        _, cfg_dict = registry._split_signature(sum_metric)
        registry.register_kernel(
            name="sum_metric",
            require_fields=("weights",),
            produce_fields=("sum_metric",),
            implementation=sum_metric,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=cfg_dict,
        )

    return Session(container=container), meta


def test_session_compute_and_get_results_roundtrip(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    # Pre-populate one parameter in storage so Session loads it at init.
    meta = ParameterMetadata(
        uid="p0",
        name="layer.0.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
    )
    proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    proxy.set_field(WEIGHTS_FIELD, weights)

    # Register a simple kernel producing a new scalar field.
    registry = container.compute_singleton.kernel_registry()

    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    req_auto, cfg_dict = registry._split_signature(w_sum)
    assert req_auto == ("w",)

    registry.register_kernel(
        name="w_sum",
        require_fields=("weights",),
        produce_fields=("w_sum",),
        implementation=w_sum,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_dict,
    )

    session = Session(container=container)
    session.compute.apply("w_sum")

    scalars = session.results.export_metrics("w_sum", export_format="dict")
    assert meta.uid in scalars
    assert scalars[meta.uid]["fields"]["w_sum"] == float(np.sum(weights))


def test_apply_returns_plane_classified_summary(temp_dir: Path) -> None:
    session, _ = _session_with_sum_metric(temp_dir)

    summary = session.compute.apply("sum_metric")

    assert "sum_metric" in summary.parameter_fields
    assert summary.in_model_fields == ()
    assert summary.cross_model_fields == ()
    assert summary.skipped == ()


def test_apply_reports_skipped_when_nothing_written(temp_dir: Path) -> None:
    session, _ = _session_with_sum_metric(temp_dir)

    session.compute.apply("sum_metric")
    summary = session.compute.apply("sum_metric")

    assert summary.parameter_fields == ()
    assert [name for name, _ in summary.skipped] == ["sum_metric"]


def test_results_erase_returns_summary(temp_dir: Path) -> None:
    session, _ = _session_with_sum_metric(temp_dir)
    session.compute.apply("sum_metric")

    summary = session.results.erase("sum_metric")

    assert summary.fields == ("sum_metric",)
    assert summary.affected_uids == 1


def test_results_erase_counts_only_entries_that_held_the_field(temp_dir: Path) -> None:
    """affected_uids reports entries that lose data, not the scope size."""
    session, _ = _session_with_sum_metric(temp_dir)
    session.models.add({"layer.w": np.ones((2, 2), dtype=np.float32)}, model_id="m2")

    session.filter(model_ids=["m1"]).compute.apply("sum_metric")
    summary = session.results.erase("sum_metric")

    assert summary.fields == ("sum_metric",)
    assert summary.affected_uids == 1


def test_models_erase_returns_summary(temp_dir: Path) -> None:
    session, _ = _session_with_sum_metric(temp_dir)

    summary = session.models.erase("m1")

    assert summary.models == ("m1",)
    assert summary.affected_uids == 1


def test_configure_kernel_returns_effective_config(temp_dir: Path) -> None:
    session, _ = _session_with_sum_metric(temp_dir)

    config = session.compute.configure_kernel("sum_metric")

    assert isinstance(config, KernelConfig)


def test_session_compute_raises_on_unknown_kernel(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(KernelNotFoundError, match="Cannot produce"):
        session.compute.apply("does_not_exist")


def test_session_patch_meta_strict_conflict_raises(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p_meta",
        name="layer.0.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
        other_meta={"k": 1},
    )
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )

    session = Session(container=container)
    with pytest.raises(SessionError, match="Metadata conflicts"):
        session.models.parameters.patch_meta(updates={"p_meta": {"k": 2}}, force=False)


def test_session_patch_meta_force_overwrites_and_persists(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p_meta2",
        name="layer.0.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
        other_meta={"k": 1},
    )
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )

    session = Session(container=container)
    session.models.parameters.patch_meta(
        updates={"p_meta2": {"k": 2, "new_key": "v"}}, force=True
    )

    # New Session should load updated metadata from storage.
    session2 = Session(container=container)
    info = [
        param
        for param in session2.models.parameters.list(verbose=True)
        if param["uid"] == "p_meta2"
    ]
    assert info[0]["other_meta"]["k"] == 2
    assert info[0]["other_meta"]["new_key"] == "v"


def test_session_ingest_fields_strict_conflict_raises(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p_fields", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )
    storage.set_field(meta.uid, "a", 1, table=TABLE_PARAMETERS)

    session = Session(container=container)
    with pytest.raises(SessionError, match="Field conflicts"):
        session.results.ingest_metrics({meta.uid: {"a": 2}}, force=False)


def test_session_ingest_fields_force_overwrites(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p_fields2", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )
    storage.set_field(meta.uid, "a", 1, table=TABLE_PARAMETERS)

    session = Session(container=container)
    session.results.ingest_metrics({meta.uid: {"a": 2}}, force=True)
    assert storage.get_field(meta.uid, "a", table=TABLE_PARAMETERS) == 2


def test_session_ingest_fields_simple_field(temp_dir: Path) -> None:
    """Test ingesting simple scalar fields."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="p_ctx", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    ParameterDataProxy.create_and_store(meta=meta, repository=repository)

    session = Session(container=container)
    session.results.ingest_metrics({meta.uid: {"metric": 123}}, force=False)
    result = session.results.export_metrics("metric", export_format="dict")
    assert result[meta.uid]["fields"]["metric"] == 123


def test_session_draw_validates_inputs(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="exactly one"):
        session.viz.draw()


@pytest.mark.skipif(
    not import_utils.is_available("torch"),
    reason="torch required for add/erase model test",
)
def test_session_add_and_erase_model_id(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    nn = import_utils.require("torch.nn")

    model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.ReLU(), nn.Linear(2, 2))

    session.models.add(model, model_id="m_add")
    assert session.models.list() == ["m_add"]

    with pytest.raises(SessionError, match="already exists"):
        session.models.add(model, model_id="m_add")

    session.models.erase("m_add")
    assert session.models.list() == []


def test_session_add_rejects_model_with_no_extractable_parameters(
    temp_dir: Path,
) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    biases_only = {"layer.bias": np.zeros(4)}
    with pytest.raises(SessionError, match="produced no parameters"):
        session.models.add(biases_only, model_id="bias_model")

    with pytest.raises(SessionError, match="produced no parameters"):
        session.models.add({}, model_id="empty_model")

    assert session.models.list() == []


def test_session_add_numpy_dict_end_to_end(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    rng = np.random.default_rng(0)
    weights = {
        "encoder.weight": rng.random((16, 8)),
        "decoder.weight": rng.random((8, 16)),
    }

    session.models.add(weights, model_id="np_model")
    assert session.models.list() == ["np_model"]

    session.compute.apply("frob_norm")
    results = session.results.export_metrics("frob_norm", export_format="dict")

    assert len(results) == 2
    for entry in results.values():
        assert entry["fields"]["frob_norm"] > 0


def test_session_erase_results_can_remove_dependents(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    meta = ParameterMetadata(
        uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    proxy.set_field(WEIGHTS_FIELD, np.ones((2, 2), dtype=np.float32))

    registry = container.compute_singleton.kernel_registry()

    def base(w: np.ndarray) -> float:
        return float(np.sum(w))

    def final(a: float) -> float:
        return a + 1.0

    _, cfg_base = registry._split_signature(base)
    _, cfg_final = registry._split_signature(final)

    registry.register_kernel(
        name="base",
        require_fields=("weights",),
        produce_fields=("a",),
        implementation=base,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_base,
    )
    registry.register_kernel(
        name="final",
        require_fields=("a",),
        produce_fields=("b",),
        implementation=final,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_final,
    )

    session = Session(container=container)
    session.compute.apply("b")
    assert storage.has_field(meta.uid, "a", table=TABLE_PARAMETERS)
    assert storage.has_field(meta.uid, "b", table=TABLE_PARAMETERS)

    session.results.erase("a", erase_dependent_also=True)
    assert not storage.has_field(meta.uid, "a", table=TABLE_PARAMETERS)
    assert not storage.has_field(meta.uid, "b", table=TABLE_PARAMETERS)


def test_session_merge_migrates_fields_and_handles_conflicts(temp_dir: Path) -> None:
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    container_a.storage.storage_manager()
    container_a.cache.cache_manager()
    container_a.nn.parameter_repository()
    container_b.storage.storage_manager()
    container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # Session A starts empty. Session B has two parameters with computed fields.
    meta_b1 = ParameterMetadata(
        uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p_b1 = ParameterDataProxy.create_and_store(
        meta=meta_b1,
        repository=repository_b,
    )
    p_b1.set_field("foo", 999)
    p_b1.set_field("bar", 3)

    meta_b2 = ParameterMetadata(
        uid="b2", name="w2", ptype=ParameterType.DENSE, model_id="m2"
    )
    p_b2 = ParameterDataProxy.create_and_store(
        meta=meta_b2,
        repository=repository_b,
    )
    p_b2.set_field("z", 7)

    # Rewire containers before creating Sessions (DI wiring is global).
    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)
    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)

    WiringConfiguration.wire(container_a)
    session_a.utils.merge_other_session(session_b, verify=True)

    # New parameters created and their computed fields migrated.
    assert set(session_a.models.list()) == {"m1", "m2"}
    got_bar = session_a.results.export_metrics("bar", export_format="dict")
    assert meta_b1.uid in got_bar
    assert got_bar[meta_b1.uid]["fields"]["bar"] == 3

    got = session_a.results.export_metrics("z", export_format="dict")
    assert meta_b2.uid in got
    assert got[meta_b2.uid]["fields"]["z"] == 7


def test_session_merge_respects_field_allowlist_and_chunks(temp_dir: Path) -> None:
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    # Ensure storages are clean (defensive against cross-test contamination).
    container_a.storage.storage_manager().clear()
    container_b.storage.storage_manager().clear()

    container_b.storage.storage_manager()
    container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # Two params with multiple fields to ensure we hit chunking and allowlist.
    meta_b1 = ParameterMetadata(
        uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p_b1 = ParameterDataProxy.create_and_store(
        meta=meta_b1,
        repository=repository_b,
    )
    p_b1.set_field("foo", 1)
    p_b1.set_field("bar", 2)

    meta_b2 = ParameterMetadata(
        uid="b2", name="w2", ptype=ParameterType.DENSE, model_id="m2"
    )
    p_b2 = ParameterDataProxy.create_and_store(
        meta=meta_b2,
        repository=repository_b,
    )
    p_b2.set_field("foo", 3)
    p_b2.set_field("bar", 4)

    # Rewire containers before creating Sessions (DI wiring is global).
    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)
    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)

    # Force very small budget to ensure multiple chunks.
    WiringConfiguration.wire(container_a)
    session_a.utils.merge_other_session(
        session_b, fields=["foo"], verify=True, read_budget_bytes=1
    )

    got_foo = session_a.results.export_metrics("foo", export_format="dict")
    assert got_foo[meta_b1.uid]["fields"]["foo"] == 1
    assert got_foo[meta_b2.uid]["fields"]["foo"] == 3

    storage_a = container_a.storage.storage_manager()
    assert not storage_a.has_field(meta_b1.uid, "bar")
    assert not storage_a.has_field(meta_b2.uid, "bar")


def test_session_merge_does_not_overwrite_existing_fields_when_verify_true(
    temp_dir: Path,
) -> None:
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    # Ensure storages are clean (defensive against cross-test contamination).
    container_a.storage.storage_manager().clear()
    container_b.storage.storage_manager().clear()

    storage_a = container_a.storage.storage_manager()
    container_a.cache.cache_manager()
    repository_a = container_a.nn.parameter_repository()
    storage_b = container_b.storage.storage_manager()
    container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # A already has (model_id, name) with field 'bar'.
    meta_a = ParameterMetadata(
        uid="a1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p_a = ParameterDataProxy.create_and_store(
        meta=meta_a,
        repository=repository_a,
    )
    p_a.set_field("bar", 111)

    # B has same (model_id, name) but different uid and a different value for 'bar'.
    meta_b = ParameterMetadata(
        uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p_b = ParameterDataProxy.create_and_store(
        meta=meta_b,
        repository=repository_b,
    )
    p_b.set_field("bar", 222)
    p_b.set_field("foo", 333)
    assert storage_b.has_field(meta_b.uid, "foo", table=TABLE_PARAMETERS)

    # Rewire containers before creating Sessions (DI wiring is global).
    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)
    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)
    assert repository_b.storage_manager.has_field(
        meta_b.uid, "foo", table=TABLE_PARAMETERS
    )
    with repository_b:
        view_b = repository_b.create_view()
        fields_b = view_b.list_fields_by_uid()[meta_b.uid]
    assert "foo" in fields_b

    WiringConfiguration.wire(container_a)
    session_a.utils.merge_other_session(session_b, verify=True)

    # Existing field should remain unchanged; new field should be migrated.
    # Also ensure we did not create a new parameter with the source uid.
    assert not storage_a.has_field(meta_b.uid, "__metadata__", table=TABLE_PARAMETERS)
    assert storage_a.get_field(meta_a.uid, "bar", table=TABLE_PARAMETERS) == 111
    assert storage_a.get_field(meta_a.uid, "foo", table=TABLE_PARAMETERS) == 333


def test_patch_meta_invalid_structure_raises(temp_dir: Path) -> None:
    """patch_meta with non-dict value should raise SessionError."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    repository = container.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="p_invalid",
        name="layer.0.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
    )
    ParameterDataProxy.create_and_store(meta=meta, repository=repository)

    session = Session(container=container)
    with pytest.raises(SessionError, match="expected dict"):
        session.models.parameters.patch_meta(
            updates={"p_invalid": "not_a_dict"}, force=False
        )


def test_erase_models_neither_args_nor_flag_raises(temp_dir: Path) -> None:
    """erase_models with no args and erase_all=False should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="No model_ids provided"):
        session.models.erase()


def test_erase_models_both_args_and_flag_raises(temp_dir: Path) -> None:
    """erase_models with both args and erase_all=True should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="Cannot specify both"):
        session.models.erase("model1", erase_all=True)


def test_erase_results_neither_args_nor_flag_raises(temp_dir: Path) -> None:
    """erase_results with no args and erase_all=False should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="No fields_to_erase provided"):
        session.results.erase()


def test_erase_results_both_args_and_flag_raises(temp_dir: Path) -> None:
    """erase_results with both args and erase_all=True should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="Cannot specify both"):
        session.results.erase("field1", erase_all=True)


def test_patch_meta_unknown_uid_raises(temp_dir: Path) -> None:
    """patch_meta with unknown UID should raise SessionError."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(SessionError, match="Unknown parameter UIDs"):
        session.models.parameters.patch_meta(
            updates={"nonexistent_uid": {"key": "value"}}, force=False
        )


def test_ingest_fields_unknown_uid_raises(temp_dir: Path) -> None:
    """ingest_fields with unknown UID should raise SessionError."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(SessionError, match="Unknown parameter UIDs"):
        session.results.ingest_metrics({"nonexistent_uid": {"field": 1}}, force=False)


def test_session_get_results_returns_structured_export(temp_dir: Path) -> None:
    """Test that export_metrics returns scalars dict."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    repository = container.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    p.set_field("scalar_field", 42)

    session = Session(container=container)
    result = session.results.export_metrics("scalar_field", export_format="dict")

    assert result[meta.uid]["fields"]["scalar_field"] == 42


def test_session_erase_results_removes_scalar_field(temp_dir: Path) -> None:
    """Test that erase_results removes scalar fields correctly."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()

    # Create a parameter with weights
    meta = ParameterMetadata(
        uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    p = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    p.set_field(WEIGHTS_FIELD, np.ones((2, 2), dtype=np.float32))

    # Register a simple kernel
    registry = container.compute_singleton.kernel_registry()

    def compute_metric(w: np.ndarray) -> float:
        return float(np.sum(w))

    _, cfg_k = registry._split_signature(compute_metric)
    registry.register_kernel(
        name="metric",
        require_fields=("weights",),
        produce_fields=("metric",),
        implementation=compute_metric,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_k,
    )

    session = Session(container=container)

    session.compute.apply("metric")
    assert storage.has_field(meta.uid, "metric", table=TABLE_PARAMETERS)

    session.results.erase("metric")

    assert not storage.has_field(meta.uid, "metric", table=TABLE_PARAMETERS)


def test_session_list_aggregates_returns_empty_initially(temp_dir: Path) -> None:
    """Test that export_aggregates for a non-ingested field returns empty."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))
    # export_aggregates on a field with nothing ingested yields an empty list.
    aggregates = session.results.export_aggregates("l_overlap", export_format="list")
    assert aggregates == []


def test_session_ingest_aggregates_creates_aggregates(temp_dir: Path) -> None:
    """Test that ingest_aggregates creates aggregates correctly."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2"):
        session.models.add({"layer.weight": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "context_params": ("layer.weight",),
                "value": np.eye(3),
            }
        ]
    )

    aggregates = session.results.export_aggregates("l_overlap", export_format="list")
    assert len(aggregates) == 1
    assert aggregates[0]["field"] == "l_overlap"
    assert aggregates[0]["context_models"] == ("m1", "m2")
    assert aggregates[0]["context_params"] == ("layer.weight",)


def test_session_ingest_aggregates_canonicalizes_context_order(
    temp_dir: Path,
) -> None:
    """Unsorted ingest context is stored in the uid's canonical (sorted) order."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2"):
        session.models.add(
            {"a.weight": np.eye(2), "b.weight": np.eye(2)}, model_id=model
        )
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m2", "m1"),
                "context_params": ("b.weight", "a.weight"),
                "value": np.eye(3),
            }
        ]
    )

    aggregates = session.results.export_aggregates("l_overlap", export_format="list")
    assert len(aggregates) == 1
    assert aggregates[0]["context_models"] == ("m1", "m2")
    assert aggregates[0]["context_params"] == ("a.weight", "b.weight")


def test_session_ingest_aggregates_conflict_raises(temp_dir: Path) -> None:
    """Test that ingest_aggregates raises on conflict when force=False."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2"):
        session.models.add({"w": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(3),
            }
        ]
    )

    with pytest.raises(SessionError, match="conflicts"):
        session.results.ingest_aggregates(
            [
                {
                    "field_name": "l_overlap",
                    "context_models": ("m1", "m2"),
                    "value": np.eye(4),
                }
            ],
            force=False,
        )


def test_session_ingest_aggregates_force_overwrites(temp_dir: Path) -> None:
    """Test that ingest_aggregates with force=True overwrites existing."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2"):
        session.models.add({"w": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(3),
            }
        ]
    )

    new_value = np.ones((4, 4))
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": new_value,
            }
        ],
        force=True,
    )

    aggregates = session.results.export_aggregates("l_overlap", export_format="list")
    assert len(aggregates) == 1
    assert np.array_equal(aggregates[0]["value"], new_value)


def test_session_erase_results_removes_aggregates(temp_dir: Path) -> None:
    """Test that erase_results removes aggregate entries by field_name."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    registry = container.compute_singleton.kernel_registry()

    # Register a dummy kernel producing the field so it can be erased; the
    # registry is a process-wide singleton, so the name must not shadow a
    # real kernel — a fake "l_overlap" would leak into every later test.
    registry.register_kernel(
        name="dummy_overlap",
        require_fields=(),
        produce_fields=("dummy_overlap",),
        implementation=lambda: None,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config={},
    )

    session = Session(container=container)

    for model in ("m1", "m2", "m3"):
        session.models.add({"w": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "dummy_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(3),
            },
            {
                "field_name": "other_field",
                "context_models": ("m1", "m3"),
                "value": np.ones((2, 2)),
            },
        ]
    )

    assert (
        len(session.results.export_aggregates("dummy_overlap", export_format="list"))
        == 1
    )
    assert (
        len(session.results.export_aggregates("other_field", export_format="list")) == 1
    )

    session.results.erase("dummy_overlap")

    aggregates = session.results.export_aggregates("other_field", export_format="list")
    assert len(aggregates) == 1
    assert aggregates[0]["field"] == "other_field"


def test_session_erase_models_removes_aggregates(temp_dir: Path) -> None:
    """Test that erase_models removes aggregates with model in context_models."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    repository = container.nn.parameter_repository()

    # Create parameters for m1 and m2
    meta_m1 = ParameterMetadata(
        uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1"
    )
    ParameterDataProxy.create_and_store(meta=meta_m1, repository=repository)

    meta_m2 = ParameterMetadata(
        uid="p2", name="w", ptype=ParameterType.DENSE, model_id="m2"
    )
    ParameterDataProxy.create_and_store(meta=meta_m2, repository=repository)

    session = Session(container=container)

    session.models.add({"w": np.eye(2)}, model_id="m3")
    session.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(3),
            },
            {
                "field_name": "r_overlap",
                "context_models": ("m2", "m3"),
                "value": np.ones((2, 2)),
            },
        ]
    )

    assert (
        len(session.results.export_aggregates("l_overlap", export_format="list")) == 1
    )
    assert (
        len(session.results.export_aggregates("r_overlap", export_format="list")) == 1
    )

    session.models.erase("m1")

    aggregates = session.results.export_aggregates("r_overlap", export_format="list")
    assert len(aggregates) == 1
    assert aggregates[0]["field"] == "r_overlap"


def _seed_three_models(cfg: Path) -> Session:
    """Seed a session with three single-parameter numpy models (a, b, c)."""
    session = Session(container=create_main_container(cfg))
    rng = np.random.default_rng(0)
    for model_id in ("a", "b", "c"):
        session.models.add({"w.weight": rng.random((4, 4))}, model_id=model_id)
    return session


def test_erase_from_filtered_context_rejects_out_of_scope_model(
    temp_dir: Path,
) -> None:
    """A filtered scope is a sandbox: erasing a model outside the scope must
    raise rather than silently destroy it or its aggregates."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = _seed_three_models(cfg)

    session.results.ingest_aggregates(
        [
            {
                "field_name": "bc_overlap",
                "context_models": ("b", "c"),
                "value": np.ones((2, 2)),
            },
        ]
    )

    scoped = session.filter(model_ids=["a"])
    with pytest.raises(ModelNotFoundError, match="'b'"):
        scoped.models.erase("b")

    # No data loss: model 'b' and its aggregate both survive the rejected erase.
    assert session.models.list() == ["a", "b", "c"]
    assert (
        len(session.results.export_aggregates("bc_overlap", export_format="list")) == 1
    )


def test_erase_from_filtered_context_honors_scope(temp_dir: Path) -> None:
    """An in-scope erase from a filtered context removes only the scoped model;
    erase_all from the same context is likewise confined to the scope."""
    cfg_single = temp_dir / "single.ini"
    _write_ram_config(cfg_single)
    session = _seed_three_models(cfg_single)
    session.filter(model_ids=["a"]).models.erase("a")
    assert session.models.list() == ["b", "c"]

    cfg_all = temp_dir / "all.ini"
    _write_ram_config(cfg_all)
    session = _seed_three_models(cfg_all)
    session.filter(model_ids=["a", "b"]).models.erase(erase_all=True)
    assert session.models.list() == ["c"]


def test_session_merge_copies_aggregates(temp_dir: Path) -> None:
    """Test that merge copies aggregates from source session."""
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    from diffract.containers import WiringConfiguration

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)
    for model in ("m1", "m2"):
        session_b.models.add({"w": np.eye(2)}, model_id=model)
    session_b.results.ingest_aggregates(
        [
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(3),
            }
        ]
    )
    assert (
        len(session_b.results.export_aggregates("l_overlap", export_format="list")) == 1
    )

    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)
    assert session_a.results.export_aggregates("l_overlap", export_format="list") == []

    session_a.utils.merge_other_session(session_b, verify=False)

    aggregates = session_a.results.export_aggregates("l_overlap", export_format="list")
    assert len(aggregates) == 1
    assert aggregates[0]["field"] == "l_overlap"


def test_session_list_aggregates_filters_by_field_name(temp_dir: Path) -> None:
    """Test that export_aggregates returns only requested field."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2"):
        session.models.add({"w": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {"field_name": "l_overlap", "context_models": ("m1", "m2"), "value": 1},
            {"field_name": "r_overlap", "context_models": ("m1", "m2"), "value": 2},
        ]
    )

    filtered = session.results.export_aggregates("l_overlap", export_format="list")
    assert len(filtered) == 1
    assert filtered[0]["field"] == "l_overlap"


def test_session_list_aggregates_filters_by_model_ids(temp_dir: Path) -> None:
    """Test that export_aggregates returns every context_models entry for a field."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    for model in ("m1", "m2", "m3", "m4"):
        session.models.add({"w": np.eye(2)}, model_id=model)
    session.results.ingest_aggregates(
        [
            {"field_name": "overlap", "context_models": ("m1", "m2"), "value": 1},
            {"field_name": "overlap", "context_models": ("m3", "m4"), "value": 2},
        ]
    )

    # New API has no model_ids filter on export; we get all aggregates for the field
    filtered = session.results.export_aggregates("overlap", export_format="list")
    assert len(filtered) == 2
    context_models_set = {frozenset(a["context_models"]) for a in filtered}
    assert frozenset(("m1", "m2")) in context_models_set
    assert frozenset(("m3", "m4")) in context_models_set


def _two_model_session_with_scope_sum(temp_dir: Path) -> Session:
    """Build a ram Session with two single-parameter models and a computed field.

    Both models carry one DENSE weight; a ``scope_sum`` kernel produces one
    scalar per parameter. Returns the session after that field is computed.
    """
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    container.cache.cache_manager()

    for model_id, uid in (("a", "pa"), ("b", "pb")):
        meta = ParameterMetadata(
            uid=uid,
            name="layer.weight",
            ptype=ParameterType.DENSE,
            model_id=model_id,
        )
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
        proxy.set_field(WEIGHTS_FIELD, np.arange(6, dtype=np.float32).reshape(2, 3))

    registry = container.compute_singleton.kernel_registry()

    def scope_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    _req, cfg_dict = registry._split_signature(scope_sum)
    registry.register_kernel(
        name="scope_sum",
        require_fields=("weights",),
        produce_fields=("scope_sum",),
        implementation=scope_sum,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_dict,
    )

    session = Session(container=container)
    session.compute.apply("scope_sum")
    return session


def test_filtered_export_then_session_erase_stays_consistent(temp_dir: Path) -> None:
    """A scoped export must not corrupt the shared cache into an erase crash.

    ``filter(...).results.export_metrics(...)`` seeds the session-shared
    field cache with only the in-scope subset; a subsequent whole-session
    ``erase`` visits every uid, including those the cache never saw, and
    must treat the cache as a possibly-partial optimization — never as the
    uid universe.
    """
    session = _two_model_session_with_scope_sum(temp_dir)

    # Seed the shared cache with only model 'a'.
    session.filter(model_ids=["a"]).results.export_metrics("scope_sum")

    # Whole-session erase must complete without raising.
    session.results.erase("scope_sum")

    # And it must have actually erased the field for BOTH models — no
    # mutate-then-raise leaving the session in a half-applied state.
    remaining = session.results.export_metrics("scope_sum", export_format="dict")
    for uid, entry in remaining.items():
        assert "scope_sum" not in entry["fields"], uid


def test_filtered_export_does_not_corrupt_list_verbose(temp_dir: Path) -> None:
    """A scoped export must not make out-of-scope fields vanish from list().

    After a scoped export seeds the shared cache with only the in-scope
    subset, ``models.parameters.list(verbose=True)`` must still report the
    real ``available_fields`` of every out-of-scope parameter — the partial
    cache must be completed, not trusted.
    """
    session = _two_model_session_with_scope_sum(temp_dir)

    # Seed the shared cache with only model 'a'.
    session.filter(model_ids=["a"]).results.export_metrics("scope_sum")

    rows = {r["uid"]: r for r in session.models.parameters.list(verbose=True)}

    # The out-of-scope parameter must still report its real computed field.
    assert "scope_sum" in rows["pb"]["available_fields"]
    assert "scope_sum" in rows["pa"]["available_fields"]


def test_global_model_list_and_rename_touch_only_instances(temp_dir: Path) -> None:
    cfg = temp_dir / "identity.ini"
    _write_ram_config(cfg)
    cfg.write_text(
        cfg.read_text().replace(
            "backend = ram",
            f"backend = sqlite\n\n[storage.sqlite]\npath = {temp_dir / 'values.db'}",
        )
    )
    container = create_main_container(cfg)
    session = Session(container=container)
    weights = np.arange(6, dtype=float).reshape(2, 3)
    with session:
        session.models.add({"a": weights, "b": weights + 1}, model_id="Encoder")
        session.results.ingest_aggregates(
            [
                {
                    "field_name": "summary",
                    "context_models": ("Encoder",),
                    "value": np.ones((2, 2)),
                }
            ]
        )
        repo = container.nn.parameter_repository()
        before = {p.meta.name: p.meta.uid for p in repo}
        instance = repo.instances.get_by_name("Encoder")
        connection = repo.metadata_index._conn
        reads = []
        writes = []

        def track(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ:
                reads.append(table)
            if action in (
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_DELETE,
            ):
                writes.append((action, table))
            return sqlite3.SQLITE_OK

        changes = connection.total_changes
        connection.set_authorizer(track)
        try:
            assert session.models.list() == ["Encoder"]
            assert "instances" in reads
            assert "parameters" not in reads
            session.models.rename("Encoder", "encoder")
        finally:
            connection.set_authorizer(None)
        assert connection.total_changes - changes == 1
        assert writes == [(sqlite3.SQLITE_UPDATE, "instances")]
        assert (
            repo.instances.get_by_name("encoder").instance_uid == instance.instance_uid
        )
        assert {p.meta.model_id for p in repo} == {"encoder"}

    other = create_main_container(cfg)
    reopened = Session(container=other)
    assert reopened.filter(model_ids=["Encoder"]).models.list() == []
    assert reopened.filter(model_ids=["re:enc.*"]).models.list() == ["encoder"]
    assert {p["name"]: p["uid"] for p in reopened.models.parameters.list()} == before
    with reopened:
        for proxy in other.nn.parameter_repository():
            np.testing.assert_array_equal(
                proxy.get_field(WEIGHTS_FIELD), weights + (proxy.meta.name == "b")
            )


def test_concurrent_model_admission_keeps_one_complete_model(temp_dir: Path) -> None:
    cfg = temp_dir / "race.ini"
    _write_ram_config(cfg)
    first = Session(container=create_main_container(cfg))
    second = Session(container=create_main_container(cfg))
    ready = Barrier(2)

    def add(session, name):
        ready.wait(timeout=10)
        try:
            session.models.add({name: np.ones((2, 2))}, model_id="shared")
        except ModelAlreadyExistsError:
            return None
        return name

    with first, second, ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(add, (first, second), ("left", "right")))
    assert results.count(None) == 1
    (winner,) = [name for name in results if name is not None]
    reopened = Session(container=create_main_container(cfg))
    assert reopened.models.list() == ["shared"]
    assert [p["name"] for p in reopened.models.parameters.list()] == [winner]


def _admit_checkpoint_from_process(marker, pipe):
    with pipe, Session(profile="local") as session:
        pipe.send("ready")
        assert pipe.recv() == "add"
        try:
            session.models.add(
                {"shared": np.eye(3) * marker, f"from{marker}": np.eye(3) + marker},
                model_id="model",
                step=4,
                display_label=f"writer {marker}",
            )
        except ModelAlreadyExistsError:
            pipe.send(None)
        else:
            pipe.send(marker)


def test_processes_racing_on_one_coordinate_publish_one_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    with Session(profile="local") as original:
        original.models.add({"shared": np.eye(3)}, model_id="model")
    spawn = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        pipes = []
        workers = []
        for value in (2, 7):
            parent, child = spawn.Pipe()
            pipes.append(stack.enter_context(parent))
            with child:
                proc = spawn.Process(
                    target=_admit_checkpoint_from_process, args=(value, child)
                )
                proc.start()
            workers.append(proc)
            stack.callback(proc.join)
            stack.callback(proc.terminate)
        for pipe in pipes:
            assert pipe.poll(30)
            assert pipe.recv() == "ready"
        for pipe in pipes:
            pipe.send("add")
        outcomes = []
        for pipe in pipes:
            assert pipe.poll(30)
            outcomes.append(pipe.recv())
        for proc in workers:
            proc.join(timeout=30)
            assert proc.exitcode == 0
    assert outcomes.count(None) == 1
    (winner,) = [m for m in outcomes if m is not None]
    with Session(profile="local") as reopened:
        rows = reopened.results.export_metrics(WEIGHTS_FIELD, export_format="dict")
        fields = {
            (p["step"], p["name"], p["step_label"]): rows[p["uid"]]["fields"][
                WEIGHTS_FIELD
            ]
            for p in reopened.models.parameters.list(verbose=True)
        }
        label = f"writer {winner}"
        assert fields.keys() == {
            (0, "shared", None),
            (4, "shared", label),
            (4, f"from{winner}", label),
        }
        np.testing.assert_array_equal(fields[0, "shared", None], np.eye(3))
        np.testing.assert_array_equal(fields[4, "shared", label], np.eye(3) * winner)
        np.testing.assert_array_equal(
            fields[4, f"from{winner}", label], np.eye(3) + winner
        )

    expected = {(uid, WEIGHTS_FIELD) for uid in rows}
    root = tmp_path / ".diffract" / "sqlite"
    with closing(
        sqlite3.connect((root / "metadata_index.db").as_uri() + "?mode=ro", uri=True)
    ) as db:
        assert set(db.execute("SELECT obj_uid,field FROM field_values")) == expected
    with closing(
        sqlite3.connect((root / "sqlite_storage.db").as_uri() + "?mode=ro", uri=True)
    ) as db:
        assert set(db.execute("SELECT table_name,obj_uid,field FROM storage")) == {
            ("", uid, field) for uid, field in expected
        }


def test_empty_add_and_scoped_erase_preserve_instance_lifetime(temp_dir: Path) -> None:
    cfg = temp_dir / "lifetime.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))
    with pytest.raises(SessionError, match="produced no parameters"):
        session.models.add({"a": np.ones(1)}, model_id="net")
    assert session.models.list() == []
    session.models.add({"a": np.eye(2), "b": np.eye(2)}, model_id="net")
    session.filter(param_names=["a"]).models.erase("net")
    assert session.models.list() == ["net"]
    assert [p["name"] for p in session.models.parameters.list()] == ["b"]
    session.models.erase("net")
    assert session.models.list() == []


def test_merge_by_coordinate_preserves_target_uid_and_verified_fields(
    temp_dir: Path,
) -> None:
    src_cfg = temp_dir / "source.ini"
    dst_cfg = temp_dir / "target.ini"
    _write_ram_config(src_cfg)
    _write_ram_config(dst_cfg)
    src = Session(config_path=src_cfg)
    dst_container = create_main_container(dst_cfg)
    dst = Session(container=dst_container)
    with src, dst:
        src.models.add({"w": np.eye(2) * 3}, model_id="a")
        dst.models.add({"w": np.eye(2)}, model_id="a")
        source_uid = src.models.parameters.list()[0]["uid"]
        target_uid = dst.models.parameters.list()[0]["uid"]
        assert source_uid != target_uid
        assert len(bytes.fromhex(target_uid)) == 16
        target = dst_container.nn.parameter_repository()
        dst.utils.merge_other_session(src, verify=True)
        np.testing.assert_array_equal(
            target.get_proxy(target_uid).get_field(WEIGHTS_FIELD), np.eye(2)
        )
        for _ in range(2):
            dst.utils.merge_other_session(src, verify=False)
        assert [p["uid"] for p in dst.models.parameters.list()] == [target_uid]
        np.testing.assert_array_equal(
            target.get_proxy(target_uid).get_field(WEIGHTS_FIELD), np.eye(2) * 3
        )


def test_wrapped_parameter_coordinates_and_provenance(temp_dir: Path) -> None:
    cfg = temp_dir / "names.ini"
    _write_ram_config(cfg)
    container = create_main_container(cfg)
    session = Session(container=container)
    raw = "module._orig_mod._fsdp_wrapped_module._checkpoint_wrapped_module.0.weight"
    with session, TestCase().assertLogs("diffract", level="INFO") as logs:
        session.models.add(
            {
                "net._extra_state": {"epoch": 4},
                raw: np.ones((2, 2)),
                "net.module.weight": np.zeros((2, 2)),
                "0.Weight": np.eye(2),
            },
            model_id="Net",
        )
        params = {p.meta.name: p.meta for p in container.nn.parameter_repository()}
        assert set(params) == {"0.weight", "net.module.weight", "0.Weight"}
        assert params["0.weight"].other_meta["in_model_idx"] == 1
        assert params["0.weight"].other_meta["raw_name"] == raw
        assert {p.other_meta["extractor"] for p in params.values()} == {
            "NumpyDictExtractor"
        }
        assert "net._extra_state" in "\n".join(logs.output)
        assert "Stripped wrapper prefixes" in "\n".join(logs.output)


def test_normalized_collision_reports_coordinate_and_rolls_back(temp_dir: Path) -> None:
    cfg = temp_dir / "collision.ini"
    _write_ram_config(cfg)
    container = create_main_container(cfg)
    session = Session(container=container)
    with session:
        with pytest.raises(DuplicateParameterError) as caught:
            session.models.add(
                {"module.weight": np.ones((2, 2)), "weight": np.eye(2)}, model_id="A"
            )
        assert caught.value.name == "weight"
        assert caught.value.step == 0
        assert len(caught.value.instance_uid) == 32
        assert caught.value.instance_uid in str(caught.value)
        assert session.models.list() == []
        assert session.models.parameters.list() == []
        assert (
            container.storage.storage_manager().list_objs(table=TABLE_PARAMETERS) == []
        )
        session.models.add({"weight": np.eye(2)}, model_id="A")


@pytest.mark.parametrize("name", [".root", "a..weight", "trailing.", "module..w"])
def test_rejects_empty_tensor_path_components(temp_dir: Path, name: str) -> None:
    cfg = temp_dir / "empty-name.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))
    with pytest.raises(InvalidIdentifierError, match="empty path component"):
        session.models.add({name: np.eye(2)}, model_id="A")
    assert session.models.list() == []


def test_nested_scopes_keep_their_own_session_after_a_second_session_opens():
    target_config = create_main_container(profile="ram")
    target_config.export.config.default_export_format.override("dict")
    with Session(container=target_config) as target:
        target.models.add(
            {"w": np.eye(4, dtype=np.float32)}, model_id="same", display_label="first"
        )
        donor_config = create_main_container(profile="ram")
        donor_config.export.config.default_export_format.override("list")
        with Session(container=donor_config) as donor:
            donor.models.add(
                {"w": 3 * np.eye(4, dtype=np.float32)},
                model_id="same",
                display_label="second",
            )
            with target as reentered:
                assert reentered is target
            scope = target.filter(model_ids=["same"])
            with scope as scoped:
                assert scoped is scope
                scoped.compute.apply("param_norm")
                (aggregate,) = scoped.results.export_aggregates(
                    "param_norm", export_format="list"
                )
                assert aggregate["value"] == pytest.approx(4)
                with scoped.filter(param_names=["w"]) as nested:
                    (param,) = nested.models.parameters.list(verbose=True)
                    assert param["uid"] == target.models.parameters.list()[0]["uid"]
                    assert param["step_label"] == "first"
                    nested.results.ingest_metrics({param["uid"]: {"observed": 17}})
                    nested.results.ingest_aggregates(
                        [
                            {
                                "field_name": "stored_summary",
                                "context_models": ("same",),
                                "context_params": ("w",),
                                "value": 27,
                            }
                        ]
                    )
                    (stored,) = target.results.export_aggregates(
                        "stored_summary", export_format="list"
                    )
                    assert stored["value"] == 27
                    values = nested.results.export_metrics("observed")
                    assert values[param["uid"]]["fields"]["observed"] == 17
            assert (
                "observed"
                not in donor.models.parameters.list(verbose=True)[0]["available_fields"]
            )
            assert (
                donor.results.export_aggregates("param_norm", export_format="list")
                == []
            )
            donor.compute.apply("param_norm")
            (aggregate,) = donor.results.export_aggregates(
                "param_norm", export_format="list"
            )
            assert aggregate["value"] == pytest.approx(36)


def test_reapply_repairs_one_partial_svd_without_rewriting_complete_parameters():
    with Session(profile="ram") as session:
        weights = np.diag(np.array([6.0, 3.0, 1.0], dtype=np.float32))
        session.models.add({"left": weights, "right": weights * 2}, model_id="net")
        session.compute.apply("weights_svals")
        (right,) = session.filter(param_names=["right"]).models.parameters.list()
        session.results.ingest_metrics(
            {right["uid"]: {"weights_svals": np.array([11.0, 12.0, 13.0])}}, force=True
        )
        left = session.filter(param_names=["left"])
        summary = left.results.erase("weights_svals")
        assert summary.fields == ("weights_lsvs", "weights_rsvs", "weights_svals")
        assert summary.affected_uids == 1
        (parameter,) = left.models.parameters.list(verbose=True)
        assert not set(summary.fields).intersection(parameter["available_fields"])
        present = session.results.export_metrics("weights_svals", export_format="dict")
        assert [
            r["metadata"]["name"]
            for r in present.values()
            if "weights_svals" in r["fields"]
        ] == ["right"]
        session.compute.apply("weights_svals")
        repaired = session.results.export_metrics("weights_svals", export_format="dict")
        values = {
            r["metadata"]["name"]: r["fields"]["weights_svals"]
            for r in repaired.values()
        }
        np.testing.assert_allclose(values["left"], [1, 3, 6])
        np.testing.assert_allclose(values["right"], [11, 12, 13])


def test_results_erase_accepts_ingested_fields_and_lists_stored_names():
    from diffract.session import FieldNotFoundError

    with Session(profile="ram") as session:
        session.models.add({"w": np.eye(2)}, model_id="net")
        (parameter,) = session.models.parameters.list()
        session.results.ingest_metrics({parameter["uid"]: {"observed_loss": 3}})

        with pytest.raises(FieldNotFoundError, match="Did you mean: observed_loss"):
            session.results.erase("observed_los")

        summary = session.results.erase("observed_loss")
        assert summary.fields == ("observed_loss",)
        assert summary.affected_uids == 1
        assert session.results.erase("weights").fields == ("weights",)
        session.results.ingest_metrics({parameter["uid"]: {"score": 9}})
        summary = session.results.erase(erase_all=True)
        assert summary.fields == ("score",)
        assert summary.affected_uids == 1
        (parameter,) = session.models.parameters.list(verbose=True)
        assert not parameter["available_fields"]


def test_results_erase_keeps_other_aggregate_groups():
    from diffract.core.compute import KernelApplyLevel

    with Session(profile="ram") as session:
        session.models.add({"w": np.eye(2)}, model_id="first")
        session.models.add({"w": np.eye(3)}, model_id="second")

        @session.compute.kernel(
            require_fields=("weights",),
            produce_fields=("group_sum", "group_negated_sum"),
            apply_level=KernelApplyLevel.IN_MODEL,
        )
        def aggregate_pair(weights: list[np.ndarray]) -> tuple[float, float]:
            total = sum(float(value.sum()) for value in weights)
            return total, -total

        session.compute.apply("group_sum")
        first = session.filter(model_ids=["first"])
        summary = first.results.erase("group_sum")
        assert summary.fields == ("group_negated_sum", "group_sum")
        rows = session.results.export_aggregates(
            "group_sum", "group_negated_sum", export_format="list"
        )
        assert {
            (row["field"], tuple(row["context_models"]), row["value"]) for row in rows
        } == {
            ("group_sum", ("second",), 3.0),
            ("group_negated_sum", ("second",), -3.0),
        }
        first.compute.apply("group_sum")
        rows = first.results.export_aggregates(
            "group_sum", "group_negated_sum", export_format="list"
        )
        assert {row["field"]: row["value"] for row in rows} == {
            "group_sum": 2.0,
            "group_negated_sum": -2.0,
        }


@pytest.mark.parametrize("bulk", [True, False])
def test_export_reflects_another_sessions_new_erased_and_overwritten_fields(
    tmp_path, monkeypatch, bulk
):
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    if not bulk:
        monkeypatch.setattr(CatalogStorageManager, "live_pointers", None)
    monkeypatch.chdir(tmp_path)
    with Session(profile="local") as reader:
        reader.models.add({"a": np.eye(3), "b": np.eye(3)}, model_id="m")
        ids = {p["name"]: p["uid"] for p in reader.models.parameters.list()}
        reader.results.ingest_metrics(
            {ids["a"]: {"frob_norm": 1}, ids["b"]: {"frob_norm": 2}}
        )
        reader.results.ingest_metrics({uid: {"observed": 5} for uid in ids.values()})
        reader.models.parameters.list(verbose=True)
        first = reader.results.export_metrics("frob_norm", export_format="dict")
        assert first[ids["a"]]["fields"]["frob_norm"] == 1
        second = create_main_container(profile="local")
        with Session(container=second) as writer:
            writer.filter(param_names=["a"]).results.erase("frob_norm")
            second.nn.parameter_repository().get_proxy(ids["a"]).erase_field("observed")
            writer.results.ingest_metrics(
                {ids["a"]: {"imported": 7}, ids["b"]: {"frob_norm": 9}}, force=True
            )
            refreshed = reader.results.export_metrics(
                "frob_norm", "imported", export_format="dict"
            )
            assert refreshed[ids["a"]]["fields"] == {"imported": 7}
            assert refreshed[ids["b"]]["fields"] == {"frob_norm": 9}
            with pytest.raises(KernelNotFoundError, match="observed"):
                reader.filter(param_names=["a"]).results.export_metrics("observed")


def test_catalog_export_reads_only_current_members_of_the_view(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    container = create_main_container(profile="local")
    with Session(container=container) as session:
        session.models.add({"a": np.eye(2)}, model_id="one")
        (a,) = session.models.parameters.list()
        repo = container.nn.parameter_repository()
        earlier = repo.create_view()
        assert earlier.list_uids() == [a["uid"]]
        session.models.add({"b": np.eye(2)}, model_id="two")
        (b,) = session.filter(model_ids=["two"]).models.parameters.list()
        session.results.ingest_metrics({a["uid"]: {"score": 3}, b["uid"]: {"score": 4}})
        repo.storage_manager.set_field("0" * 32, "score", 99, table=TABLE_PARAMETERS)
        statements = []
        repo.metadata_index._conn.set_trace_callback(statements.append)

        def catalog_query():
            (selection,) = [sql for sql in statements if "live_field_values" in sql]
            statements.clear()
            return selection

        rows = session.results.export_metrics("score", export_format="dict")
        assert {uid: row["fields"] for uid, row in rows.items()} == {
            a["uid"]: {"score": 3},
            b["uid"]: {"score": 4},
        }
        full = catalog_query()
        assert a["uid"] not in full
        assert b["uid"] not in full
        assert earlier.read_catalog_fields(("score",)) == {a["uid"]: {"score": 3}}
        snapshot = catalog_query()
        assert a["uid"] in snapshot
        assert b["uid"] not in snapshot
        selected = session.filter(model_ids=["two"]).results.export_metrics("score")
        assert {uid: r["fields"] for uid, r in selected.items()} == {
            b["uid"]: {"score": 4}
        }
        scoped = catalog_query()
        assert b["uid"] in scoped
        assert a["uid"] not in scoped
        repo.metadata_index._conn.set_trace_callback(None)


def test_derived_catalog_keeps_its_owner_selection_contract(tmp_path, monkeypatch):
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    class SelectedCatalog(CatalogStorageManager):
        def live_pointers(self, uids, fields, *, table):
            return super().live_pointers(
                tuple(dict.fromkeys(uids)), fields, table=table
            )

    monkeypatch.chdir(tmp_path)
    container = create_main_container(profile="local")
    with Session(container=container) as session:
        repo = container.nn.parameter_repository()
        native = repo.storage_manager
        selected = SelectedCatalog(native.backend, native.catalog.index)
        monkeypatch.setattr(repo, "_storage_manager", selected)
        session.models.add({"weights": np.eye(3)}, model_id="network")
        (param,) = session.models.parameters.list()
        session.results.ingest_metrics({param["uid"]: {"observed": 12}})
        rows = session.results.export_metrics("observed", export_format="dict")
        assert rows[param["uid"]]["fields"] == {"observed": 12}


def test_direct_exporter_resolves_individual_revisions_with_warm_field_cache(
    tmp_path, monkeypatch
):
    from diffract.core.export.exporters import ResultExporter
    from diffract.core.export.formatters.dict_formatter import DictFormatter
    from diffract.core.storage.catalog_manager import CatalogStorageManager
    from diffract.session.field_cache import SessionFieldCache

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(CatalogStorageManager, "live_pointers", None)
    container = create_main_container(profile="local")
    with Session(container=container) as reader:
        reader.models.add({"a": np.eye(3)}, model_id="m")
        (param,) = reader.models.parameters.list()
        uid = param["uid"]
        reader.results.ingest_metrics({uid: {"first": 7}})
        view = container.nn.parameter_repository().create_view()
        cache = SessionFieldCache()
        cache.set({uid: ["first"]})

        def export():
            return (
                ResultExporter()
                .export_results(
                    "first",
                    "second",
                    parameters=view,
                    formatter=DictFormatter(),
                    field_cache=cache,
                )
                .scalars[uid]["fields"]
            )

        assert export() == {"first": 7}
        other = create_main_container(profile="local")
        with Session(container=other) as writer:
            other.nn.parameter_repository().get_proxy(uid).erase_field("first")
            writer.results.ingest_metrics({uid: {"second": 9}})
            assert export() == {"second": 9}


def test_export_uses_durable_value_when_optional_cache_write_fails(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    container = create_main_container(profile="local")
    with Session(container=container) as session:
        session.models.add({"a": np.eye(3)}, model_id="m")
        (param,) = session.models.parameters.list()
        uid = param["uid"]
        session.results.ingest_metrics({uid: {"observed": 41}})
        cache = container.nn.parameter_repository().cache_manager
        cache.erase_field(uid, "observed")

        def unavailable(*args, **kwargs):
            raise OSError("cache unavailable")

        monkeypatch.setattr(cache, "set_field", unavailable)
        actual = session.results.export_metrics("observed", export_format="dict")
        assert actual[uid]["fields"] == {"observed": 41}
