"""Unit tests for Session public API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffract.containers import WiringConfiguration, create_main_container
from diffract.core import WEIGHTS_FIELD
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import KernelNotFoundError, Session, SessionError

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


def test_session_compute_and_get_results_roundtrip(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

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
    session.compute("w_sum")

    result = session.get_results("w_sum", export_format="dict")
    scalars = result.scalars
    assert meta.uid in scalars
    assert scalars[meta.uid]["fields"]["w_sum"] == float(np.sum(weights))


def test_session_compute_raises_on_unknown_kernel(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(KernelNotFoundError, match="Cannot produce"):
        session.compute("does_not_exist")


def test_session_patch_meta_strict_conflict_raises(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

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
        session.patch_meta(updates={"p_meta": {"k": 2}}, force=False)


def test_session_patch_meta_force_overwrites_and_persists(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

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
    session.patch_meta(updates={"p_meta2": {"k": 2, "new_key": "v"}}, force=True)

    # New Session should load updated metadata from storage.
    session2 = Session(container=container)
    info = session2.list_parameters(parameter_uids=["p_meta2"], verbose=True)
    assert info[0]["other_meta"]["k"] == 2
    assert info[0]["other_meta"]["new_key"] == "v"


def test_session_ingest_fields_strict_conflict_raises(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

    meta = ParameterMetadata(uid="p_fields", name="w", ptype=ParameterType.DENSE, model_id="m1")
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )
    storage.set_field(meta.uid, "a", 1, table=TABLE_PARAMETERS)

    session = Session(container=container)
    with pytest.raises(SessionError, match="Field conflicts"):
        session.ingest_fields({meta.uid: {"a": 2}}, force=False)


def test_session_ingest_fields_force_overwrites(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

    meta = ParameterMetadata(uid="p_fields2", name="w", ptype=ParameterType.DENSE, model_id="m1")
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )
    storage.set_field(meta.uid, "a", 1, table=TABLE_PARAMETERS)

    session = Session(container=container)
    session.ingest_fields({meta.uid: {"a": 2}}, force=True)
    assert storage.get_field(meta.uid, "a", table=TABLE_PARAMETERS) == 2


def test_session_ingest_fields_simple_field(temp_dir: Path) -> None:
    """Test ingesting simple scalar fields."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()

    meta = ParameterMetadata(uid="p_ctx", name="w", ptype=ParameterType.DENSE, model_id="m1")
    ParameterDataProxy.create_and_store(meta=meta, repository=repository)

    session = Session(container=container)
    session.ingest_fields({meta.uid: {"metric": 123}}, force=False)
    result = session.get_results("metric", export_format="dict")
    assert result.scalars[meta.uid]["fields"]["metric"] == 123


def test_session_draw_validates_inputs(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="exactly one"):
        session.draw()


def test_session_add_and_erase_model_id(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    torch = import_utils.require("torch")
    nn = import_utils.require("torch.nn")

    model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.ReLU(), nn.Linear(2, 2))

    session.add(model, model_id="m_add")
    assert session.list_models() == ["m_add"]

    with pytest.raises(SessionError, match="already exists"):
        session.add(model, model_id="m_add")

    session.erase_models("m_add")
    assert session.list_models() == []


def test_session_erase_results_can_remove_dependents(temp_dir: Path) -> None:
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()
    cache = container.cache.cache_manager()

    meta = ParameterMetadata(uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1")
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
    session.compute("b")
    assert storage.has_field(meta.uid, "a", table=TABLE_PARAMETERS)
    assert storage.has_field(meta.uid, "b", table=TABLE_PARAMETERS)

    session.erase_results("a", erase_dependent_also=True)
    assert not storage.has_field(meta.uid, "a", table=TABLE_PARAMETERS)
    assert not storage.has_field(meta.uid, "b", table=TABLE_PARAMETERS)


def test_session_merge_migrates_fields_and_handles_conflicts(temp_dir: Path) -> None:
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    storage_a = container_a.storage.storage_manager()
    cache_a = container_a.cache.cache_manager()
    repository_a = container_a.nn.parameter_repository()
    storage_b = container_b.storage.storage_manager()
    cache_b = container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # Session A starts empty. Session B has two parameters with computed fields.
    meta_b1 = ParameterMetadata(uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1")
    p_b1 = ParameterDataProxy.create_and_store(
        meta=meta_b1,
        repository=repository_b,
    )
    p_b1.set_field("foo", 999)
    p_b1.set_field("bar", 3)

    meta_b2 = ParameterMetadata(uid="b2", name="w2", ptype=ParameterType.DENSE, model_id="m2")
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
    session_a.merge(session_b, verify=True)

    # New parameters created and their computed fields migrated.
    assert set(session_a.list_models()) == {"m1", "m2"}
    result_bar = session_a.get_results("bar", export_format="dict")
    got_bar = result_bar.scalars
    assert meta_b1.uid in got_bar
    assert got_bar[meta_b1.uid]["fields"]["bar"] == 3

    result_z = session_a.get_results("z", export_format="dict")
    got = result_z.scalars
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

    storage_b = container_b.storage.storage_manager()
    cache_b = container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # Two params with multiple fields to ensure we hit chunking and allowlist.
    meta_b1 = ParameterMetadata(uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1")
    p_b1 = ParameterDataProxy.create_and_store(
        meta=meta_b1,
        repository=repository_b,
    )
    p_b1.set_field("foo", 1)
    p_b1.set_field("bar", 2)

    meta_b2 = ParameterMetadata(uid="b2", name="w2", ptype=ParameterType.DENSE, model_id="m2")
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
    session_a.merge(session_b, fields=["foo"], verify=True, read_budget_bytes=1)

    result_foo = session_a.get_results("foo", export_format="dict")
    got_foo = result_foo.scalars
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
    cache_a = container_a.cache.cache_manager()
    repository_a = container_a.nn.parameter_repository()
    storage_b = container_b.storage.storage_manager()
    cache_b = container_b.cache.cache_manager()
    repository_b = container_b.nn.parameter_repository()

    # A already has (model_id, name) with field 'bar'.
    meta_a = ParameterMetadata(uid="a1", name="w", ptype=ParameterType.DENSE, model_id="m1")
    p_a = ParameterDataProxy.create_and_store(
        meta=meta_a,
        repository=repository_a,
    )
    p_a.set_field("bar", 111)

    # B has same (model_id, name) but different uid and a different value for 'bar'.
    meta_b = ParameterMetadata(uid="b1", name="w", ptype=ParameterType.DENSE, model_id="m1")
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
    assert session_b._parameter_repository.storage_manager.has_field(  # noqa: SLF001
        meta_b.uid, "foo", table=TABLE_PARAMETERS
    )
    fields_b = session_b._get_view().list_fields_by_uid()[meta_b.uid]  # noqa: SLF001
    assert "foo" in fields_b

    WiringConfiguration.wire(container_a)
    session_a.merge(session_b, verify=True)

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
        session.patch_meta(updates={"p_invalid": "not_a_dict"}, force=False)


def test_erase_models_neither_args_nor_flag_raises(temp_dir: Path) -> None:
    """erase_models with no args and erase_all=False should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="No model_ids provided"):
        session.erase_models()


def test_erase_models_both_args_and_flag_raises(temp_dir: Path) -> None:
    """erase_models with both args and erase_all=True should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="Cannot specify both"):
        session.erase_models("model1", erase_all=True)


def test_erase_results_neither_args_nor_flag_raises(temp_dir: Path) -> None:
    """erase_results with no args and erase_all=False should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="No fields_to_erase provided"):
        session.erase_results()


def test_erase_results_both_args_and_flag_raises(temp_dir: Path) -> None:
    """erase_results with both args and erase_all=True should raise."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(ValueError, match="Cannot specify both"):
        session.erase_results("field1", erase_all=True)


def test_patch_meta_unknown_uid_raises(temp_dir: Path) -> None:
    """patch_meta with unknown UID should raise SessionError."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(SessionError, match="Unknown parameter UIDs"):
        session.patch_meta(updates={"nonexistent_uid": {"key": "value"}}, force=False)


def test_ingest_fields_unknown_uid_raises(temp_dir: Path) -> None:
    """ingest_fields with unknown UID should raise SessionError."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)
    session = Session(container=create_main_container(cfg))

    with pytest.raises(SessionError, match="Unknown parameter UIDs"):
        session.ingest_fields({"nonexistent_uid": {"field": 1}}, force=False)


def test_session_get_results_returns_structured_export(temp_dir: Path) -> None:
    """Test that get_results returns StructuredExportResult with scalars and aggregates."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    repository = container.nn.parameter_repository()

    meta = ParameterMetadata(uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1")
    p = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    p.set_field("scalar_field", 42)

    session = Session(container=container)
    result = session.get_results("scalar_field", export_format="dict")

    # Should have scalars dict and empty aggregates list
    assert hasattr(result, "scalars")
    assert hasattr(result, "aggregates")
    assert result.scalars[meta.uid]["fields"]["scalar_field"] == 42
    assert result.aggregates == []


def test_session_erase_results_removes_scalar_field(temp_dir: Path) -> None:
    """Test that erase_results removes scalar fields correctly."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    storage = container.storage.storage_manager()
    repository = container.nn.parameter_repository()

    # Create a parameter with weights
    meta = ParameterMetadata(uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1")
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

    # Compute the metric
    session.compute("metric")
    assert storage.has_field(meta.uid, "metric", table=TABLE_PARAMETERS)

    # Erase 'metric'
    session.erase_results("metric")

    assert not storage.has_field(meta.uid, "metric", table=TABLE_PARAMETERS)


def test_session_list_aggregates_returns_empty_initially(temp_dir: Path) -> None:
    """Test that list_aggregates returns empty list when no aggregates exist."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))
    aggregates = session.list_aggregates()
    assert aggregates == []


def test_session_ingest_aggregates_creates_aggregates(temp_dir: Path) -> None:
    """Test that ingest_aggregates creates aggregates correctly."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    # Ingest an aggregate
    session.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "context_params": ("layer.weight",),
            "value": np.eye(3),
        }
    ])

    aggregates = session.list_aggregates()
    assert len(aggregates) == 1
    assert aggregates[0]["field_name"] == "l_overlap"
    assert aggregates[0]["context_models"] == ("m1", "m2")
    assert aggregates[0]["context_params"] == ("layer.weight",)


def test_session_ingest_aggregates_conflict_raises(temp_dir: Path) -> None:
    """Test that ingest_aggregates raises on conflict when force=False."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    # Ingest initial aggregate
    session.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "value": np.eye(3),
        }
    ])

    # Try to ingest same aggregate again
    with pytest.raises(SessionError, match="conflicts"):
        session.ingest_aggregates([
            {
                "field_name": "l_overlap",
                "context_models": ("m1", "m2"),
                "value": np.eye(4),
            }
        ], force=False)


def test_session_ingest_aggregates_force_overwrites(temp_dir: Path) -> None:
    """Test that ingest_aggregates with force=True overwrites existing."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    # Ingest initial aggregate
    session.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "value": np.eye(3),
        }
    ])

    # Overwrite with force=True
    new_value = np.ones((4, 4))
    session.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "value": new_value,
        }
    ], force=True)

    aggregates = session.list_aggregates(verbose=True)
    assert len(aggregates) == 1
    assert np.array_equal(aggregates[0]["value"], new_value)


def test_session_erase_results_removes_aggregates(temp_dir: Path) -> None:
    """Test that erase_results removes aggregate entries by field_name."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    registry = container.compute_singleton.kernel_registry()

    # Register a dummy kernel that "produces" l_overlap so it can be erased
    registry.register_kernel(
        name="l_overlap",
        require_fields=(),
        produce_fields=("l_overlap",),
        implementation=lambda: None,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config={},
    )

    session = Session(container=container)

    # Ingest aggregates
    session.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "value": np.eye(3),
        },
        {
            "field_name": "other_field",
            "context_models": ("m1", "m3"),
            "value": np.ones((2, 2)),
        },
    ])

    assert len(session.list_aggregates()) == 2

    # Erase l_overlap
    session.erase_results("l_overlap")

    # Only other_field should remain
    aggregates = session.list_aggregates()
    assert len(aggregates) == 1
    assert aggregates[0]["field_name"] == "other_field"


def test_session_erase_models_removes_aggregates(temp_dir: Path) -> None:
    """Test that erase_models removes aggregates with model in context_models."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    container = create_main_container(cfg)
    repository = container.nn.parameter_repository()

    # Create parameters for m1 and m2
    meta_m1 = ParameterMetadata(uid="p1", name="w", ptype=ParameterType.DENSE, model_id="m1")
    ParameterDataProxy.create_and_store(meta=meta_m1, repository=repository)

    meta_m2 = ParameterMetadata(uid="p2", name="w", ptype=ParameterType.DENSE, model_id="m2")
    ParameterDataProxy.create_and_store(meta=meta_m2, repository=repository)

    session = Session(container=container)

    # Ingest aggregates involving m1
    session.ingest_aggregates([
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
    ])

    assert len(session.list_aggregates()) == 2

    # Erase model m1
    session.erase_models("m1")

    # Only r_overlap (m2, m3) should remain - l_overlap (m1, m2) should be gone
    aggregates = session.list_aggregates()
    assert len(aggregates) == 1
    assert aggregates[0]["field_name"] == "r_overlap"


def test_session_merge_copies_aggregates(temp_dir: Path) -> None:
    """Test that merge copies aggregates from source session."""
    cfg_a = temp_dir / "a.ini"
    cfg_b = temp_dir / "b.ini"
    _write_ram_config(cfg_a)
    _write_ram_config(cfg_b)

    from diffract.containers import WiringConfiguration

    container_a = create_main_container(cfg_a)
    container_b = create_main_container(cfg_b)

    # Session B has aggregates
    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)
    session_b.ingest_aggregates([
        {
            "field_name": "l_overlap",
            "context_models": ("m1", "m2"),
            "value": np.eye(3),
        }
    ])
    assert len(session_b.list_aggregates()) == 1

    # Check that aggregate_repository has the aggregate before wiring change
    agg_repo_b = session_b._aggregate_repository
    assert len(agg_repo_b) == 1

    # Session A is empty
    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)
    assert len(session_a.list_aggregates()) == 0

    # Check session_b's aggregate_repository after wiring change
    # It should still have the aggregate since it's the same object
    assert len(session_b._aggregate_repository) == 1
    assert session_b._aggregate_repository is agg_repo_b

    # Merge B into A
    session_a.merge(session_b, verify=False)

    # A should now have the aggregate
    aggregates = session_a.list_aggregates()
    assert len(aggregates) == 1
    assert aggregates[0]["field_name"] == "l_overlap"


def test_session_list_aggregates_filters_by_field_name(temp_dir: Path) -> None:
    """Test that list_aggregates can filter by field_name."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    session.ingest_aggregates([
        {"field_name": "l_overlap", "context_models": ("m1", "m2"), "value": 1},
        {"field_name": "r_overlap", "context_models": ("m1", "m2"), "value": 2},
    ])

    filtered = session.list_aggregates(field_names=["l_overlap"])
    assert len(filtered) == 1
    assert filtered[0]["field_name"] == "l_overlap"


def test_session_list_aggregates_filters_by_model_ids(temp_dir: Path) -> None:
    """Test that list_aggregates can filter by model_ids."""
    cfg = temp_dir / "cfg.ini"
    _write_ram_config(cfg)

    session = Session(container=create_main_container(cfg))

    session.ingest_aggregates([
        {"field_name": "overlap", "context_models": ("m1", "m2"), "value": 1},
        {"field_name": "overlap", "context_models": ("m3", "m4"), "value": 2},
    ])

    filtered = session.list_aggregates(model_ids=["m1"])
    assert len(filtered) == 1
    assert "m1" in filtered[0]["context_models"]
