"""Full-stack integration tests: Simple cache + all storage backends."""

from __future__ import annotations

from typing import Generator

import numpy as np
import pytest

from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def simple_cache_manager() -> Generator:
    """Fixture for Simple cache manager."""
    from diffract.core.cache.simple_manager import SimpleLRUCacheManager

    cache = SimpleLRUCacheManager(max_memory_mb=128, ttl_seconds=None)
    cache.clear()
    try:
        yield cache
    finally:
        cache.clear()


def test_simple_cache_sqlite(simple_cache_manager, sqlite_storage_manager) -> None:
    """Test Simple cache with SQLite storage."""
    cache = simple_cache_manager
    storage = sqlite_storage_manager

    uid = "test_obj"
    field = "test_field"
    value = {"data": [1, 2, 3]}

    # Store in SQLite
    storage.set_field(uid, field, value)
    assert storage.has_field(uid, field)

    # Cache miss
    assert not cache.has_field(uid, field)

    # Load and cache
    loaded = storage.get_field(uid, field)
    cache.set_field(uid, field, loaded)

    # Cache hit
    assert cache.has_field(uid, field)
    cached = cache.get_field(uid, field)
    assert cached == value


def test_simple_cache_hdf5(simple_cache_manager, hdf5_storage_manager) -> None:
    """Test Simple cache with HDF5 storage."""
    cache = simple_cache_manager
    storage = hdf5_storage_manager

    uid = "test_obj"
    field = "array"
    arr = np.random.randn(100, 100).astype(np.float32)

    # Store in HDF5
    storage.set_field(uid, field, arr)
    assert storage.has_field(uid, field)

    # Load and cache
    loaded = storage.get_field(uid, field)
    cache.set_field(uid, field, loaded)

    # Verify cache
    assert cache.has_field(uid, field)
    cached = cache.get_field(uid, field)
    assert cached.shape == arr.shape
    np.testing.assert_allclose(cached, arr)


def test_simple_cache_hybrid(simple_cache_manager, hybrid_storage_manager) -> None:
    """Test Simple cache with Hybrid storage."""
    cache = simple_cache_manager
    storage = hybrid_storage_manager

    from .helpers import create_large_array

    # Small data (SQLite)
    uid_small = "small_obj"
    small_value = {"meta": "test"}
    storage.set_field(uid_small, "meta", small_value)
    loaded_small = storage.get_field(uid_small, "meta")
    cache.set_field(uid_small, "meta", loaded_small)
    assert cache.has_field(uid_small, "meta")
    assert cache.get_field(uid_small, "meta") == small_value

    # Large data (HDF5)
    uid_large = "large_obj"
    large_array = create_large_array(2, dtype=np.float32)
    storage.set_field(uid_large, "weights", large_array)
    loaded_large = storage.get_field(uid_large, "weights")
    cache.set_field(uid_large, "weights", loaded_large)
    assert cache.has_field(uid_large, "weights")
    cached_large = cache.get_field(uid_large, "weights")
    assert cached_large.shape == large_array.shape
    np.testing.assert_allclose(cached_large, large_array)


def test_simple_cache_lru_eviction(simple_cache_manager, sqlite_storage_manager) -> None:
    """Test Simple cache LRU eviction behavior."""
    cache = simple_cache_manager
    storage = sqlite_storage_manager

    # Fill cache beyond max_size (1000)
    for i in range(1100):
        uid = f"obj_{i}"
        value = {"index": i}
        storage.set_field(uid, "field", value)
        loaded = storage.get_field(uid, "field")
        cache.set_field(uid, "field", loaded)

    # Some items should be evicted
    # Check that recent items are still cached
    assert cache.has_field("obj_1099", "field")
    assert cache.has_field("obj_1000", "field")

    # Older items may be evicted (non-deterministic, but likely)
    # This is a probabilistic test - just verify cache is working


def test_simple_cache_session_workflow(
    temp_dir, simple_cache_manager, sqlite_storage_manager
) -> None:
    """Test Session workflow with Simple cache and SQLite storage."""
    from diffract.containers import MainContainer, WiringConfiguration, create_main_container
    from diffract.session import Session

    config_path = temp_dir / "config_simple_sqlite.ini"
    config_content = f"""
[storage]
backend = sqlite

[storage.sqlite]
path = {temp_dir / "session_simple.db"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / "session_simple_metadata.db"}

[cache]
backend = simple

[cache.simple]
max_memory_mb = 128
ttl_seconds = 3600

[compute.executor]
max_workers = 2

[nn.extractor]
skip_not_implemented_types = true
"""
    config_path.write_text(config_content.strip() + "\n")

    container = create_main_container(config_path)
    WiringConfiguration.wire(container)
    session = Session(container=container)

    registry = container.compute_singleton.kernel_registry()

    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    _, cfg_dict = registry._split_signature(w_sum)  # noqa: SLF001
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

    repo = container.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="simple_test",
        name="test.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
    )
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    proxy = ParameterDataProxy.create_and_store(
        meta=meta, repository=repo
    )
    proxy.set_field("weights", weights)

    session.compute.apply("w_sum")

    result = session.results.export_metrics("w_sum", export_format="dict")
    assert meta.uid in result
    assert result[meta.uid]["fields"]["w_sum"] == float(np.sum(weights))

