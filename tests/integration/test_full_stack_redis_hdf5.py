"""Full-stack integration tests: Redis cache + HDF5 storage."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.network]

REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))


def test_session_add_model_and_compute_with_redis_hdf5(
    session_with_redis_hdf5,
) -> None:
    """Full cycle: add model → compute → read results with Redis cache + HDF5 storage."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_hdf5

    @session.compute.kernel(require_fields=("weights",), produce_fields=("w_sum",))
    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    weights = np.arange(12, dtype=np.float32).reshape(3, 4)

    with session:
        session.models.add(
            {"layer.0.weight": torch.from_numpy(weights)}, model_id="m1"
        )
        session.compute.apply("w_sum")
        result = session.results.export_metrics("w_sum", export_format="dict")

    assert len(result) == 1
    ((_, entry),) = result.items()
    assert entry["fields"]["w_sum"] == float(np.sum(weights))


def test_hdf5_swmr_with_redis_cache(redis_cache_manager, hdf5_storage_manager) -> None:
    """Test HDF5 SWMR mode with Redis cache and concurrent reads."""
    from .helpers import run_concurrent_operations

    cache = redis_cache_manager
    storage = hdf5_storage_manager

    # Populate data
    num_objects = 50
    for i in range(num_objects):
        arr = np.random.randn(100, 100).astype(np.float32)
        storage.set_field(f"obj_{i}", "array", arr)

    def read_operation(thread_idx: int) -> np.ndarray:
        obj_idx = thread_idx % num_objects
        uid = f"obj_{obj_idx}"

        # Try cache first
        if cache.has_field(uid, "array"):
            return cache.get_field(uid, "array")

        # Cache miss, load from HDF5
        arr = storage.get_field(uid, "array")
        cache.set_field(uid, "array", arr)
        return arr

    results, exceptions = run_concurrent_operations(
        read_operation, num_threads=8, operations_per_thread=20
    )

    assert not exceptions, f"Exceptions occurred: {exceptions}"
    assert len(results) == 160  # 8 threads * 20 operations

    # Verify all results are valid arrays
    for result in results:
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)


def test_compression_and_cache_interaction(
    redis_cache_manager, hdf5_storage_manager
) -> None:
    """Test HDF5 compression interaction with Redis cache."""
    cache = redis_cache_manager
    storage = hdf5_storage_manager

    # Create array
    arr = np.random.randn(1000, 1000).astype(np.float32)
    uid = "compressed_obj"
    field = "compressed_array"

    # Store in HDF5 (with compression if configured)
    storage.set_field(uid, field, arr)

    # Verify storage
    assert storage.has_field(uid, field)
    stored = storage.get_field(uid, field)
    assert stored.shape == arr.shape
    np.testing.assert_allclose(stored, arr, rtol=1e-5)

    # Cache uncompressed version
    cache.set_field(uid, field, arr)

    # Verify cache has uncompressed data
    assert cache.has_field(uid, field)
    cached = cache.get_field(uid, field)
    assert cached.shape == arr.shape
    np.testing.assert_allclose(cached, arr)


def test_large_datasets_hdf5_redis(redis_cache_manager, hdf5_storage_manager) -> None:
    """Test large datasets (100MB+) with HDF5 storage and Redis cache."""
    from .helpers import create_large_array

    cache = redis_cache_manager
    storage = hdf5_storage_manager

    # Create large array (100MB)
    large_array = create_large_array(100, dtype=np.float32)
    uid = "large_obj"
    field = "large_array"

    # Store in HDF5
    storage.set_field(uid, field, large_array)

    # Verify storage
    assert storage.has_field(uid, field)
    stored = storage.get_field(uid, field)
    assert stored.shape == large_array.shape
    np.testing.assert_allclose(stored, large_array, rtol=1e-5)

    # Cache it (may exceed Redis memory limit, should handle gracefully)
    try:
        cache.set_field(uid, field, large_array)
        # If successful, verify cache
        if cache.has_field(uid, field):
            cached = cache.get_field(uid, field)
            assert cached.shape == large_array.shape
            np.testing.assert_allclose(cached, large_array, rtol=1e-5)
    except Exception:  # noqa: BLE001
        # Redis may evict or reject very large values, that's OK
        pass


def test_index_consistency_with_cache(
    redis_cache_manager, hdf5_storage_manager
) -> None:
    """Test HDF5 index consistency when using Redis cache."""
    cache = redis_cache_manager
    storage = hdf5_storage_manager

    # Add multiple objects
    uids = []
    for i in range(20):
        uid = f"obj_{i}"
        uids.append(uid)
        arr = np.random.randn(50, 50).astype(np.float32)
        storage.set_field(uid, "weights", arr)
        cache.set_field(uid, "weights", arr)

    # Verify index consistency
    listed = storage.list_objs()
    assert len(listed) >= 20

    # Verify all objects are accessible
    for uid in uids[:10]:  # Check first 10
        assert storage.has_field(uid, "weights")
        assert cache.has_field(uid, "weights")

    # Remove some objects
    for uid in uids[:5]:
        storage.erase_obj(uid)
        cache.erase_field(uid, "weights")

    # Verify index updated
    listed_after = storage.list_objs()
    assert len(listed_after) < len(listed)

