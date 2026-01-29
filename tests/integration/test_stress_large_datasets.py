"""Stress tests for large datasets."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.slow]


def test_stress_large_arrays_hybrid(redis_cache_manager, hybrid_storage_manager) -> None:
    """Stress test: 1GB+ arrays through Hybrid storage."""
    from .helpers import create_large_array, measure_memory_usage

    cache = redis_cache_manager
    storage = hybrid_storage_manager

    # Create multiple large arrays (total > 1GB)
    num_arrays = 5
    size_mb_per_array = 250  # 250MB each = 1.25GB total

    mem_before = measure_memory_usage()

    for i in range(num_arrays):
        uid = f"large_array_{i}"
        field = "weights"

        large_array = create_large_array(size_mb_per_array, dtype=np.float32)
        storage.set_field(uid, field, large_array)

        # Verify storage
        assert storage.has_field(uid, field)
        stored = storage.get_field(uid, field)
        assert stored.shape == large_array.shape
        np.testing.assert_allclose(stored, large_array, rtol=1e-5)

    mem_after = measure_memory_usage()

    # Memory should not have grown excessively (arrays should be on disk)
    # Allow some growth for metadata and caching
    memory_growth_mb = mem_after["rss_mb"] - mem_before["rss_mb"]
    assert memory_growth_mb < 500, f"Excessive memory growth: {memory_growth_mb}MB"


def test_stress_many_parameters_sqlite(redis_cache_manager, sqlite_storage_manager) -> None:
    """Stress test: 10,000+ parameters in SQLite."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    num_parameters = 10000

    # Create many parameters
    with storage:
        for i in range(num_parameters):
            uid = f"param_{i}"
            value = {
                "index": i,
                "name": f"layer_{i % 100}.weight",
                "data": np.random.randn(10).astype(np.float32).tolist(),
            }
            storage.set_field(uid, "metadata", value)

    # Verify all stored
    all_objs = storage.list_objs()
    assert len(all_objs) >= num_parameters

    # Verify random samples
    import random

    sample_indices = random.sample(range(num_parameters), min(100, num_parameters))
    for idx in sample_indices:
        uid = f"param_{idx}"
        assert storage.has_field(uid, "metadata")
        value = storage.get_field(uid, "metadata")
        assert value["index"] == idx

    # Cache some
    cached_count = 0
    for idx in range(0, num_parameters, 100):  # Cache every 100th
        uid = f"param_{idx}"
        if storage.has_field(uid, "metadata"):
            value = storage.get_field(uid, "metadata")
            cache.set_field(uid, "metadata", value)
            cached_count += 1

    assert cached_count > 0


def test_stress_memory_limits_redis(redis_cache_manager, sqlite_storage_manager) -> None:
    """Stress test: work at Redis memory limits."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Get memory limits
    mem_info = cache.get_memory_usage()
    max_memory_mb = mem_info.get("max_memory_mb", 128)

    # Create objects that will fill cache
    # Each object ~2MB
    num_objects = int(max_memory_mb / 2) + 10  # Slightly more than limit

    for i in range(num_objects):
        uid = f"mem_test_{i}"
        # ~2MB of data
        large_data = np.random.randn(500000).astype(np.float32).tolist()
        storage.set_field(uid, "field", large_data)

    # Try to cache all (will trigger eviction)
    cached_successfully = 0
    for i in range(num_objects):
        uid = f"mem_test_{i}"
        if storage.has_field(uid, "field"):
            value = storage.get_field(uid, "field")
            try:
                cache.set_field(uid, "field", value)
                cached_successfully += 1
            except Exception:  # noqa: BLE001
                # Redis may evict or reject
                pass

    # Verify Redis is still responsive
    cache.set_field("probe", "test", {"ok": True})
    assert cache.get_field("probe", "test") == {"ok": True}

    # Check final memory usage
    final_mem = cache.get_memory_usage()
    assert "used_memory_mb" in final_mem
    assert final_mem["used_memory_mb"] <= max_memory_mb * 1.1  # Allow 10% overhead


def test_stress_batch_size_limits(sqlite_storage_manager) -> None:
    """Stress test: automatic flush when batch size limits exceeded."""
    storage = sqlite_storage_manager

    # Create storage with small batch limit
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager

    small_batch_storage = SQLiteStorageManager(
        path=storage._path,  # noqa: SLF001
        batch_size_limit_bytes=10 * 1024 * 1024,  # 10MB limit
        batch_soft_limit_ratio=0.9,
    )
    small_batch_storage.connect()

    try:
        # Create objects that will exceed batch limit
        # Each object ~1MB
        num_objects = 15  # 15MB total > 10MB limit

        for i in range(num_objects):
            uid = f"batch_test_{i}"
            # ~1MB of data
            large_data = np.random.randn(250000).astype(np.float32).tolist()
            # Should trigger auto-flush
            small_batch_storage.set_field(uid, "field", large_data)

        # Verify all stored (auto-flush should have worked)
        for i in range(num_objects):
            uid = f"batch_test_{i}"
            assert small_batch_storage.has_field(uid, "field")

    finally:
        small_batch_storage.close()


def test_stress_blob_file_operations(sqlite_storage_manager) -> None:
    """Stress test: parallel blob file operations."""
    storage = sqlite_storage_manager

    # Configure for blob storage (low threshold)
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager

    blob_storage = SQLiteStorageManager(
        path=storage._path,  # noqa: SLF001
        array_threshold=1024 * 1024,  # 1MB threshold
        blob_write_workers=8,  # Parallel blob writes
    )
    blob_storage.connect()

    try:
        # Create many large arrays that will be stored as blobs
        num_arrays = 50
        size_mb = 5  # 5MB each

        from .helpers import create_large_array

        for i in range(num_arrays):
            uid = f"blob_test_{i}"
            large_array = create_large_array(size_mb, dtype=np.float32)
            blob_storage.set_field(uid, "array", large_array)

        # Verify all stored
        for i in range(num_arrays):
            uid = f"blob_test_{i}"
            assert blob_storage.has_field(uid, "array")
            stored = blob_storage.get_field(uid, "array")
            assert stored.shape[0] > 0

    finally:
        blob_storage.close()


def test_stress_mixed_sizes_hybrid(redis_cache_manager, hybrid_storage_manager) -> None:
    """Stress test: mixed data sizes in Hybrid storage."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    # Mix of small (SQLite) and large (HDF5) data
    num_small = 1000
    num_large = 50

    from .helpers import create_large_array

    # Small data (SQLite)
    for i in range(num_small):
        uid = f"small_{i}"
        value = {"index": i, "metadata": f"obj_{i}"}
        storage.set_field(uid, "meta", value)
        cache.set_field(uid, "meta", value)

    # Large data (HDF5)
    for i in range(num_large):
        uid = f"large_{i}"
        large_array = create_large_array(10, dtype=np.float32)  # 10MB each
        storage.set_field(uid, "weights", large_array)
        # Don't cache large arrays to avoid memory issues

    # Verify all
    all_objs = storage.list_objs()
    assert len(all_objs) >= num_small + num_large

    # Verify samples
    for i in range(0, num_small, 100):
        uid = f"small_{i}"
        assert storage.has_field(uid, "meta")
        assert cache.has_field(uid, "meta")

    for i in range(0, num_large, 10):
        uid = f"large_{i}"
        assert storage.has_field(uid, "weights")

