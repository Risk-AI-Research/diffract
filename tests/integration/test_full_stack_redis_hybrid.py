"""Full-stack integration tests: Redis cache + Hybrid storage (SQLite + HDF5)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.network]

REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))


def test_hybrid_routing_with_redis_cache(
    redis_cache_manager, hybrid_storage_manager
) -> None:
    """Test hybrid routing (small data → SQLite, large → HDF5) with Redis cache."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    # Small data should go to SQLite
    small_value = {"metadata": "test", "value": 42}
    storage.set_field("small_obj", "meta", small_value)
    assert storage.has_field("small_obj", "meta")
    stored_small = storage.get_field("small_obj", "meta")
    assert stored_small == small_value

    # Cache it
    cache.set_field("small_obj", "meta", small_value)
    assert cache.has_field("small_obj", "meta")

    # Large array should go to HDF5 (threshold is 1MB in fixture)
    from .helpers import create_large_array

    large_array = create_large_array(2, dtype=np.float32)  # 2MB array
    storage.set_field("large_obj", "weights", large_array)
    assert storage.has_field("large_obj", "weights")
    stored_large = storage.get_field("large_obj", "weights")
    assert stored_large.shape == large_array.shape
    np.testing.assert_allclose(stored_large, large_array)

    # Cache large array
    cache.set_field("large_obj", "weights", large_array)
    assert cache.has_field("large_obj", "weights")
    cached_large = cache.get_field("large_obj", "weights")
    assert cached_large.shape == large_array.shape
    np.testing.assert_allclose(cached_large, large_array)


def test_hybrid_cross_backend_operations(
    redis_cache_manager, hybrid_storage_manager
) -> None:
    """Test operations that span both SQLite and HDF5 backends."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    # Create object with both small and large fields
    uid = "mixed_obj"
    small_field = "metadata"
    large_field = "weights"

    small_data = {"model": "test", "layer": 0}
    from .helpers import create_large_array

    large_data = create_large_array(2, dtype=np.float32)

    # Store both
    storage.set_field(uid, small_field, small_data)
    storage.set_field(uid, large_field, large_data)

    # Verify both exist
    assert storage.has_field(uid, small_field)
    assert storage.has_field(uid, large_field)

    # List all fields for this object
    fields = storage.list_fields(uid)
    assert small_field in fields
    assert large_field in fields

    # Cache both
    cache.set_field(uid, small_field, small_data)
    cache.set_field(uid, large_field, large_data)

    # Verify cache
    assert cache.has_field(uid, small_field)
    assert cache.has_field(uid, large_field)


def test_hybrid_batch_context_consistency(
    redis_cache_manager, hybrid_storage_manager
) -> None:
    """Test batch context consistency across both backends."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    # Batch operations
    with storage:
        for i in range(20):
            small_data = {"index": i}
            storage.set_field(f"obj_{i}", "meta", small_data)

            if i % 2 == 0:  # Every other object gets large array
                from .helpers import create_large_array

                large_array = create_large_array(1, dtype=np.float32)
                storage.set_field(f"obj_{i}", "array", large_array)

    # Verify all written
    for i in range(20):
        assert storage.has_field(f"obj_{i}", "meta")
        if i % 2 == 0:
            assert storage.has_field(f"obj_{i}", "array")

    # Batch cache operations
    for i in range(20):
        if storage.has_field(f"obj_{i}", "meta"):
            value = storage.get_field(f"obj_{i}", "meta")
            cache.set_field(f"obj_{i}", "meta", value)

        if storage.has_field(f"obj_{i}", "array"):
            value = storage.get_field(f"obj_{i}", "array")
            cache.set_field(f"obj_{i}", "array", value)


def test_hybrid_large_and_small_data_mix(
    redis_cache_manager, hybrid_storage_manager
) -> None:
    """Test mixed data: metadata in SQLite, large arrays in HDF5."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    from .helpers import create_large_array

    # Create multiple objects with mixed data
    for i in range(10):
        uid = f"mixed_{i}"
        meta = {"id": i, "name": f"layer_{i}"}
        weights = create_large_array(1, dtype=np.float32)

        storage.set_field(uid, "metadata", meta)
        storage.set_field(uid, "weights", weights)

        # Cache both
        cache.set_field(uid, "metadata", meta)
        cache.set_field(uid, "weights", weights)

    # Verify all
    for i in range(10):
        uid = f"mixed_{i}"
        assert storage.has_field(uid, "metadata")
        assert storage.has_field(uid, "weights")
        assert cache.has_field(uid, "metadata")
        assert cache.has_field(uid, "weights")

        # Verify data integrity
        meta = storage.get_field(uid, "metadata")
        assert meta["id"] == i

        weights = storage.get_field(uid, "weights")
        assert isinstance(weights, np.ndarray)


def test_hybrid_erase_operations(redis_cache_manager, hybrid_storage_manager) -> None:
    """Test erase operations that remove data from both backends."""
    cache = redis_cache_manager
    storage = hybrid_storage_manager

    from .helpers import create_large_array

    uid = "erase_test"
    small_field = "meta"
    large_field = "array"

    # Create data
    storage.set_field(uid, small_field, {"test": True})
    storage.set_field(uid, large_field, create_large_array(1, dtype=np.float32))
    cache.set_field(uid, small_field, {"test": True})
    cache.set_field(uid, large_field, create_large_array(1, dtype=np.float32))

    # Erase field
    storage.erase_field(uid, small_field)
    assert not storage.has_field(uid, small_field)
    cache.erase_field(uid, small_field)
    assert not cache.has_field(uid, small_field)

    # Erase other field
    storage.erase_field(uid, large_field)
    assert not storage.has_field(uid, large_field)
    cache.erase_field(uid, large_field)
    assert not cache.has_field(uid, large_field)

    # Erase entire object
    storage.set_field(uid, small_field, {"test": True})
    storage.set_field(uid, large_field, create_large_array(1, dtype=np.float32))
    storage.erase_obj(uid)
    assert not storage.has_field(uid, small_field)
    assert not storage.has_field(uid, large_field)


def test_session_with_redis_hybrid(session_with_redis_hybrid) -> None:
    """Test full Session workflow with Redis cache and Hybrid storage."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_hybrid

    @session.compute.kernel(require_fields=("weights",), produce_fields=("w_sum",))
    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((100, 100)).astype(np.float32)

    with session:
        session.models.add({"test.weight": torch.from_numpy(weights)}, model_id="m1")
        session.compute.apply("w_sum")
        result = session.results.export_metrics("w_sum", export_format="dict")

    assert len(result) == 1
    ((_, entry),) = result.items()
    assert entry["fields"]["w_sum"] == float(np.sum(weights))
