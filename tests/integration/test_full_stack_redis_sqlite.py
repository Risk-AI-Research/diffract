"""Full-stack integration tests: Redis cache + SQLite storage."""

from __future__ import annotations

import os

import numpy as np
import pytest

from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = [pytest.mark.integration, pytest.mark.network]

REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))


def test_session_add_model_and_compute_with_redis_sqlite(
    session_with_redis_sqlite,
) -> None:
    """Add model, compute and read results with Redis cache + SQLite storage."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_sqlite

    @session.compute.kernel(require_fields=("weights",), produce_fields=("w_sum",))
    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    weights = np.arange(6, dtype=np.float32).reshape(2, 3)

    with session:
        session.models.add({"layer.0.weight": torch.from_numpy(weights)}, model_id="m1")
        session.compute.apply("w_sum")
        result = session.results.export_metrics("w_sum", export_format="dict")

    assert len(result) == 1
    ((_, entry),) = result.items()
    assert entry["fields"]["w_sum"] == float(np.sum(weights))


def test_cache_hit_miss_behavior_redis_sqlite(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Test cache hit/miss behavior with Redis cache and SQLite storage."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    uid = "test_obj"
    field = "test_field"
    value = {"data": [1, 2, 3]}

    # First access: cache miss, should load from storage
    storage.set_field(uid, field, value)
    assert not cache.has_field(uid, field)

    # Simulate cache miss by loading from storage
    loaded = storage.get_field(uid, field)
    assert loaded == value

    # Cache the value
    cache.set_field(uid, field, loaded)

    # Second access: cache hit
    assert cache.has_field(uid, field)
    cached = cache.get_field(uid, field)
    assert cached == value

    # Verify storage still has it
    assert storage.has_field(uid, field)


def test_concurrent_reads_with_redis_sqlite(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Test concurrent reads with Redis cache and SQLite storage."""
    from .helpers import run_concurrent_operations

    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Populate data
    rng = np.random.default_rng(0)
    num_objects = 100
    for i in range(num_objects):
        value = {"index": i, "data": rng.standard_normal(10).astype(np.float32)}
        storage.set_field(f"obj_{i}", "field", value)

    def read_operation(thread_idx: int) -> dict:
        obj_idx = thread_idx % num_objects
        uid = f"obj_{obj_idx}"

        # Try cache first
        if cache.has_field(uid, "field"):
            return cache.get_field(uid, "field")

        # Cache miss, load from storage
        value = storage.get_field(uid, "field")
        cache.set_field(uid, "field", value)
        return value

    results, exceptions = run_concurrent_operations(
        read_operation, num_threads=10, operations_per_thread=50
    )

    assert not exceptions, f"Exceptions occurred: {exceptions}"
    assert len(results) == 500  # 10 threads * 50 operations


def test_batch_operations_redis_sqlite(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Test batch operations with Redis cache and SQLite storage."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Batch write operations
    with storage:
        for i in range(50):
            storage.set_field(f"obj_{i}", "field", {"value": i})

    # Verify all written
    for i in range(50):
        assert storage.has_field(f"obj_{i}", "field")
        value = storage.get_field(f"obj_{i}", "field")
        assert value["value"] == i

    # Batch cache operations
    for i in range(50):
        if storage.has_field(f"obj_{i}", "field"):
            value = storage.get_field(f"obj_{i}", "field")
            cache.set_field(f"obj_{i}", "field", value)

    # Verify cache
    for i in range(50):
        assert cache.has_field(f"obj_{i}", "field")


def test_session_merge_with_redis_sqlite(
    temp_dir, redis_cache_manager, sqlite_storage_manager
) -> None:
    """Test session merge with Redis cache and SQLite storage."""
    from diffract.containers import WiringConfiguration, create_main_container
    from diffract.session import Session

    # Create two sessions with same storage/cache
    config_path_a = temp_dir / "config_a.ini"
    config_path_b = temp_dir / "config_b.ini"

    for idx, config_path in enumerate([config_path_a, config_path_b]):
        config_content = f"""
[storage]
backend = sqlite

[storage.sqlite]
path = {temp_dir / f"merge_storage_{idx}.db"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / f"merge_metadata_{idx}.db"}

[cache]
backend = redis

[cache.redis]
host = localhost
port = 6379
db = {REDIS_DB}
max_memory_mb = 128
key_prefix = diffract:test:merge:

[compute.executor]
max_workers = 2

[nn.extractor]
skip_not_implemented_types = true
"""
        config_path.write_text(config_content.strip() + "\n")

    try:
        container_a = create_main_container(config_path_a)
        WiringConfiguration.wire(container_a)
        session_a = Session(container=container_a)

        container_b = create_main_container(config_path_b)
        WiringConfiguration.wire(container_b)
        session_b = Session(container=container_b)

        # Add data to session_b
        container_b.storage.storage_manager()
        container_b.cache.cache_manager()

        meta = ParameterMetadata(
            uid="merge_test",
            name="test.weight",
            ptype=ParameterType.DENSE,
            model_id="m_merge",
        )
        weights = np.ones((5, 5), dtype=np.float32)
        repo_b = container_b.nn.parameter_repository()
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repo_b)
        proxy.set_field("weights", weights)
        proxy.set_field("test_field", 42)

        # Merge
        WiringConfiguration.wire(container_a)
        session_a.utils.merge_other_session(session_b, verify=True)

        # Verify merge
        result = session_a.results.export_metrics("test_field", export_format="dict")
        assert meta.uid in result
        assert result[meta.uid]["fields"]["test_field"] == 42

    except Exception as e:
        if "Redis" in str(e) or "redis" in str(e).lower():
            pytest.skip(f"Redis not available: {e}")
        raise


def test_large_arrays_sqlite_blobs(redis_cache_manager, sqlite_storage_manager) -> None:
    """Test large arrays through SQLite blob storage with Redis cache."""
    from .helpers import create_large_array

    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Create large array (5MB)
    large_array = create_large_array(5, dtype=np.float32)
    uid = "large_obj"
    field = "large_array"

    # Store in SQLite (should use blob storage if threshold is low)
    storage.set_field(uid, field, large_array)

    # Verify storage
    assert storage.has_field(uid, field)
    stored = storage.get_field(uid, field)
    assert stored.shape == large_array.shape
    np.testing.assert_allclose(stored, large_array)

    # Cache it
    cache.set_field(uid, field, large_array)

    # Verify cache
    assert cache.has_field(uid, field)
    cached = cache.get_field(uid, field)
    assert cached.shape == large_array.shape
    np.testing.assert_allclose(cached, large_array)
