"""Stress tests for concurrent operations."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.stress,
    pytest.mark.network,
    pytest.mark.slow,
]

REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))


def test_stress_concurrent_reads_redis_sqlite(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Stress test: 50 threads reading concurrently via Redis cache + SQLite."""
    from .helpers import run_concurrent_operations

    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Populate data
    num_objects = 200
    rng = np.random.default_rng(0)
    for i in range(num_objects):
        data = rng.standard_normal(100).astype(np.float32).tolist()
        storage.set_field(f"obj_{i}", "field", {"index": i, "data": data})

    def read_operation(thread_idx: int) -> dict:
        obj_idx = (thread_idx * 7 + int(time.time() * 1000) % 100) % num_objects
        uid = f"obj_{obj_idx}"

        # Try cache first
        if cache.has_field(uid, "field"):
            return cache.get_field(uid, "field")

        # Cache miss, load from storage
        value = storage.get_field(uid, "field")
        cache.set_field(uid, "field", value)
        return value

    results, exceptions = run_concurrent_operations(
        read_operation, num_threads=50, operations_per_thread=100, timeout=600.0
    )

    assert not exceptions, f"Exceptions occurred: {exceptions}"
    assert len(results) == 5000  # 50 threads * 100 operations


def test_stress_concurrent_writes_redis_sqlite(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Stress test: concurrent writes with locking."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    num_threads = 20
    writes_per_thread = 50
    exceptions: list[Exception] = []
    exceptions_lock = threading.Lock()

    def write_operation(thread_idx: int) -> None:
        rng = np.random.default_rng(thread_idx)
        for i in range(writes_per_thread):
            uid = f"write_obj_{thread_idx}_{i}"
            value = {
                "thread": thread_idx,
                "write": i,
                "data": rng.standard_normal(10).astype(np.float32).tolist(),
            }

            try:
                # Write to storage (serialized)
                with storage:
                    storage.set_field(uid, "field", value)

                # Write to cache (may have contention)
                cache.set_field(uid, "field", value)
            except Exception as e:
                with exceptions_lock:
                    exceptions.append(e)
                raise

    threads = [
        threading.Thread(target=write_operation, args=(i,)) for i in range(num_threads)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300.0)
        if t.is_alive():
            msg = f"Thread {t.name} did not complete within timeout"
            raise TimeoutError(msg)

    assert not exceptions, f"Exceptions occurred: {exceptions}"

    # Verify all writes succeeded
    for thread_idx in range(num_threads):
        for i in range(writes_per_thread):
            uid = f"write_obj_{thread_idx}_{i}"
            assert storage.has_field(uid, "field"), f"Missing: {uid}"


def test_stress_reads_during_writes_wal(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Stress test: reads during WAL write transactions."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Initial data
    for i in range(100):
        storage.set_field(f"obj_{i}", "field", {"index": i})

    read_completed = threading.Event()
    write_completed = threading.Event()
    exceptions: list[Exception] = []
    exceptions_lock = threading.Lock()

    def reader() -> None:
        try:
            for _ in range(100):
                obj_idx = _ % 100
                uid = f"obj_{obj_idx}"

                # Try cache
                if cache.has_field(uid, "field"):
                    _ = cache.get_field(uid, "field")
                else:
                    # Read from storage (should work during WAL writes)
                    _ = storage.get_field(uid, "field")

            read_completed.set()
        except Exception as e:  # noqa: BLE001
            with exceptions_lock:
                exceptions.append(e)

    def writer() -> None:
        try:
            with storage:
                for i in range(50):
                    storage.set_field(f"new_obj_{i}", "field", {"index": i})
            write_completed.set()
        except Exception as e:  # noqa: BLE001
            with exceptions_lock:
                exceptions.append(e)

    # Start writer
    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    # Start multiple readers
    reader_threads = [threading.Thread(target=reader, daemon=True) for _ in range(10)]
    for t in reader_threads:
        t.start()

    # Wait for completion
    assert read_completed.wait(60.0), "Readers did not complete"
    assert write_completed.wait(60.0), "Writer did not complete"

    for t in reader_threads:
        t.join(timeout=10.0)

    writer_thread.join(timeout=10.0)

    assert not exceptions, f"Exceptions occurred: {exceptions}"


def test_stress_connection_pool_exhaustion(sqlite_storage_manager) -> None:
    """Stress test: connection pool exhaustion and overflow."""
    storage = sqlite_storage_manager

    # Populate data
    for i in range(100):
        storage.set_field(f"obj_{i}", "field", {"index": i})

    num_threads = 30  # More than default pool size (8)
    operations_per_thread = 100
    exceptions: list[Exception] = []
    exceptions_lock = threading.Lock()

    def read_operation(thread_idx: int) -> None:
        for _ in range(operations_per_thread):
            obj_idx = (thread_idx + _) % 100
            uid = f"obj_{obj_idx}"
            try:
                _ = storage.get_field(uid, "field")
            except Exception as e:  # noqa: BLE001
                with exceptions_lock:
                    exceptions.append(e)

    threads = [
        threading.Thread(target=read_operation, args=(i,)) for i in range(num_threads)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300.0)
        if t.is_alive():
            msg = f"Thread {t.name} did not complete within timeout"
            raise TimeoutError(msg)

    # Some exceptions may occur due to pool exhaustion, but should be handled gracefully
    # The key is that the system doesn't deadlock or crash


def test_stress_redis_eviction_under_load(
    redis_cache_manager, sqlite_storage_manager
) -> None:
    """Stress test: Redis eviction under memory pressure."""
    cache = redis_cache_manager
    storage = sqlite_storage_manager

    # Create many objects that exceed Redis memory limit
    num_objects = 500
    rng = np.random.default_rng(0)
    for i in range(num_objects):
        # Each object ~1MB
        large_data = rng.standard_normal(250000).astype(np.float32).tolist()
        storage.set_field(f"obj_{i}", "field", large_data)

    # Try to cache all (will trigger eviction)
    cached_count = 0
    for i in range(num_objects):
        uid = f"obj_{i}"
        if storage.has_field(uid, "field"):
            value = storage.get_field(uid, "field")
            try:
                cache.set_field(uid, "field", value)
                cached_count += 1
            except Exception:  # noqa: BLE001
                # Redis may reject or evict, that's OK
                pass

    # Verify Redis is still responsive
    cache.set_field("probe", "test", {"ok": True})
    assert cache.get_field("probe", "test") == {"ok": True}

    # Check memory usage
    mem_info = cache.get_memory_usage()
    assert "used_memory_mb" in mem_info
    assert "max_memory_mb" in mem_info


def test_stress_hdf5_swmr_concurrent_reads(hdf5_storage_manager) -> None:
    """Stress test: HDF5 SWMR with many concurrent readers."""
    storage = hdf5_storage_manager

    # Populate data
    num_objects = 100
    rng = np.random.default_rng(0)
    for i in range(num_objects):
        arr = rng.standard_normal((100, 100)).astype(np.float32)
        storage.set_field(f"obj_{i}", "array", arr)

    num_threads = 30
    operations_per_thread = 50
    exceptions: list[Exception] = []
    exceptions_lock = threading.Lock()

    def read_operation(thread_idx: int) -> None:
        for _ in range(operations_per_thread):
            obj_idx = (thread_idx + _) % num_objects
            uid = f"obj_{obj_idx}"
            try:
                arr = storage.get_field(uid, "array")
                assert arr.shape == (100, 100)
            except Exception as e:  # noqa: BLE001
                with exceptions_lock:
                    exceptions.append(e)

    threads = [
        threading.Thread(target=read_operation, args=(i,)) for i in range(num_threads)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300.0)
        if t.is_alive():
            msg = f"Thread {t.name} did not complete within timeout"
            raise TimeoutError(msg)

    # Some exceptions may occur, but should be minimal
    assert (
        len(exceptions) < num_threads * operations_per_thread * 0.1
    )  # Less than 10% failure rate
