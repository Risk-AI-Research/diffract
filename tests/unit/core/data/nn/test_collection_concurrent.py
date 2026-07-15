"""Tests for concurrent operations in ParameterRepository."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Self

import numpy as np
import pytest

from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.parallel import ParallelContext, calibrate_thread_pool_overhead

pytestmark = pytest.mark.unit


class ThreadSafeStorage:
    """Thread-safe fake storage for concurrent tests."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._latency_s = latency_s
        self._call_count: dict[str, int] = {}
        self._call_count_lock = threading.Lock()
        self._threads_by_method: dict[str, set[int]] = {}
        self._threads_lock = threading.Lock()

    def _track_call(self, method: str) -> None:
        with self._call_count_lock:
            self._call_count[method] = self._call_count.get(method, 0) + 1
        with self._threads_lock:
            self._threads_by_method.setdefault(method, set()).add(threading.get_ident())

    def get_call_count(self, method: str) -> int:
        with self._call_count_lock:
            return self._call_count.get(method, 0)

    def reset_thread_ids(self) -> None:
        with self._threads_lock:
            self._threads_by_method.clear()

    def get_thread_ids(self, method: str) -> set[int]:
        with self._threads_lock:
            return set(self._threads_by_method.get(method, set()))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = "default"
    ) -> None:
        self._track_call("set_field")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            self._store.setdefault(field_name, {})[obj_uid] = value

    def get_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> Any:
        self._track_call("get_field")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            try:
                return self._store[field_name][obj_uid]
            except KeyError as err:
                raise KeyError(f"Field {field_name} not found for {obj_uid}") from err

    def has_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> bool:
        self._track_call("has_field")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            return obj_uid in self._store.get(field_name, {})

    def erase_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> None:
        self._track_call("erase_field")
        with self._lock:
            self._store.get(field_name, {}).pop(obj_uid, None)

    def erase_obj(self, obj_uid: str, *, table: str = "default") -> None:
        self._track_call("erase_obj")
        with self._lock:
            for field_data in self._store.values():
                field_data.pop(obj_uid, None)

    def list_fields(
        self, obj_uid: str | None = None, *, table: str = "default"
    ) -> list[str]:
        self._track_call("list_fields")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            if obj_uid is None:
                return list(self._store.keys())
            return [f for f, t in self._store.items() if obj_uid in t]

    def list_objs(self, *, table: str = "default") -> list[str]:
        self._track_call("list_objs")
        with self._lock:
            uids: set[str] = set()
            for data in self._store.values():
                uids.update(data.keys())
            return sorted(uids)

    def list_objs_has_field(
        self, field_name: str, *, table: str = "default"
    ) -> list[str]:
        self._track_call("list_objs_has_field")
        with self._lock:
            return sorted(self._store.get(field_name, {}).keys())

    def get_field_metadata(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> dict[str, Any] | None:
        self._track_call("get_field_metadata")
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            if field_name not in self._store or obj_uid not in self._store[field_name]:
                return None
            val = self._store[field_name][obj_uid]
            if isinstance(val, np.ndarray):
                return {"shape": list(val.shape), "dtype": str(val.dtype)}
            return None


class ThreadSafeCache:
    """Thread-safe fake cache for concurrent tests."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self._cache: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self._latency_s = latency_s
        self._threads: set[int] = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set_field(self, obj_uid: str, field_name: str, value: Any) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            self._cache[(obj_uid, field_name)] = value
        self._threads.add(threading.get_ident())

    def get_field(self, obj_uid: str, field_name: str) -> Any:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            return self._cache[(obj_uid, field_name)]

    def has_field(self, obj_uid: str, field_name: str) -> bool:
        with self._lock:
            return (obj_uid, field_name) in self._cache

    def erase_field(self, obj_uid: str, field_name: str) -> None:
        with self._lock:
            self._cache.pop((obj_uid, field_name), None)

    def reset_thread_ids(self) -> None:
        self._threads.clear()

    def get_thread_ids(self) -> set[int]:
        return set(self._threads)

    def list_uids(self, *, table: str = "default") -> list[str]:
        with self._lock:
            uids = set()
            for obj_uid, _ in self._cache:
                uids.add(obj_uid)
            return list(uids)

    def upsert(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = "default"
    ) -> None:
        self.set_field(obj_uid, field_name, value)


class ThreadSafeMetadataIndex:
    """Thread-safe fake metadata index for concurrent tests."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self._tables: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._latency_s = latency_s

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def define_table(
        self,
        table: str,
        columns: dict[str, type],
        indexes: list[str] | None = None,
    ) -> None:
        with self._lock:
            if table not in self._tables:
                self._tables[table] = {}

    def insert(self, table: str, uid: str, **fields: Any) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            if table not in self._tables:
                self._tables[table] = {}
            self._tables[table][uid] = {"uid": uid, **fields}

    def update(self, table: str, uid: str, **fields: Any) -> None:
        with self._lock:
            if table in self._tables and uid in self._tables[table]:
                self._tables[table][uid].update(fields)

    def upsert(self, table: str, uid: str, **fields: Any) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            if table not in self._tables:
                self._tables[table] = {}
            if uid in self._tables[table]:
                self._tables[table][uid].update(fields)
            else:
                self._tables[table][uid] = {"uid": uid, **fields}

    def get(self, table: str, uid: str) -> dict[str, Any] | None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        with self._lock:
            if table not in self._tables:
                return None
            return self._tables[table].get(uid)

    def query(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        where_in: dict[str, list[Any]] | None = None,
        where_like: dict[str, str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        with self._lock:
            if table not in self._tables:
                return []

            results = []
            for uid, record in self._tables[table].items():
                match = True

                if where:
                    for col, val in where.items():
                        if record.get(col) != val:
                            match = False
                            break

                if match and where_in:
                    for col, vals in where_in.items():
                        if col == "uid":
                            if uid not in vals:
                                match = False
                                break
                        elif record.get(col) not in vals:
                            match = False
                            break

                if match:
                    results.append(uid)

            if limit is not None:
                results = results[:limit]

            return results

    def delete(self, table: str, uid: str) -> None:
        with self._lock:
            if table in self._tables:
                self._tables[table].pop(uid, None)

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        with self._lock:
            if table not in self._tables:
                return 0
            if where is None:
                return len(self._tables[table])
            return len(self.query(table, where=where))

    def distinct(self, table: str, column: str) -> list[Any]:
        with self._lock:
            if table not in self._tables:
                return []
            values = set()
            for record in self._tables[table].values():
                if column in record:
                    values.add(record[column])
            return list(values)

    def list_uids(self, table: str) -> list[str]:
        with self._lock:
            if table not in self._tables:
                return []
            return list(self._tables[table].keys())

    def clear(self, table: str | None = None) -> None:
        with self._lock:
            if table is None:
                self._tables.clear()
            elif table in self._tables:
                self._tables[table].clear()

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def thread_safe_managers():
    return ThreadSafeStorage(), ThreadSafeCache(), ThreadSafeMetadataIndex()


@pytest.fixture
def slow_managers():
    """Managers with artificial latency to test parallelism."""
    return (
        ThreadSafeStorage(latency_s=0.01),
        ThreadSafeCache(latency_s=0.01),
        ThreadSafeMetadataIndex(latency_s=0.01),
    )


@pytest.fixture
def parallel_4() -> ParallelContext:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    calibration = calibrate_thread_pool_overhead(
        submit=executor.submit,
        workers=4,
    )
    try:
        yield ParallelContext(executor=executor, calibration=calibration, workers=4)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _create_param(
    name: str,
    model_id: str,
    storage: ThreadSafeStorage,
    cache: ThreadSafeCache,
    metadata_index: ThreadSafeMetadataIndex,
    ptype: ParameterType = ParameterType.DENSE,
) -> tuple[ParameterDataProxy, ParameterRepository]:
    """Create a parameter and return both the proxy and its repository."""
    meta = ParameterMetadata(name=name, ptype=ptype, model_id=model_id)
    repository = ParameterRepository.initialize(storage, metadata_index, cache)
    proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    return proxy, repository


class TestListFieldsByUidConcurrent:
    """Tests for concurrent list_fields_by_uid."""

    def test_parallel_execution_uses_multiple_threads(
        self, slow_managers: tuple, parallel_4: ParallelContext
    ) -> None:
        """Verify field listing runs across multiple worker threads."""
        storage, cache, metadata_index = slow_managers
        n_params = 10

        # Create a single repository for all params
        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        params = []
        for i in range(n_params):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
            storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))
            storage.set_field(p.meta.uid, "bias", np.ones((2,)))
            params.append(p)

        view = collection.create_view()

        storage.reset_thread_ids()
        result = view.list_fields_by_uid(parallel=parallel_4)

        assert len(result) == n_params
        for uid in result:
            assert "weights" in result[uid]
            assert "bias" in result[uid]

        thread_ids = storage.get_thread_ids("list_fields")
        assert len(thread_ids) > 1

    def test_empty_collection(self, thread_safe_managers: tuple) -> None:
        """Empty collection returns empty mapping."""
        storage, cache, metadata_index = thread_safe_managers
        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        result = collection.create_view().list_fields_by_uid()
        assert result == {}

    def test_single_param(self, thread_safe_managers: tuple) -> None:
        """Single param works correctly."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta = ParameterMetadata(name="p0", ptype=ParameterType.DENSE, model_id="m1")
        p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
        storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))

        result = collection.create_view().list_fields_by_uid()
        assert len(result) == 1
        assert p.meta.uid in result
        assert "weights" in result[p.meta.uid]

    def test_handles_exceptions_gracefully(self, thread_safe_managers: tuple) -> None:
        """Exceptions in list_fields don't crash the whole operation."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)

        # Create normal param
        meta1 = ParameterMetadata(name="p1", ptype=ParameterType.DENSE, model_id="m1")
        p1 = ParameterDataProxy.create_and_store(meta=meta1, repository=collection)
        storage.set_field(p1.meta.uid, "weights", np.ones((2, 2)))

        # Create second param
        meta2 = ParameterMetadata(name="p2", ptype=ParameterType.DENSE, model_id="m1")
        p2 = ParameterDataProxy.create_and_store(meta=meta2, repository=collection)

        result = collection.create_view().list_fields_by_uid()
        assert p1.meta.uid in result
        assert "weights" in result[p1.meta.uid]
        # p2 should be present with whatever fields it has
        assert p2.meta.uid in result


class TestPrefetchFieldsConcurrent:
    """Tests for concurrent prefetch_fields."""

    def test_parallel_prefetch_uses_multiple_threads(
        self, slow_managers: tuple, parallel_4: ParallelContext
    ) -> None:
        """Verify prefetch runs across multiple worker threads."""
        storage, cache, metadata_index = slow_managers
        n_params = 10

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        params = []
        fields_by_uid = {}
        for i in range(n_params):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
            storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))
            storage.set_field(p.meta.uid, "bias", np.ones((2,)))
            fields_by_uid[p.meta.uid] = ["weights", "bias"]
            params.append(p)

        view = collection.create_view()

        storage.reset_thread_ids()
        cache.reset_thread_ids()
        view.prefetch_fields(fields_by_uid=fields_by_uid, parallel=parallel_4)

        # All fields should be prefetched
        for p in params:
            assert cache.has_field(p.meta.uid, "weights")
            assert cache.has_field(p.meta.uid, "bias")

        assert len(storage.get_thread_ids("get_field")) > 1
        assert len(cache.get_thread_ids()) > 1

    def test_empty_fields_by_uid(self, thread_safe_managers: tuple) -> None:
        """Empty input returns without error."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta = ParameterMetadata(name="p0", ptype=ParameterType.DENSE, model_id="m1")
        ParameterDataProxy.create_and_store(meta=meta, repository=collection)
        view = collection.create_view()

        # Should not raise
        view.prefetch_fields(fields_by_uid={})

    def test_missing_uid_ignored(self, thread_safe_managers: tuple) -> None:
        """UIDs not in collection are ignored."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta = ParameterMetadata(name="p0", ptype=ParameterType.DENSE, model_id="m1")
        p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
        storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))

        view = collection.create_view()

        # Request prefetch for non-existent uid
        view.prefetch_fields(fields_by_uid={"non_existent_uid": ["weights"]})
        # Should not crash, p's weights not prefetched
        assert not cache.has_field(p.meta.uid, "weights")

    def test_missing_field_handled(self, thread_safe_managers: tuple) -> None:
        """Missing fields are handled gracefully."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta = ParameterMetadata(name="p0", ptype=ParameterType.DENSE, model_id="m1")
        p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
        # Don't set 'weights' field

        view = collection.create_view()

        # Should not crash even though 'weights' doesn't exist
        view.prefetch_fields(fields_by_uid={p.meta.uid: ["weights"]})


class TestFilterByFieldsConcurrent:
    """Tests for concurrent filter_by_fields."""

    def test_parallel_filter_uses_multiple_threads(
        self, slow_managers: tuple, parallel_4: ParallelContext
    ) -> None:
        """Verify filtering runs across multiple worker threads."""
        storage, cache, metadata_index = slow_managers
        n_params = 10

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        params = []
        for i in range(n_params):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
            # Only half have the 'weights' field
            if i % 2 == 0:
                storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))
            params.append(p)

        storage.reset_thread_ids()
        filtered = collection.create_view().filter_by_fields(
            "weights", parallel=parallel_4
        )

        assert len(filtered) == n_params // 2

        assert len(storage.get_thread_ids("has_field")) > 1

    def test_empty_fields_returns_all(self, thread_safe_managers: tuple) -> None:
        """No field filters returns all params."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        for i in range(3):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            ParameterDataProxy.create_and_store(meta=meta, repository=collection)

        filtered = collection.create_view().filter_by_fields()
        assert len(filtered) == 3

    def test_inverse_mask(self, thread_safe_managers: tuple) -> None:
        """Inverse mask excludes params with the field."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta1 = ParameterMetadata(name="p1", ptype=ParameterType.DENSE, model_id="m1")
        p1 = ParameterDataProxy.create_and_store(meta=meta1, repository=collection)
        meta2 = ParameterMetadata(name="p2", ptype=ParameterType.DENSE, model_id="m1")
        p2 = ParameterDataProxy.create_and_store(meta=meta2, repository=collection)
        storage.set_field(p1.meta.uid, "weights", np.ones((2, 2)))

        filtered = collection.create_view().filter_by_fields(
            "weights", inverse_mask=True
        )
        assert len(filtered) == 1
        assert next(iter(filtered)).meta.uid == p2.meta.uid


class TestPrefetchFieldsAllConcurrent:
    """Tests for concurrent prefetch_fields with shared fields."""

    def test_parallel_prefetch_all_fields_uses_multiple_threads(
        self, slow_managers: tuple, parallel_4: ParallelContext
    ) -> None:
        """All fields are prefetched using multiple worker threads."""
        storage, cache, metadata_index = slow_managers
        n_params = 8

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        params = []
        for i in range(n_params):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
            storage.set_field(p.meta.uid, "weights", np.ones((2, 2)))
            params.append(p)

        view = collection.create_view()

        storage.reset_thread_ids()
        cache.reset_thread_ids()
        ok = view.prefetch_fields(fields=["weights"], parallel=parallel_4)

        assert ok is True
        for p in params:
            assert cache.has_field(p.meta.uid, "weights")

        assert len(storage.get_thread_ids("get_field")) > 1
        assert len(cache.get_thread_ids()) > 1

    def test_returns_false_on_missing_field(self, thread_safe_managers: tuple) -> None:
        """Returns False if any param lacks the field."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta1 = ParameterMetadata(name="p1", ptype=ParameterType.DENSE, model_id="m1")
        p1 = ParameterDataProxy.create_and_store(meta=meta1, repository=collection)
        meta2 = ParameterMetadata(name="p2", ptype=ParameterType.DENSE, model_id="m1")
        ParameterDataProxy.create_and_store(meta=meta2, repository=collection)
        storage.set_field(p1.meta.uid, "weights", np.ones((2, 2)))
        # p2 doesn't have weights

        ok = collection.create_view().prefetch_fields(
            fields=["weights"], verify_prefetch=True
        )
        assert ok is False


class TestIterChunksByReadBudget:
    """Tests for iter_chunks_by_read_budget."""

    def test_chunks_by_size(self, thread_safe_managers: tuple) -> None:
        """Chunks are created based on estimated field sizes."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        required_by_uid = {}
        params = []
        for i in range(5):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            p = ParameterDataProxy.create_and_store(meta=meta, repository=collection)
            # Each param has 100 float32 values = 400 bytes
            storage.set_field(
                p.meta.uid, "weights", np.ones((10, 10), dtype=np.float32)
            )
            required_by_uid[p.meta.uid] = ["weights"]
            params.append(p)

        view = collection.create_view()

        # Budget of 1000 bytes should fit ~2 params per chunk (400 each)
        chunks = list(
            view.iter_chunks_by_read_budget(
                budget_bytes=1000,
                required_fields_by_uid=required_by_uid,
            )
        )

        assert len(chunks) >= 2
        total_params = sum(len(c) for c in chunks)
        assert total_params == 5

    def test_empty_collection(self, thread_safe_managers: tuple) -> None:
        """Empty collection yields nothing."""
        storage, cache, metadata_index = thread_safe_managers
        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        view = collection.create_view()
        chunks = list(
            view.iter_chunks_by_read_budget(
                budget_bytes=1000, required_fields_by_uid={}
            )
        )
        assert chunks == []

    def test_zero_budget_yields_all_at_once(self, thread_safe_managers: tuple) -> None:
        """Zero budget yields entire collection as one chunk."""
        storage, cache, metadata_index = thread_safe_managers

        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        for i in range(3):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            ParameterDataProxy.create_and_store(meta=meta, repository=collection)

        view = collection.create_view()

        chunks = list(
            view.iter_chunks_by_read_budget(budget_bytes=0, required_fields_by_uid={})
        )
        assert len(chunks) == 1
        assert len(chunks[0]) == 3


class TestLoadFromStorageConcurrent:
    """Tests for repository initialization."""

    def test_parallel_load(self, slow_managers: tuple) -> None:
        """Creating parameters happens without blocking."""
        storage, cache, metadata_index = slow_managers
        n_params = 10

        # Create collection and populate with parameters
        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        start = time.perf_counter()
        for i in range(n_params):
            meta = ParameterMetadata(
                name=f"p{i}", ptype=ParameterType.DENSE, model_id="m1"
            )
            ParameterDataProxy.create_and_store(meta=meta, repository=collection)
        creation_time = time.perf_counter() - start

        assert len(collection) == n_params
        # Creation performs no extra lookups, so latency stays bounded
        assert creation_time < n_params * 0.05

    def test_handles_params_correctly(self, thread_safe_managers: tuple) -> None:
        """Parameters are correctly stored and retrieved."""
        storage, cache, metadata_index = thread_safe_managers

        # Create collection and add valid param
        collection = ParameterRepository.initialize(storage, metadata_index, cache)
        meta1 = ParameterMetadata(name="p1", ptype=ParameterType.DENSE, model_id="m1")
        ParameterDataProxy.create_and_store(meta=meta1, repository=collection)

        assert len(collection) == 1
        assert next(iter(collection)).meta.uid == meta1.uid
