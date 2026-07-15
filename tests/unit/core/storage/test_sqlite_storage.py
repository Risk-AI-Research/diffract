import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from diffract.core.storage.sqlite_manager import ConnectionPool, SQLiteStorageManager


def _wait_for_path_gone(path: str | Path, timeout_s: float = 2.0) -> None:
    p = Path(path)
    deadline = time.time() + timeout_s
    while p.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert not p.exists(), f"File still exists after timeout: {p}"


@pytest.fixture
def db_path(temp_dir: Path) -> str:
    return str(temp_dir / "store.sqlite")


@pytest.fixture
def storage(db_path: str) -> SQLiteStorageManager:
    s = SQLiteStorageManager(path=db_path)
    s.connect()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def storage_external(db_path: str) -> SQLiteStorageManager:
    # Force external file storage for any ndarray.
    s = SQLiteStorageManager(
        path=db_path, array_threshold=1, array_dir="sqlite_blobs_test"
    )
    s.connect()
    try:
        yield s
    finally:
        s.close()


def _fetch_row(db_path: str, uid: str, field: str) -> tuple[str, bytes, str | None]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT value_type, value_data, file_path FROM storage "
            "WHERE field = ? AND obj_uid = ?",
            (field, uid),
        ).fetchone()
        assert row is not None
        return row[0], row[1], row[2]
    finally:
        conn.close()


def test_schema_and_indexes_created(
    storage: SQLiteStorageManager, db_path: str
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "storage" in tables
        assert "obj_registry" in tables
        assert "field_registry" in tables
        indexes = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert "idx_table_field" in indexes
        assert "idx_table_obj" in indexes
    finally:
        conn.close()


def test_set_get_json(storage: SQLiteStorageManager) -> None:
    uid = "u1"
    meta = {"a": 1, "b": [1, 2, 3], "s": "ok"}

    storage.set_field(uid, "__metadata__", meta)
    assert storage.has_field(uid, "__metadata__") is True
    assert storage.get_field(uid, "__metadata__") == meta


def test_set_unserializable_value_raises(storage: SQLiteStorageManager) -> None:
    uid = "u_unserializable"
    field = "v"
    value = {1, 2, 3}  # neither an ndarray, bytes, nor JSON-serializable

    with pytest.raises(ValueError, match="cannot serialize"):
        storage.set_field(uid, field, value)


def test_set_get_ndarray_inline_in_db(
    storage: SQLiteStorageManager, db_path: str
) -> None:
    uid = "u_arr_inline"
    field = "weights"
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((16, 8)).astype(np.float32) * 0.1).copy()

    storage.set_field(uid, field, arr)
    got = storage.get_field(uid, field)
    np.testing.assert_allclose(got, arr)

    dtype, value_data, file_path = _fetch_row(db_path, uid, field)
    assert dtype == "ndarray"
    assert isinstance(value_data, (bytes, bytearray))
    assert len(value_data) > 0
    assert file_path is None


def test_get_field_metadata(storage: SQLiteStorageManager) -> None:
    uid = "meta1"
    field = "weights"
    arr = np.ones((2, 3), dtype=np.float32)

    storage.set_field(uid, field, arr)
    meta = storage.get_field_metadata(uid, field)

    assert meta is not None
    assert meta.get("kind") == "matrix"
    assert tuple(meta.get("shape") or []) == (2, 3)
    assert meta.get("dtype") == "float32"


def test_set_get_ndarray_external_file(
    storage_external: SQLiteStorageManager, db_path: str
) -> None:
    uid = "u_arr_file"
    field = "weights"
    arr = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)

    storage_external.set_field(uid, field, arr)
    got = storage_external.get_field(uid, field)
    np.testing.assert_allclose(got, arr)

    dtype, value_data, file_path = _fetch_row(db_path, uid, field)
    assert dtype == "ndarray"
    assert value_data == b""
    assert file_path is not None
    assert Path(file_path).exists()


def test_async_file_deletion_on_erase_field(
    storage_external: SQLiteStorageManager, db_path: str
) -> None:
    uid = "u_del_field"
    field = "big"
    arr = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)

    storage_external.set_field(uid, field, arr)
    _, _, file_path = _fetch_row(db_path, uid, field)
    assert file_path is not None
    assert Path(file_path).exists()

    storage_external.erase_field(uid, field)
    assert storage_external.has_field(uid, field) is False

    _wait_for_path_gone(file_path)


def test_async_file_deletion_on_erase_obj(
    storage_external: SQLiteStorageManager, db_path: str
) -> None:
    uid = "u_del_obj"
    rng = np.random.default_rng(0)
    storage_external.set_field(uid, "a", 1)
    storage_external.set_field(uid, "b", rng.standard_normal((8, 8)).astype(np.float32))
    storage_external.set_field(uid, "c", {"x": 1})

    _, _, file_path = _fetch_row(db_path, uid, "b")
    assert file_path is not None
    assert Path(file_path).exists()

    storage_external.erase_obj(uid)
    assert uid not in storage_external.list_objs()
    assert storage_external.has_field(uid, "a") is False
    assert storage_external.has_field(uid, "b") is False
    assert storage_external.has_field(uid, "c") is False

    _wait_for_path_gone(file_path)


def test_erase_field_for_all(storage: SQLiteStorageManager) -> None:
    storage.set_field("q1", "dead", 1)
    storage.set_field("q2", "dead", 2)
    storage.set_field("q3", "alive", 3)

    storage.erase_field_for_all("dead")

    assert storage.has_field("q1", "dead") is False
    assert storage.has_field("q2", "dead") is False
    assert storage.has_field("q3", "alive") is True


def test_clear_removes_all_and_deletes_files(
    storage_external: SQLiteStorageManager, db_path: str
) -> None:
    rng = np.random.default_rng(0)
    storage_external.set_field(
        "m1", "f", rng.standard_normal((8, 8)).astype(np.float32)
    )
    storage_external.set_field(
        "m2", "f", rng.standard_normal((8, 8)).astype(np.float32)
    )

    _, _, fp1 = _fetch_row(db_path, "m1", "f")
    _, _, fp2 = _fetch_row(db_path, "m2", "f")
    assert fp1 is not None
    assert Path(fp1).exists()
    assert fp2 is not None
    assert Path(fp2).exists()

    storage_external.clear()
    assert storage_external.list_objs() == []

    _wait_for_path_gone(fp1)
    _wait_for_path_gone(fp2)


def test_list_fields_and_objs(storage: SQLiteStorageManager) -> None:
    uids = ["a", "b", "c"]
    for u in uids:
        storage.set_field(u, "f1", 1)
        storage.set_field(u, "f2", 2)

    assert {"f1", "f2"}.issubset(set(storage.list_fields("a")))
    assert set(uids).issubset(set(storage.list_objs()))


def test_registry_consistency(storage: SQLiteStorageManager, db_path: str) -> None:
    """Test that obj_registry and field_registry stay consistent with storage."""
    # Add some data
    storage.set_field("obj1", "field_a", 1)
    storage.set_field("obj1", "field_b", 2)
    storage.set_field("obj2", "field_a", 3)

    # Check registries via list_* methods (which use registries)
    assert set(storage.list_objs()) == {"obj1", "obj2"}
    assert set(storage.list_fields()) == {"field_a", "field_b"}

    # Erase one field - field_b should be removed from registry since only obj1 had it
    storage.erase_field("obj1", "field_b")
    assert set(storage.list_fields()) == {"field_a"}

    # Erase one object
    storage.erase_obj("obj2")
    assert set(storage.list_objs()) == {"obj1"}

    # Erase field for all - should remove from registry
    storage.set_field("obj1", "to_remove", 100)
    storage.set_field("obj3", "to_remove", 200)
    assert "to_remove" in storage.list_fields()
    storage.erase_field_for_all("to_remove")
    assert "to_remove" not in storage.list_fields()

    # Clear all - registries should be empty
    storage.clear()
    assert storage.list_objs() == []
    assert storage.list_fields() == []


def test_overwrite_field(storage: SQLiteStorageManager) -> None:
    uid = "u_overwrite"
    storage.set_field(uid, "v", 1)
    storage.set_field(uid, "v", 2)
    assert storage.get_field(uid, "v") == 2


def test_batch_context_commits_and_allows_reads(storage: SQLiteStorageManager) -> None:
    uid = "u_batch"
    with storage:
        storage.set_field(uid, "a", 1)
        storage.set_field(uid, "b", 2)
        # Note: reads are executed via the read-only pool (separate connections),
        # so they do not see uncommitted writes inside the active transaction.
        assert storage.has_field(uid, "a") is False
        storage.erase_field(uid, "b")
    assert storage.has_field(uid, "a") is True
    assert storage.has_field(uid, "b") is False


def test_nested_batch_contexts(storage: SQLiteStorageManager) -> None:
    """Test that nested context managers work correctly."""
    uid = "u_nested"
    with storage:
        storage.set_field(uid, "a", 1)
        with storage:  # Nested context
            storage.set_field(uid, "b", 2)
            storage.set_field(uid, "c", 3)
        # Inner context exited, but outer still active
        storage.set_field(uid, "d", 4)
    # All fields should be committed after outermost context exits
    assert storage.get_field(uid, "a") == 1
    assert storage.get_field(uid, "b") == 2
    assert storage.get_field(uid, "c") == 3
    assert storage.get_field(uid, "d") == 4


def test_nested_batch_context_rollback_on_exception(
    storage: SQLiteStorageManager,
) -> None:
    """Test that exceptions in nested contexts properly rollback."""
    uid = "u_nested_exception"

    def _write_then_raise() -> None:
        with storage:
            storage.set_field(uid, "a", 1)
            with storage:  # Nested context
                storage.set_field(uid, "b", 2)
                raise ValueError("Test exception")

    with pytest.raises(ValueError, match="Test exception"):
        _write_then_raise()

    # All changes should be rolled back
    assert storage.has_field(uid, "a") is False
    assert storage.has_field(uid, "b") is False


def test_readonly_mode_allows_reads_but_rejects_writes(db_path: str) -> None:
    s1 = SQLiteStorageManager(path=db_path)
    s1.connect()
    s1.set_field("r1", "k", {"a": 1})
    s1.close()

    s2 = SQLiteStorageManager(path=db_path, readonly=True)
    s2.connect()
    try:
        assert s2.get_field("r1", "k") == {"a": 1}
        with pytest.raises(OSError):
            s2.set_field("r1", "k2", 123)
    finally:
        s2.close()


def test_connection_pool_readonly_and_overflow(db_path: str) -> None:
    # Create DB schema + one row.
    s = SQLiteStorageManager(path=db_path)
    s.connect()
    s.set_field("x", "y", 1)
    s.close()

    pool = ConnectionPool(
        path=db_path,
        pool_size=1,
        timeout=1.0,
        pool_acquire_timeout=0.1,
        max_overflow=2,
    )
    pool.connect()
    try:
        got_second = threading.Event()
        errors: list[BaseException] = []

        with pool.acquire() as conn1:
            # Ensure read-only (query_only).
            cur1 = conn1.cursor()
            assert cur1.execute("PRAGMA query_only").fetchone()[0] == 1

            # Second acquire should succeed quickly via overflow connection.
            def _worker() -> None:
                try:
                    with pool.acquire() as conn2:
                        cur2 = conn2.cursor()
                        assert cur2.execute("SELECT 1").fetchone()[0] == 1
                        assert cur2.execute("PRAGMA query_only").fetchone()[0] == 1
                        # Writes should fail in query_only mode.
                        try:
                            cur2.execute("CREATE TABLE should_fail(x INT)")
                        except sqlite3.OperationalError:
                            pass
                        else:
                            raise AssertionError(
                                "Write succeeded on query_only connection"
                            )
                        got_second.set()
                except BaseException as e:  # noqa: BLE001 - test helper
                    errors.append(e)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            assert got_second.wait(2.0), "Second acquire did not succeed in time"
            t.join(timeout=1.0)

        assert not errors, f"Errors in worker thread: {errors!r}"
    finally:
        pool.close_all()


def test_concurrent_reads_from_pool(db_path: str) -> None:
    """Test multiple threads can read concurrently from the connection pool."""
    storage = SQLiteStorageManager(path=db_path, read_pool_size=4)
    storage.connect()
    try:
        # Populate data
        for i in range(100):
            storage.set_field(f"uid_{i}", "value", i)

        n_readers = 8
        n_reads_per_thread = 50
        results: list[list[int]] = [[] for _ in range(n_readers)]
        errors: list[BaseException] = []
        barrier = threading.Barrier(n_readers)

        def _reader(thread_idx: int) -> None:
            try:
                barrier.wait()  # Start all threads at once
                for _ in range(n_reads_per_thread):
                    uid_idx = (thread_idx * n_reads_per_thread + _) % 100
                    val = storage.get_field(f"uid_{uid_idx}", "value")
                    results[thread_idx].append(val)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=_reader, args=(i,)) for i in range(n_readers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Errors in reader threads: {errors!r}"
        for i, r in enumerate(results):
            assert len(r) == n_reads_per_thread, f"Thread {i} got {len(r)} results"
    finally:
        storage.close()


def test_concurrent_reads_during_write_transaction(db_path: str) -> None:
    """Test reads can proceed while a write transaction is active (WAL mode)."""
    storage = SQLiteStorageManager(path=db_path, wal_mode=True, read_pool_size=4)
    storage.connect()
    try:
        # Initial data
        storage.set_field("uid_0", "value", 0)

        read_completed = threading.Event()
        errors: list[BaseException] = []

        def _reader() -> None:
            try:
                # Read should work even during write transaction
                val = storage.get_field("uid_0", "value")
                assert val == 0
                read_completed.set()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        with storage:
            # Start write transaction
            storage.set_field("uid_1", "value", 1)
            # Reader should still be able to read
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            assert read_completed.wait(2.0), (
                "Read did not complete during write transaction"
            )
            t.join(timeout=1.0)

        assert not errors, f"Errors in reader thread: {errors!r}"
    finally:
        storage.close()


def test_parallel_blob_writes(db_path: str) -> None:
    """Test parallel blob file writes work correctly."""
    storage = SQLiteStorageManager(
        path=db_path,
        array_threshold=1,  # Force external file storage
        blob_write_workers=4,
    )
    storage.connect()
    try:
        n_arrays = 10
        rng = np.random.default_rng(0)
        arrays = [
            rng.standard_normal((32, 32)).astype(np.float32) for _ in range(n_arrays)
        ]

        # Write all arrays in one batch context
        with storage:
            for i, arr in enumerate(arrays):
                storage.set_field(f"uid_{i}", "weights", arr)

        # Verify all arrays were written correctly
        for i, expected in enumerate(arrays):
            got = storage.get_field(f"uid_{i}", "weights")
            np.testing.assert_allclose(got, expected)

        # Verify files were created
        for i in range(n_arrays):
            _, _, file_path = _fetch_row(db_path, f"uid_{i}", "weights")
            assert file_path is not None
            assert Path(file_path).exists()
    finally:
        storage.close()


def test_auto_flush_on_batch_size_limit(db_path: str) -> None:
    """Test auto-flush triggers when batch size exceeds soft limit."""
    # Small batch limit to trigger auto-flush quickly
    storage = SQLiteStorageManager(
        path=db_path,
        batch_size_limit_bytes=1024,  # 1KB
        batch_soft_limit_ratio=0.5,  # 512 bytes triggers flush
    )
    storage.connect()
    try:
        with storage:
            # First write - should not trigger flush yet
            storage.set_field("uid_1", "small", {"a": 1})

            # Large write that should trigger auto-flush
            large_arr = np.ones((100,), dtype=np.float32)  # 400 bytes
            storage.set_field("uid_2", "large", large_arr)

            # Even more data - should trigger another flush
            storage.set_field("uid_3", "large2", large_arr)

        # Verify all data was written
        assert storage.get_field("uid_1", "small") == {"a": 1}
        np.testing.assert_allclose(storage.get_field("uid_2", "large"), large_arr)
        np.testing.assert_allclose(storage.get_field("uid_3", "large2"), large_arr)
    finally:
        storage.close()


def test_batch_size_tracking_accuracy(db_path: str) -> None:
    """Test that batch size tracking handles overwrites correctly."""
    storage = SQLiteStorageManager(
        path=db_path,
        batch_size_limit_bytes=10 * 1024 * 1024,  # Large limit
    )
    storage.connect()
    try:
        with storage:
            # First write
            storage.set_field("uid", "field", np.ones((100,), dtype=np.float32))
            first_size = storage._pending_set_bytes

            # Overwrite with smaller value
            storage.set_field("uid", "field", {"small": 1})
            second_size = storage._pending_set_bytes

            # Size should decrease
            assert second_size < first_size

            # Overwrite with larger value
            storage.set_field("uid", "field", np.ones((1000,), dtype=np.float32))
            third_size = storage._pending_set_bytes

            # Size should increase
            assert third_size > second_size
    finally:
        storage.close()


def test_blob_executor_shutdown_on_close(db_path: str) -> None:
    """Test blob executor is properly shut down when storage is closed."""
    storage = SQLiteStorageManager(
        path=db_path,
        array_threshold=1,
        blob_write_workers=2,
    )
    storage.connect()
    try:
        # Trigger blob executor creation
        storage.set_field("uid", "arr", np.ones((10, 10), dtype=np.float32))
        assert storage._blob_executor is not None
    finally:
        storage.close()

    # After close, executor should be None
    assert storage._blob_executor is None
