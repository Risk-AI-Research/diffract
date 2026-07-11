"""SQLite storage manager with optimized multithreaded reading support.

This module provides SQLite-based storage with connection pooling for
concurrent read operations while maintaining serialized writes.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager, suppress
from io import BytesIO
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from diffract.core.utils.exceptions import format_exception_message

from .base_manager import BaseStorageManager
from .interface import DEFAULT_TABLE, UID
from .metadata import infer_value_metadata

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe connection pool for read-only SQLite connections.

    Maintains a pool of read-only connections that can be safely shared
    across multiple threads for concurrent read operations.
    """

    def __init__(
        self,
        path: str,
        pool_size: int = 8,
        timeout: float = 5.0,
        read_cache_size_kb: int = 32000,
        read_mmap_size: int = 134217728,
        pool_acquire_timeout: float = 1.0,
        max_overflow: int = 2,
    ) -> None:
        """Initialize connection pool.

        Args:
            path: Path to SQLite database file.
            pool_size: Maximum number of connections to maintain.
            timeout: Timeout for database operations in seconds.
            read_cache_size_kb: Cache size in kilobytes for read connections.
            read_mmap_size: Memory-mapped I/O size in bytes for read connections.
            pool_acquire_timeout: Timeout for acquiring connection from pool.
            max_overflow: Maximum overflow multiplier for temporary connections.
        """
        self._path = path
        self._pool_size = pool_size
        self._timeout = timeout
        self._read_cache_size_kb = read_cache_size_kb
        self._read_mmap_size = read_mmap_size
        self._pool_acquire_timeout = pool_acquire_timeout
        self._max_overflow = max_overflow
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._created_connections = 0
        self._lock = threading.Lock()
        self._initialized = False

    def _initialize_pool(self) -> None:
        """Create initial pool of read-only connections."""
        for _ in range(self._pool_size):
            try:
                conn = self._create_connection()
                self._pool.put(conn, block=False)
            except (OSError, sqlite3.Error) as e:
                logger.warning(
                    "Failed to pre-create connection: %s",
                    format_exception_message(e),
                )
                break

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new read-only connection.

        Returns:
            Configured read-only SQLite connection.
        """
        uri = f"file:{self._path}?mode=ro"

        conn = sqlite3.connect(
            uri,
            check_same_thread=False,
            uri=True,
            timeout=self._timeout,
            isolation_level=None,
        )

        with closing(conn.cursor()) as cur:
            cur.execute("PRAGMA query_only = ON")
            cur.execute("PRAGMA temp_store = MEMORY")
            cur.execute(f"PRAGMA cache_size = -{self._read_cache_size_kb}")
            cur.execute("PRAGMA page_size = 4096")
            cur.execute(f"PRAGMA mmap_size = {self._read_mmap_size}")

        with self._lock:
            self._created_connections += 1

        return conn

    @contextmanager
    def acquire(self) -> Generator[sqlite3.Connection, None, None]:
        """Acquire a connection from the pool.

        Yields:
            Read-only SQLite connection.

        Raises:
            TimeoutError: If connection cannot be acquired within timeout.
        """
        conn = None
        try:
            try:
                conn = self._pool.get(timeout=self._pool_acquire_timeout)
            except Empty:
                with self._lock:
                    can_create = (
                        self._created_connections < self._pool_size * self._max_overflow
                    )
                if can_create:
                    conn = self._create_connection()
                else:
                    conn = self._pool.get(timeout=self._timeout)

            if not self._validate_connection(conn):
                conn.close()
                conn = self._create_connection()

            yield conn

        finally:
            if conn is not None:
                try:
                    self._pool.put(conn, block=False)
                except Full:
                    conn.close()

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is still valid.

        Args:
            conn: Connection to validate.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            _ = conn.total_changes
        except sqlite3.Error:
            return False
        else:
            return True

    def connect(self) -> None:
        """Initialize the connection pool.

        This method is idempotent - calling it multiple times has no effect
        after the first call.
        """
        if not self._initialized:
            self._initialize_pool()
            self._initialized = True

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get(block=False)
            except Empty:
                break
            with suppress(sqlite3.Error, OSError):
                conn.close()
        # Reset initialized flag so connect() can reinitialize the pool
        self._initialized = False
        with self._lock:
            self._created_connections = 0

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with suppress(Exception):
            self.close_all()


class SQLiteStorageManager(BaseStorageManager):
    """SQLite storage manager with optimized concurrent read support.

    This implementation uses a connection pool for read operations and a
    dedicated write connection with locking for write operations.

    Read operations (concurrent):
        - get_field, has_field, list_* methods
        - Use read-only connection pool
        - No locking required
        - Safe for parallel execution

    Write operations (serialized):
        - set_field, erase_*, clear
        - Use dedicated write connection
        - Protected by write lock
        - WAL mode allows concurrent reads during writes
    """

    def __init__(
        self,
        path: str,
        *,
        wal_mode: bool = True,
        synchronous: str = "NORMAL",
        cache_size_kb: int = 64000,
        use_json: bool = True,
        readonly: bool = False,
        timeout: float = 5.0,
        read_pool_size: int = 8,
        array_threshold: int = 128 * 1024 * 1024,
        array_dir: str = "sqlite_blobs",
        write_mmap_size: int = 268435456,
        read_cache_size_kb: int = 32000,
        read_mmap_size: int = 134217728,
        pool_acquire_timeout: float = 1.0,
        max_overflow: int = 2,
        blob_write_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize SQLite storage manager.

        Args:
            path: Path to SQLite database file.
            wal_mode: Enable Write-Ahead Logging for better concurrency.
            synchronous: SQLite synchronous mode (NORMAL, FULL, OFF).
            cache_size_kb: Cache size in kilobytes for write connection.
            use_json: Prefer JSON serialization when possible.
            readonly: Open database in read-only mode.
            timeout: Database operation timeout in seconds.
            read_pool_size: Number of read-only connections in pool.
            array_threshold: Threshold in bytes for storing arrays in external files.
            array_dir: Directory name for storing large array files.
            write_mmap_size: Memory-mapped I/O size in bytes for write connection.
            read_cache_size_kb: Cache size in kilobytes for read connections.
            read_mmap_size: Memory-mapped I/O size in bytes for read connections.
            pool_acquire_timeout: Timeout for acquiring connection from pool.
            max_overflow: Maximum overflow multiplier for temporary connections.
            batch_size_limit_bytes: Hard limit for buffer batch size before auto-flush.
            batch_soft_limit_ratio: Ratio of limit to trigger early flush (e.g., 0.9).
            blob_write_workers: Max workers for parallel blob file writes.
            **kwargs: Additional keyword arguments for BaseStorageManager.
        """
        super().__init__(use_json=use_json, **kwargs)

        self._path = path
        self._readonly = readonly
        self._write_lock = threading.RLock()  # Reentrant for nested contexts
        self._base_dir = Path(path).parent
        self._array_dir = self._base_dir / array_dir
        self._array_threshold = array_threshold

        self._wal_mode = wal_mode
        self._synchronous = synchronous
        self._cache_size_kb = cache_size_kb
        self._timeout = timeout
        self._write_mmap_size = write_mmap_size
        self._blob_write_workers = max(1, blob_write_workers)
        self._blob_executor: ThreadPoolExecutor | None = None

        if not readonly:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._array_dir.mkdir(parents=True, exist_ok=True)

        self._uri = f"file:{path}{'?mode=ro' if readonly else ''}"

        self._write_conn = None
        self._read_pool: ConnectionPool | None = (
            ConnectionPool(
                path=path,
                pool_size=read_pool_size,
                timeout=timeout,
                read_cache_size_kb=read_cache_size_kb,
                read_mmap_size=read_mmap_size,
                pool_acquire_timeout=pool_acquire_timeout,
                max_overflow=max_overflow,
            )
            if not readonly
            else None
        )

    def _apply_pragmas(self) -> None:
        """Apply SQLite pragmas to write connection."""
        pragmas = [
            ("journal_mode", "WAL") if self._wal_mode and not self._readonly else None,
            ("synchronous", self._synchronous),
            ("cache_size", -self._cache_size_kb),
            ("temp_store", "MEMORY"),
            ("page_size", 4096),
            ("mmap_size", self._write_mmap_size) if not self._readonly else None,
        ]
        with closing(self._write_conn.cursor()) as cur:
            for pragma in pragmas:
                if pragma:
                    key, value = pragma
                    cur.execute(f"PRAGMA {key}={value}")

    def _init_schema(self) -> None:
        """Initialize database schema with table support."""
        with closing(self._write_conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS storage (
                    table_name TEXT NOT NULL DEFAULT 'default',
                    field TEXT NOT NULL,
                    obj_uid TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_data BLOB NOT NULL,
                    file_path TEXT,
                    value_meta TEXT,
                    PRIMARY KEY (table_name, field, obj_uid)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_table_field "
                "ON storage(table_name, field)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_table_obj "
                "ON storage(table_name, obj_uid)"
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS obj_registry (
                    table_name TEXT NOT NULL DEFAULT 'default',
                    obj_uid TEXT NOT NULL,
                    PRIMARY KEY (table_name, obj_uid)
                ) WITHOUT ROWID
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS field_registry (
                    table_name TEXT NOT NULL DEFAULT 'default',
                    field TEXT NOT NULL,
                    PRIMARY KEY (table_name, field)
                ) WITHOUT ROWID
                """
            )

    def _serialize(self, value: Any) -> tuple[bytes, str]:
        """Serialize value to bytes.

        Args:
            value: Value to serialize.

        Returns:
            Tuple of (serialized bytes, type string).
        """
        if isinstance(value, np.ndarray):
            bio = BytesIO()
            np.save(bio, value, allow_pickle=False)
            return bio.getvalue(), "ndarray"
        if self._use_json:
            with suppress(TypeError, ValueError):
                return json.dumps(value, ensure_ascii=False).encode("utf-8"), "json"
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), "pickle"

    def _encode_value_meta(self, value: Any) -> str:
        """Encode value metadata into json format."""
        meta = infer_value_metadata(value)
        return json.dumps(meta.to_jsonable(), ensure_ascii=False)

    def _init_blob_executor(self) -> ThreadPoolExecutor:
        """Lazily initialize blob write executor."""
        if self._blob_executor is None:
            self._blob_executor = ThreadPoolExecutor(
                max_workers=self._blob_write_workers, thread_name_prefix="sqlite-blob"
            )
        return self._blob_executor

    def _submit_blob_write(self, file_path: str, data: bytes) -> Future[None]:
        """Submit blob write to executor."""
        executor = self._init_blob_executor()

        def _write() -> None:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                f.write(data)

        return executor.submit(_write)

    def _wait_for_blob_writes(self, futures: list[Future[None]]) -> None:
        """Wait for blob writes to complete and surface exceptions."""
        if not futures:
            return
        done, _ = wait(futures)
        for fut in done:
            exc = fut.exception()
            if exc:
                raise exc

    def _deserialize(self, data: bytes, dtype: str) -> Any:
        """Deserialize bytes to value.

        Args:
            data: Serialized bytes.
            dtype: Type string.

        Returns:
            Deserialized value.

        Raises:
            ValueError: If dtype is unknown.
        """
        if dtype == "ndarray":
            return np.load(BytesIO(data), allow_pickle=False)
        if dtype == "json":
            return json.loads(data.decode("utf-8"))
        if dtype == "pickle":
            return pickle.loads(data)  # noqa: S301
        msg = f"Unknown type: {dtype}"
        raise ValueError(msg)

    def _store_large_array(
        self,
        table: str,
        field: str,
        obj_uid: str,
        arr_bytes: bytes,
        futures: list[Future[None]],
    ) -> str:
        """Schedule storing large array to external file."""
        fname = f"{table}_{field}_{obj_uid}_{os.urandom(8).hex()}.npy"
        fpath = self._array_dir / fname
        futures.append(self._submit_blob_write(str(fpath), arr_bytes))
        return str(fpath)

    def _load_large_array(self, file_path: str) -> np.ndarray:
        """Load array from external file.

        Args:
            file_path: Path to array file.

        Returns:
            Loaded array.
        """
        path = Path(file_path)
        with path.open("rb") as f:
            return np.load(f, allow_pickle=False)

    def _delete_array_files(self, file_rows: list[tuple[str | None, str]]) -> None:
        """Delete array files asynchronously in background thread.

        Args:
            file_rows: List of (file_path, value_type) tuples from database.
        """
        if not file_rows:
            return

        file_paths = [
            file_path
            for file_path, dtype in file_rows
            if dtype == "ndarray" and file_path
        ]

        if not file_paths:
            return

        def _delete_files() -> None:
            for file_path in file_paths:
                with suppress(OSError):
                    Path(file_path).unlink()

        thread = threading.Thread(target=_delete_files, daemon=True)
        thread.start()

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Context manager for write transactions."""
        acquired = False
        try:
            if self._context_depth == 0:
                self._write_lock.acquire()
                acquired = True
                self._write_conn.execute("BEGIN IMMEDIATE")
            self._context_depth += 1
            yield
            self._context_depth -= 1
            if self._context_depth == 0 and acquired:
                self._write_conn.execute("COMMIT")
        except Exception:
            self._context_depth = max(0, self._context_depth - 1)
            if self._context_depth == 0 and acquired:
                self._write_conn.execute("ROLLBACK")
            raise
        finally:
            if self._context_depth == 0 and acquired:
                self._write_lock.release()

    def _execute_read(
        self, query: str, params: tuple = (), fetchall: bool = True
    ) -> list[tuple]:
        """Execute read query using connection pool.

        Args:
            query: SQL query string.
            params: Query parameters.
            fetchall: Whether to fetch all rows.

        Returns:
            Query results as list of tuples.

        Raises:
            OSError: On database errors.
        """
        try:
            if self._write_conn is None:
                self.connect()

            if self._read_pool is not None:
                with self._read_pool.acquire() as conn, closing(conn.cursor()) as cur:
                    if fetchall:
                        return cur.execute(query, params).fetchall()
                    return cur.execute(query, params).fetchone()
            else:
                with closing(self._write_conn.cursor()) as cur:
                    if fetchall:
                        return cur.execute(query, params).fetchall()
                    return cur.execute(query, params).fetchone()
        except sqlite3.Error as e:
            logger.exception("SQLite read error for query: %s", query)
            msg = f"SQLite error: {format_exception_message(e)}"
            raise OSError(msg) from e

    def _execute_write(self, query: str, params: tuple | tuple[tuple] = ()) -> None:
        """Execute write query with transaction and locking.

        Args:
            query: SQL query string.
            params: Query parameters.

        Raises:
            OSError: On database errors.
        """
        try:
            if self._write_conn is None:
                self.connect()

            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                if params == ():
                    cur.execute(query)
                    return
                if isinstance(params, (list, tuple)) and len(params) == 0:
                    return
                if params and isinstance(params[0], tuple):
                    cur.executemany(query, params)
                else:
                    cur.execute(query, params)

        except sqlite3.Error as e:
            logger.exception("SQLite write error for query: %s", query)
            msg = f"SQLite error: {format_exception_message(e)}"
            raise OSError(msg) from e

    def _has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Check if field exists.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            True if field exists, False otherwise.
        """
        query = (
            "SELECT 1 FROM storage "
            "WHERE table_name = ? AND field = ? AND obj_uid = ? LIMIT 1"
        )
        try:
            row = self._execute_read(
                query, (table, field_name, obj_uid), fetchall=False
            )
            return row is not None  # noqa: TRY300
        except OSError:
            return False

    def _get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Retrieve field value.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            Field value.

        Raises:
            KeyError: If field not found.
        """
        row = self._execute_read(
            "SELECT value_type, value_data, file_path FROM storage "
            "WHERE table_name = ? AND field = ? AND obj_uid = ?",
            (table, field_name, obj_uid),
            fetchall=False,
        )

        if row is None:
            msg = f"Field '{field_name}' of '{obj_uid}' not found"
            raise KeyError(msg)

        dtype, data, file_path = row
        if dtype == "ndarray" and file_path:
            return self._load_large_array(file_path)
        return self._deserialize(data, dtype)

    def _get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Return stored metadata for a field if present."""
        query = (
            "SELECT value_meta FROM storage "
            "WHERE table_name = ? AND field = ? AND obj_uid = ?"
        )
        try:
            row = self._execute_read(
                query, (table, field_name, obj_uid), fetchall=False
            )
            if row is None or row[0] is None:
                return None
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return None
        except OSError:
            return None

    def _list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List fields.

        Args:
            obj_uid: Optional object unique identifier. If None, lists all fields.
            table: Table name for logical data separation.

        Returns:
            List of field names.
        """
        if obj_uid is None:
            rows = self._execute_read(
                "SELECT field FROM field_registry WHERE table_name = ?", (table,)
            )
        else:
            rows = self._execute_read(
                "SELECT field FROM storage WHERE table_name = ? AND obj_uid = ?",
                (table, obj_uid),
            )

        return [row[0] for row in rows]

    def _list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """List all objects.

        Args:
            table: Table name for logical data separation.

        Returns:
            List of object unique identifiers.
        """
        rows = self._execute_read(
            "SELECT obj_uid FROM obj_registry WHERE table_name = ?", (table,)
        )
        return [row[0] for row in rows]

    def _list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]:
        """List objects with specific field.

        Args:
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            List of object unique identifiers.
        """
        rows = self._execute_read(
            "SELECT obj_uid FROM storage WHERE table_name = ? AND field = ?",
            (table, field_name),
        )
        return [row[0] for row in rows]

    def _flush_set_field_batch(self) -> None:
        """Store field values from batch."""
        if not self._set_field_batch:
            return

        operations = []
        done: list[tuple[str, UID, str]] = []
        blob_futures: list[Future[None]] = []
        new_obj_uids: set[tuple[str, str]] = set()
        new_fields: set[tuple[str, str]] = set()

        try:
            for (tbl, obj_uid, field_name), value in self._set_field_batch.items():
                data, dtype = self._serialize(value)
                value_meta = self._encode_value_meta(value)
                file_path = None

                if dtype == "ndarray" and len(data) > self._array_threshold:
                    file_path = self._store_large_array(
                        tbl, field_name, obj_uid, data, blob_futures
                    )
                    data = b""

                operations.append(
                    (tbl, field_name, obj_uid, dtype, data, file_path, value_meta)
                )
                done.append((tbl, obj_uid, field_name))
                new_obj_uids.add((tbl, obj_uid))
                new_fields.add((tbl, field_name))

            self._wait_for_blob_writes(blob_futures)
            self._execute_write(
                "INSERT OR REPLACE INTO storage "
                "(table_name, field, obj_uid, value_type, value_data, "
                "file_path, value_meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                operations,
            )
            if new_obj_uids:
                self._execute_write(
                    "INSERT OR IGNORE INTO obj_registry "
                    "(table_name, obj_uid) VALUES (?, ?)",
                    tuple(new_obj_uids),
                )
            if new_fields:
                self._execute_write(
                    "INSERT OR IGNORE INTO field_registry "
                    "(table_name, field) VALUES (?, ?)",
                    tuple(new_fields),
                )
        finally:
            for key in done:
                del self._set_field_batch[key]
                size = self._set_field_sizes.pop(key, 0)
                if size:
                    self._pending_set_bytes = max(0, self._pending_set_bytes - size)
            if not self._set_field_sizes:
                self._pending_set_bytes = 0

    def _flush_erase_field_batch(self) -> None:
        """Remove fields from batch."""
        if not self._erase_field_batch:
            return

        try:
            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                tmp_table = "tmp_erase_field"
                cur.execute(
                    f"""
                    CREATE TEMP TABLE IF NOT EXISTS {tmp_table} (
                        table_name TEXT NOT NULL,
                        field TEXT NOT NULL,
                        obj_uid TEXT NOT NULL
                    )
                    """
                )

                cur.executemany(
                    f"INSERT INTO {tmp_table} "
                    "(table_name, field, obj_uid) VALUES (?, ?, ?)",
                    tuple(
                        (tbl, field_name, obj_uid)
                        for tbl, obj_uid, field_name in self._erase_field_batch
                    ),
                )

                rows = cur.execute(
                    f"""
                    SELECT file_path, value_type
                    FROM storage
                    WHERE (table_name, field, obj_uid) IN (
                        SELECT table_name, field, obj_uid FROM {tmp_table}
                    )
                    """
                ).fetchall()

                cur.execute(
                    f"""
                    DELETE FROM storage
                    WHERE (table_name, field, obj_uid) IN (
                        SELECT table_name, field, obj_uid FROM {tmp_table}
                    )
                    """
                )

                cur.execute(
                    f"""
                    DELETE FROM obj_registry
                    WHERE (table_name, obj_uid) IN (
                        SELECT table_name, obj_uid FROM {tmp_table}
                    )
                      AND (table_name, obj_uid) NOT IN (
                        SELECT DISTINCT table_name, obj_uid FROM storage
                      )
                    """
                )

                cur.execute(
                    f"""
                    DELETE FROM field_registry
                    WHERE (table_name, field) IN (
                        SELECT table_name, field FROM {tmp_table}
                    )
                      AND (table_name, field) NOT IN (
                        SELECT DISTINCT table_name, field FROM storage
                      )
                    """
                )

                cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")

            self._delete_array_files(rows)
        finally:
            self._erase_field_batch.clear()

    def _flush_erase_obj_batch(self) -> None:
        """Remove objects from batch."""
        if not self._erase_obj_batch:
            return

        try:
            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                tmp_table = "tmp_erase_obj"
                cur.execute(
                    f"""
                    CREATE TEMP TABLE IF NOT EXISTS {tmp_table} (
                        table_name TEXT NOT NULL,
                        obj_uid TEXT NOT NULL
                    )
                    """
                )

                cur.executemany(
                    f"INSERT INTO {tmp_table} (table_name, obj_uid) VALUES (?, ?)",
                    tuple(self._erase_obj_batch),
                )

                rows = cur.execute(
                    f"""
                    SELECT file_path, value_type
                    FROM storage
                    WHERE (table_name, obj_uid) IN (
                        SELECT table_name, obj_uid FROM {tmp_table}
                    )
                    """
                ).fetchall()

                cur.execute(
                    f"""
                    DELETE FROM storage
                    WHERE (table_name, obj_uid) IN (
                        SELECT table_name, obj_uid FROM {tmp_table}
                    )
                    """
                )

                cur.execute(
                    f"""
                    DELETE FROM obj_registry
                    WHERE (table_name, obj_uid) IN (
                        SELECT table_name, obj_uid FROM {tmp_table}
                    )
                    """
                )

                cur.execute(
                    """
                    DELETE FROM field_registry
                    WHERE (table_name, field) NOT IN (
                        SELECT DISTINCT table_name, field FROM storage
                    )
                    """
                )

                cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")

            self._delete_array_files(rows)
        finally:
            self._erase_obj_batch.clear()

    def _flush_erase_field_for_all_batch(self) -> None:
        """Remove field from all objects in batch."""
        if not self._erase_field_for_all_batch:
            return

        try:
            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                tmp_table = "tmp_erase_field_for_all"
                cur.execute(
                    f"""
                    CREATE TEMP TABLE IF NOT EXISTS {tmp_table} (
                        table_name TEXT NOT NULL,
                        field TEXT NOT NULL
                    )
                    """
                )

                cur.executemany(
                    f"INSERT INTO {tmp_table} (table_name, field) VALUES (?, ?)",
                    tuple(self._erase_field_for_all_batch),
                )

                rows = cur.execute(
                    f"""
                    SELECT file_path, value_type
                    FROM storage
                    WHERE (table_name, field) IN (
                        SELECT table_name, field FROM {tmp_table}
                    )
                    """
                ).fetchall()

                cur.execute(
                    f"""
                    DELETE FROM storage
                    WHERE (table_name, field) IN (
                        SELECT table_name, field FROM {tmp_table}
                    )
                    """
                )

                cur.execute(
                    f"""
                    DELETE FROM field_registry
                    WHERE (table_name, field) IN (
                        SELECT table_name, field FROM {tmp_table}
                    )
                    """
                )

                cur.execute(
                    """
                    DELETE FROM obj_registry
                    WHERE (table_name, obj_uid) NOT IN (
                        SELECT DISTINCT table_name, obj_uid FROM storage
                    )
                    """
                )

                cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")

            self._delete_array_files(rows)
        finally:
            self._erase_field_for_all_batch.clear()

    def _clear(self, *, table: str | None = None) -> None:
        """Clear data from storage.

        Args:
            table: If provided, clear only this table. If None, clear all data.
        """
        if table is None:
            rows = self._execute_read("SELECT file_path, value_type FROM storage")
            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                cur.execute("DELETE FROM storage")
                cur.execute("DELETE FROM obj_registry")
                cur.execute("DELETE FROM field_registry")
        else:
            rows = self._execute_read(
                "SELECT file_path, value_type FROM storage WHERE table_name = ?",
                (table,),
            )
            with self._transaction(), closing(self._write_conn.cursor()) as cur:
                cur.execute("DELETE FROM storage WHERE table_name = ?", (table,))
                cur.execute("DELETE FROM obj_registry WHERE table_name = ?", (table,))
                cur.execute("DELETE FROM field_registry WHERE table_name = ?", (table,))

        self._delete_array_files(rows)

    def connect(self) -> None:
        """Connect to database and initialize schema."""
        self._write_conn = sqlite3.connect(
            self._uri,
            check_same_thread=False,
            uri=True,
            timeout=self._timeout,
            isolation_level=None,
        )
        self._apply_pragmas()
        if not self._readonly:
            self._init_schema()

        if self._read_pool is not None:
            self._read_pool.connect()

    def close(self) -> None:
        """Close all connections."""
        if self._read_pool is not None:
            self._read_pool.close_all()
        if self._write_conn:
            self._write_conn.close()
            self._write_conn = None
        if self._blob_executor is not None:
            self._blob_executor.shutdown(wait=True)
            self._blob_executor = None

    def __enter__(self) -> Self:
        """Enter batch write context manager with transaction."""
        if self._context_depth == 0:
            self._write_lock.acquire()
            self._write_conn.execute("BEGIN IMMEDIATE")

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit batch write context manager."""
        try:
            if self._context_depth == 1:
                if exc_type is None:
                    try:
                        self._perform_batch_operations()
                        self._write_conn.execute("COMMIT")
                    except Exception:
                        self._write_conn.execute("ROLLBACK")
                        raise
                else:
                    self._write_conn.execute("ROLLBACK")
        finally:
            self._context_depth = max(0, self._context_depth - 1)
            if self._context_depth == 0:
                self._write_lock.release()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with suppress(Exception):
            self.close()
