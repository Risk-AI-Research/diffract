"""SQLite-based metadata index implementation.

This module provides a SQLite implementation of the metadata index
with optimized querying and large-filter handling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import ExitStack, closing, contextmanager, nullcontext, suppress
from functools import lru_cache
from itertools import batched, groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .errors import constraint_error
from .interface import IMetadataIndex
from .migrations import SCHEMA_META_TABLE, ensure_schema_version
from .retry import RETRY_ATTEMPTS, RETRY_JITTER_MS, with_write_retry
from .runs import OwnerVerification, ValueTrace

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from typing import Literal

    from .interface import JoinSpec
    from .runs import Generation

logger = logging.getLogger(__name__)

# Mapping from Python types to SQLite types
_TYPE_MAP: dict[type, str] = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    bool: "INTEGER",
}

_MULTIROW_INSERT_MIN_ROWS = 16
_DEFAULT_MAX_SQL_VARIABLES = 999
_SQL_VARIABLE_SAFETY_MARGIN = 32
_DEFAULT_IN_TEMP_TABLE_THRESHOLD = 512

# SQLite sentinel selecting an in-memory database. Not a filesystem path, so
# configuration path resolution carries it verbatim.
IN_MEMORY_DATABASE = ":memory:"


def _identifier(name: str) -> str:
    if not name.isascii() or not name.isidentifier():
        raise ValueError(f"Invalid metadata identifier: {name!r}")
    return f'"{name}"'


def _column(name: str, aliases: tuple[str, ...]) -> str:
    alias, dot, column = name.partition(".")
    if not dot:
        if len(aliases) > 1:
            raise ValueError(f"Joined columns must be qualified: {name!r}")
        alias, column = "base", name
    if alias not in aliases:
        raise ValueError(f"Unknown table alias: {alias!r}")
    return f"{_identifier(alias)}.{_identifier(column)}"


def _sort_column(name: str, aliases: tuple[str, ...]) -> str:
    column, space, direction = name.rpartition(" ")
    if space and direction.upper() in {"ASC", "DESC"}:
        return f"{_column(column.rstrip(), aliases)} {direction.upper()}"
    return _column(name, aliases)


@lru_cache(maxsize=128)
def _point_select(table: str, columns: tuple[str, ...], key: tuple[str, ...]) -> str:
    projection = ", ".join(map(_identifier, columns))
    predicate = " AND ".join(f"{_identifier(column)} = ?" for column in key)
    return f"SELECT {projection} FROM {_identifier(table)} WHERE {predicate} LIMIT 1"  # nosec B608 - validated identifiers


@lru_cache(maxsize=128)
def _insert_record(table: str, columns: tuple[str, ...]) -> str:
    names = ", ".join(map(_identifier, columns))
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {_identifier(table)} ({names}) VALUES ({placeholders})"  # nosec B608 - validated identifiers


@lru_cache(maxsize=128)
def _upsert(table: str, key: tuple[str, ...], values: tuple[str, ...]) -> str:
    columns = (*key, *values)
    target = ", ".join(map(_identifier, key))
    names = ", ".join(map(_identifier, columns))
    markers = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{_identifier(name)} = excluded.{_identifier(name)}" for name in values
    )
    return (
        f"INSERT INTO {_identifier(table)} ({names}) VALUES ({markers}) "  # nosec B608 - validated identifiers
        f"ON CONFLICT ({target}) DO UPDATE SET {updates}"
    )


_FIELD_POINT_SELECT = (
    "SELECT backend, storage_key, size_bytes, format_version, revision "
    "FROM live_field_values WHERE obj_uid=? AND field=? AND obj_kind=? LIMIT 1"
)
_GENERATED_FIELD_POINT_SELECT = (
    "SELECT backend, storage_key, size_bytes, format_version, revision, "
    "run_id, output_name, built_at_seq, changed_at_seq "
    "FROM live_field_values WHERE obj_uid=? AND field=? AND obj_kind=? LIMIT 1"
)
_FIELD_KEY_COLUMNS = ("obj_uid", "field")
_FIELD_POINTER_COLUMNS = (
    "obj_kind",
    "backend",
    "storage_key",
    "size_bytes",
    "format_version",
    "revision",
    "tombstoned_at",
)
_GENERATED_FIELD_POINTER_COLUMNS = (
    *_FIELD_POINTER_COLUMNS,
    "run_id",
    "output_name",
    "built_at_seq",
    "changed_at_seq",
)
_FIELD_UPSERT = _upsert("field_values", _FIELD_KEY_COLUMNS, _FIELD_POINTER_COLUMNS)
_GENERATED_FIELD_UPSERT = _upsert(
    "field_values", _FIELD_KEY_COLUMNS, _GENERATED_FIELD_POINTER_COLUMNS
)
_FIELD_INSERT_CONFLICT = " ON CONFLICT (obj_uid, field) DO NOTHING"
_FIELD_INSERT = (
    _insert_record("field_values", _FIELD_KEY_COLUMNS + _FIELD_POINTER_COLUMNS)
    + _FIELD_INSERT_CONFLICT
)
_GENERATED_FIELD_INSERT = (
    _insert_record(
        "field_values", _FIELD_KEY_COLUMNS + _GENERATED_FIELD_POINTER_COLUMNS
    )
    + _FIELD_INSERT_CONFLICT
)


class SQLiteMetadataIndex(IMetadataIndex):
    """SQLite-based metadata index with optimized querying.

    Provides structured metadata storage with SQL-based filtering.
    Uses a single connection with serialized writes and supports
    batch operations through context manager.
    """

    def __init__(
        self,
        path: str,
        *,
        timeout: float = 5.0,
        cache_size_kb: int = 32000,
        **kwargs: Any,
    ) -> None:
        """Initialize SQLite metadata index.

        Args:
            path: Path to SQLite database file.
            timeout: Database operation timeout in seconds.
            cache_size_kb: Cache size in kilobytes.
            **kwargs: Optional tuning args. Supports:
                - in_temp_table_threshold: switch to temp-table IN strategy
                  when value list reaches this size.
        """
        self._path = path
        self._timeout = timeout
        self._cache_size_kb = cache_size_kb
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._context_depth = 0
        self._writer_thread: int | None = None
        self._write_tables: dict[tuple[str, str], None] = {}
        self._retrying = False
        self._tables: dict[str, dict[str, type]] = {}
        self._max_sql_variables = _DEFAULT_MAX_SQL_VARIABLES
        self._temp_table_counter = 0

        raw_threshold = kwargs.get(
            "in_temp_table_threshold", _DEFAULT_IN_TEMP_TABLE_THRESHOLD
        )
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError):
            threshold = _DEFAULT_IN_TEMP_TABLE_THRESHOLD
        self._in_temp_table_threshold = max(1, threshold)

        # Ensure parent directory exists (skip for in-memory databases)
        if path == IN_MEMORY_DATABASE:
            self._close_in_destructor_only = True
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._close_in_destructor_only = False
        self._in_destructor = False

    @property
    def database_path(self) -> str:
        """Return the path used for this metadata index."""
        return self._path

    @property
    def in_write_context(self) -> bool:
        """Return whether this thread owns the index write transaction."""
        return self._writer_thread == threading.get_ident()

    def connect(self) -> None:
        """Connect to database, apply pragmas, and verify the schema version.

        Raises:
            SchemaVersionError: If the database is at a different schema
                version; the connection is not kept.
        """
        with self._lock:
            if self._conn is None:
                self._retry_open(self._connect)

    def with_write_retry[T](
        self,
        attempt: Callable[[], T],
        *,
        attempts: int = RETRY_ATTEMPTS,
        jitter_ms: tuple[int, int] = RETRY_JITTER_MS,
        retry_on: tuple[type[Exception], ...] = (),
    ) -> T:
        """Run a transaction-owning operation with a bounded contention budget."""
        with self._lock:
            if self._retrying:
                raise RuntimeError("Write retry budgets cannot be nested")
            self._retrying = True
            try:
                return with_write_retry(
                    attempt,
                    in_transaction=lambda: bool(
                        self._context_depth
                        or (self._conn and self._conn.in_transaction)
                    ),
                    attempts=attempts,
                    jitter_ms=jitter_ms,
                    retry_on=retry_on,
                )
            finally:
                self._retrying = False

    def _retry_open[T](self, operation: Callable[[], T]) -> T:
        return operation() if self._retrying else self.with_write_retry(operation)

    def _connect(self) -> None:
        conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            with closing(conn.cursor()) as cur:
                # Connection-scoped pragmas; none mutate the database file.
                cur.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
                cur.execute("PRAGMA foreign_keys=OFF")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute(f"PRAGMA cache_size=-{self._cache_size_kb}")
                cur.execute("PRAGMA temp_store=MEMORY")
            # Verify the version before journal_mode=WAL: switching journal
            # mode persists to the file header, so a refused open must reject
            # the store before writing anything to it.
            ensure_schema_version(conn, path=self._path)
            with closing(conn.cursor()) as cur:
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            with suppress(sqlite3.Error):
                conn.close()
            raise

        self._conn = conn
        self._max_sql_variables = self._detect_max_sql_variables()

    def close(self) -> None:
        """Close database connection."""
        if self._close_in_destructor_only and not self._in_destructor:
            return

        if self._conn is not None:
            with suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        """Enter batch operation context."""
        self._lock.acquire()
        try:
            if self._context_depth == 0:
                self._retry_open(self._begin_write)
                self._write_tables.clear()
                self._writer_thread = threading.get_ident()
        except BaseException:
            self._lock.release()
            raise
        self._context_depth += 1
        return self

    def _begin_write(self) -> None:
        self._ensure_connected()
        self._conn.execute("BEGIN IMMEDIATE")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit batch context and commit/rollback."""
        try:
            self._context_depth -= 1
            if self._context_depth == 0:
                if exc_type is not None:
                    self._conn.rollback()
                else:
                    try:
                        self._conn.commit()
                    except sqlite3.IntegrityError as cause:
                        try:
                            table = self._constraint_statement_table()
                        finally:
                            self._conn.rollback()
                        raise constraint_error(cause, table) from cause
                    except BaseException:
                        self._conn.rollback()
                        raise
        finally:
            if self._context_depth == 0:
                self._writer_thread = None
            self._lock.release()

    def _constraint_statement_table(self) -> str:
        last_table = next(reversed(self._write_tables), ("COMMIT", ""))[0]
        with closing(self._conn.cursor()) as cur:
            for child, _, parent, _ in cur.execute("PRAGMA foreign_key_check"):
                for table, operation in reversed(self._write_tables):
                    if (table == child and operation in {"insert", "update"}) or (
                        table == parent and operation in {"update", "delete"}
                    ):
                        return table
        return last_table

    def _record_write(self, table: str, operation: str) -> None:
        key = (table, operation)
        self._write_tables.pop(key, None)
        self._write_tables[key] = None

    def _ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if self._conn is None:
            self.connect()

    def _detect_max_sql_variables(self) -> int:
        """Detect SQLite's max number of bound variables."""
        self._ensure_connected()
        try:
            with closing(self._conn.cursor()) as cur:
                options = cur.execute("PRAGMA compile_options").fetchall()
        except sqlite3.Error:
            return _DEFAULT_MAX_SQL_VARIABLES

        prefix = "MAX_VARIABLE_NUMBER="
        for (option,) in options:
            if not isinstance(option, str) or not option.startswith(prefix):
                continue
            with suppress(ValueError):
                return max(_DEFAULT_MAX_SQL_VARIABLES, int(option[len(prefix) :]))

        return _DEFAULT_MAX_SQL_VARIABLES

    def _inline_param_limit(self, *, existing_params: int = 0) -> int:
        """Return safe max number of inline SQL variables."""
        return max(
            1,
            self._max_sql_variables - _SQL_VARIABLE_SAFETY_MARGIN - existing_params,
        )

    def _can_inline_values(
        self,
        value_count: int,
        *,
        existing_params: int = 0,
    ) -> bool:
        """Return True when values can be safely inlined in an IN clause."""
        if value_count <= 0:
            return True
        if value_count >= self._in_temp_table_threshold:
            return False
        return value_count <= self._inline_param_limit(existing_params=existing_params)

    def _next_temp_table_name(self, prefix: str) -> str:
        """Generate a unique temp table name."""
        safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix)
        self._temp_table_counter += 1
        return f"tmp_{safe_prefix}_{threading.get_ident()}_{self._temp_table_counter}"

    def _deduplicate_values(self, values: list[Any]) -> list[Any]:
        """Best-effort deduplication preserving order."""
        try:
            return list(dict.fromkeys(values))
        except TypeError:
            return values

    def _create_temp_values_table(
        self,
        cur: sqlite3.Cursor,
        values: list[Any],
        *,
        prefix: str,
    ) -> str:
        """Create and populate a temp table with a single value column."""
        table_name = self._next_temp_table_name(prefix)
        text_keys = all(isinstance(value, str) for value in values)
        ddl = "(value TEXT PRIMARY KEY) WITHOUT ROWID" if text_keys else "(value)"
        cur.execute(f"CREATE TEMP TABLE {table_name} {ddl}")
        try:
            deduplicated = self._deduplicate_values(values)
            limit = cur.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            prefix = f"INSERT INTO {table_name} (value) VALUES "  # nosec B608 - generated table name
            for batch in batched(deduplicated, min(128, limit)):
                cur.execute(prefix + ", ".join(["(?)"] * len(batch)), batch)
            if not text_keys:
                cur.execute(
                    f"CREATE INDEX {table_name}_value_idx ON {table_name}(value)"
                )
        except BaseException:
            self._drop_temp_table(cur, table_name)
            raise
        return table_name

    def _create_temp_sequence_table(
        self,
        cur: sqlite3.Cursor,
        values: list[Any],
        *,
        prefix: str,
    ) -> str:
        """Create and populate a temp table preserving input order."""
        table_name = self._next_temp_table_name(prefix)
        index_name = f"{table_name}_value_idx"
        cur.execute(f"CREATE TEMP TABLE {table_name} (seq INTEGER PRIMARY KEY, value)")
        cur.executemany(
            f"INSERT INTO {table_name} "  # nosec B608 - name from _next_temp_table_name
            "(seq, value) VALUES (?, ?)",
            ((i, value) for i, value in enumerate(values)),
        )
        cur.execute(f"CREATE INDEX {index_name} ON {table_name}(value)")
        return table_name

    def _drop_temp_table(self, cur: sqlite3.Cursor, table_name: str) -> None:
        """Drop a temp table, suppressing cleanup errors."""
        with suppress(sqlite3.Error):
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    def _table_exists(self, table: str) -> bool:
        """Check if table exists."""
        self._ensure_connected()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            return cur.fetchone() is not None

    def define_table(
        self,
        table: str,
        columns: dict[str, type],
        indexes: list[str] | None = None,
    ) -> None:
        """Define a table schema with typed columns."""
        # Build column definitions
        col_defs = ["uid TEXT PRIMARY KEY"]
        for col_name, col_type in columns.items():
            sql_type = _TYPE_MAP.get(col_type, "TEXT")
            col_defs.append(f"{col_name} {sql_type}")

        # Add json_data column for extensibility
        col_defs.append("json_data TEXT")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)})"

        with self._row_write(table, "define", multi_statement=True) as cur:
            cur.execute(create_sql)

            # Create indexes
            if indexes:
                for col in indexes:
                    idx_name = f"idx_{table}_{col}_uid"
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col}, uid)"
                    )

        self._tables[table] = columns

    def insert(self, table: str, uid: str, **fields: Any) -> None:
        """Insert a new record."""
        columns = self._tables.get(table, {})
        col_names = ["uid"]
        col_values: list[Any] = [uid]
        json_extra: dict[str, Any] = {}

        for key, value in fields.items():
            if key in columns:
                col_names.append(key)
                col_values.append(self._serialize_value(value))
            else:
                json_extra[key] = value

        col_names.append("json_data")
        col_values.append(json.dumps(json_extra) if json_extra else None)

        sql = _insert_record(table, tuple(col_names))
        self._execute_row_write(table, "insert", sql, col_values)

    def update(self, table: str, uid: str, **fields: Any) -> None:
        """Update an existing record."""
        columns = self._tables.get(table, {})
        set_parts: list[str] = []
        values: list[Any] = []

        for key, value in fields.items():
            if key in columns:
                set_parts.append(f"{key} = ?")
                values.append(self._serialize_value(value))

        if not set_parts:
            return

        values.append(uid)
        sql = (
            f"UPDATE {table} "  # nosec B608 - table constant; schema columns
            f"SET {', '.join(set_parts)} WHERE uid = ?"
        )

        with self._row_write(table, "update") as cur:
            cur.execute(sql, values)

    def upsert(self, table: str, uid: str, **fields: Any) -> None:
        """Insert or update a record."""
        columns = self._tables.get(table, {})
        col_names = ["uid"]
        col_values: list[Any] = [uid]
        json_extra: dict[str, Any] = {}

        for key, value in fields.items():
            if key in columns:
                col_names.append(key)
                col_values.append(self._serialize_value(value))
            else:
                json_extra[key] = value

        col_names.append("json_data")
        col_values.append(json.dumps(json_extra) if json_extra else None)

        assignments = ", ".join(
            f"{_identifier(col)} = excluded.{_identifier(col)}"
            for col in (*columns, "json_data")
        )
        insert = _insert_record(table, tuple(col_names))
        sql = f"{insert} ON CONFLICT(uid) DO UPDATE SET {assignments}"  # nosec B608

        with self._row_write(table, "update") as cur:
            cur.execute(sql, col_values)

    def get(self, table: str, uid: str) -> dict[str, Any] | None:
        """Get a single record by uid."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT * FROM {table} "  # nosec B608 - table is a repository constant
                "WHERE uid = ?",
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cur.description, row)

    def get_batch(self, table: str, uids: list[str]) -> list[dict[str, Any] | None]:
        """Get multiple records by uids."""
        if not uids:
            return []

        self._ensure_connected()

        with self._lock, closing(self._conn.cursor()) as cur:
            if self._can_inline_values(len(uids)):
                placeholders = ", ".join("?" * len(uids))
                sql = (
                    f"SELECT * FROM {table} "  # nosec B608 - table is a repo constant
                    f"WHERE uid IN ({placeholders})"
                )
                cur.execute(sql, uids)
                rows = cur.fetchall()

                # Build uid -> row mapping
                uid_to_row: dict[str, dict[str, Any]] = {}
                for row in rows:
                    row_dict = self._row_to_dict(cur.description, row)
                    uid_to_row[row_dict["uid"]] = row_dict

                # Return in same order as input uids
                return [uid_to_row.get(uid) for uid in uids]

            tmp_table = self._create_temp_sequence_table(
                cur, uids, prefix=f"{table}_get_batch"
            )
            try:
                cur.execute(
                    f"SELECT f.seq, t.* "  # nosec B608 - repo constant + sanitized temp
                    f"FROM {tmp_table} AS f "
                    f"LEFT JOIN {table} AS t ON t.uid = f.value "
                    "ORDER BY f.seq"
                )
                rows = cur.fetchall()
                description = cur.description

                results: list[dict[str, Any] | None] = [None] * len(uids)
                for row in rows:
                    seq = int(row[0])
                    if row[1] is None:
                        continue
                    results[seq] = self._row_to_dict(description[1:], row[1:])

                return results
            finally:
                self._drop_temp_table(cur, tmp_table)

    def query(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        where_in: dict[str, list[Any]] | None = None,
        where_like: dict[str, str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Query UIDs matching criteria."""
        if where and any(value is None for value in where.values()):
            return []
        _, rows = self._select_rows(
            table,
            None,
            ("uid",),
            where,
            where_in,
            where_like,
            order_by,
            limit,
            legacy=True,
        )
        return [uid for (uid,) in rows]

    def read_live_field(
        self, uid: str, field: str, kind: str, *, include_generation: bool = False
    ) -> tuple[Any, ...] | None:
        """Read the compact live-field projection by its primary key."""
        params = (uid, field, kind)
        sql = (
            _GENERATED_FIELD_POINT_SELECT if include_generation else _FIELD_POINT_SELECT
        )
        if self.in_write_context:
            with closing(self._conn.execute(sql, params)) as cursor:
                return cursor.fetchone()
        with self._lock:
            self._ensure_connected()
            with closing(self._conn.execute(sql, params)) as cursor:
                return cursor.fetchone()

    def unverified_parameter_owners(
        self,
        owners: Sequence[str] | None,
        inputs: Sequence[str],
        outputs: Sequence[str],
        config_hash: str,
    ) -> OwnerVerification:
        """Return owners to verify and whether all outputs are absent."""
        output_fields = tuple(dict.fromkeys(outputs))
        input_fields = tuple(dict.fromkeys(inputs))
        with self._lock, ExitStack() as stack:
            self._ensure_connected()
            cursor = stack.enter_context(closing(self._conn.cursor()))
            output_clause, params = self._row_filters(
                stack,
                cursor,
                "live_field_values",
                ("base",),
                {"obj_kind": "parameter"},
                {"field": output_fields},
                extra_params=3,
            )
            present = f"SELECT 1 FROM live_field_values AS base{output_clause} LIMIT 1"  # nosec B608
            if cursor.execute(present, params).fetchone() is None:
                return OwnerVerification([], all_outputs_absent=True)
            if owners is None and self._all_parameter_groups_fresh(
                stack,
                cursor,
                output_clause,
                params,
                len(output_fields),
                input_fields,
                config_hash,
            ):
                return OwnerVerification([], all_outputs_absent=False)
            reusable = (
                "SELECT MIN(base.built_at_seq) AS built "  # nosec B608
                "FROM live_field_values AS base JOIN runs AS r ON r.run_id=base.run_id"
                + output_clause
                + " AND base.built_at_seq IS NOT NULL "
                "AND base.obj_uid=owner.uid "
                "AND r.kind='kernel' AND r.status='COMPLETE' AND r.config_hash=? "
                "HAVING COUNT(*)=? AND MIN(base.run_id)=MAX(base.run_id)"
            )
            params.extend((config_hash, len(output_fields)))
            if input_fields:
                input_clause, input_params = self._row_filters(
                    stack,
                    cursor,
                    "live_field_values",
                    ("base",),
                    {"obj_kind": "parameter"},
                    {"field": input_fields},
                    extra_params=len(params) + 1,
                )
                condition = (
                    f"(({reusable}) >= "  # nosec B608
                    "(SELECT COALESCE(MAX(base.changed_at_seq), 0) "
                    "FROM live_field_values AS base"
                    + input_clause
                    + " AND base.obj_uid=owner.uid "
                    "HAVING COUNT(*)=?)) IS NOT TRUE"
                )
                params.extend((*input_params, len(input_fields)))
            else:
                condition = "NOT EXISTS (" + reusable + ")"
            owner_clause, owner_params = self._row_filters(
                stack,
                cursor,
                "parameters",
                ("owner",),
                None,
                None if owners is None else {"owner.uid": owners},
                extra_params=len(params),
            )
            sql = (
                "SELECT owner.uid FROM parameters AS owner"  # nosec B608
                + (owner_clause + " AND " if owner_clause else " WHERE ")
                + condition
            )
            return OwnerVerification(
                [uid for (uid,) in cursor.execute(sql, (*owner_params, *params))],
                all_outputs_absent=False,
            )

    def _all_parameter_groups_fresh(
        self,
        stack: ExitStack,
        cursor: sqlite3.Cursor,
        output_clause: str,
        output_params: list[Any],
        output_count: int,
        inputs: Sequence[str],
        config_hash: str,
    ) -> bool:
        sql = (
            "WITH scope AS (SELECT COUNT(*) AS size FROM parameters), outputs AS ("  # nosec B608
            "SELECT MIN(base.built_at_seq) AS built, MIN(base.run_id) AS run_id "
            "FROM live_field_values AS base JOIN parameters AS p ON p.uid=base.obj_uid"
            + output_clause
            + " AND base.built_at_seq IS NOT NULL "
            "HAVING COUNT(*)=(SELECT size FROM scope)*? "
            "AND MIN(base.run_id)=MAX(base.run_id)"
            ") SELECT 1 FROM outputs JOIN runs AS r ON r.run_id=outputs.run_id "
            "WHERE r.kind='kernel' AND r.status='COMPLETE' AND r.config_hash=?"
        )
        params = [*output_params, output_count, config_hash]
        if inputs:
            input_clause, input_params = self._row_filters(
                stack,
                cursor,
                "live_field_values",
                ("base",),
                {"obj_kind": "parameter"},
                {"field": inputs},
                extra_params=len(params) + 1,
            )
            sql += (
                " AND outputs.built>=(SELECT COALESCE(MAX(base.changed_at_seq), 0) "  # nosec B608
                "FROM live_field_values AS base JOIN parameters AS p "
                "ON p.uid=base.obj_uid"
                + input_clause
                + " HAVING COUNT(*)=(SELECT size FROM scope)*?)"
            )
            params.extend((*input_params, len(inputs)))
        return cursor.execute(sql, params).fetchone() is not None

    def read_value_traces(
        self,
        selections: Mapping[str, tuple[Sequence[str], Sequence[str]]],
        *,
        include_config: bool = False,
    ) -> dict[tuple[str, str, str], ValueTrace]:
        """Read the live lineage of a selection in one metadata snapshot."""
        with self._lock, ExitStack() as stack:
            self._ensure_connected()
            cursor = stack.enter_context(closing(self._conn.cursor()))
            statements: list[str] = []
            params: list[Any] = []
            config = ", r.config_json" if include_config else ""
            projection = (
                "SELECT base.obj_kind, base.obj_uid, base.field, base.revision, "  # nosec B608
                "base.format_version, "
                "base.run_id, r.kind, r.status, r.config_hash, "
                "base.built_at_seq, base.changed_at_seq"
                + config
                + " FROM live_field_values AS base "
                "LEFT JOIN runs AS r ON r.run_id = base.run_id"
            )
            for kind, (owners, fields) in selections.items():
                if not owners or not fields:
                    continue
                clause, values = self._row_filters(
                    stack,
                    cursor,
                    "live_field_values",
                    ("base",),
                    {"obj_kind": kind},
                    {"obj_uid": owners, "field": fields},
                    extra_params=len(params),
                )
                statements.append(projection + clause)
                params.extend(values)
            if not statements:
                return {}
            return {
                (kind, uid, field): ValueTrace(*trace)
                for kind, uid, field, *trace in cursor.execute(
                    " UNION ALL ".join(statements), params
                )
            }

    def write_field_pointer(
        self,
        uid: str,
        field: str,
        kind: str,
        backend: str,
        storage_key: str,
        size_bytes: int | None,
        format_version: int,
        revision: str,
        generation: Generation | None = None,
        *,
        insert_only: bool = False,
    ) -> bool:
        """Write a field pointer, reporting whether the row changed."""
        values: tuple[Any, ...] = (
            uid,
            field,
            kind,
            backend,
            storage_key,
            size_bytes,
            format_version,
            revision,
            None,
        )
        sql = _FIELD_INSERT if insert_only else _FIELD_UPSERT
        if generation is not None:
            sql = _GENERATED_FIELD_INSERT if insert_only else _GENERATED_FIELD_UPSERT
            values = (
                *values,
                generation.run_id,
                generation.output_name,
                generation.built_at_seq,
                generation.changed_at_seq,
            )
        operation = "insert" if insert_only else "upsert"
        return self._execute_row_write("field_values", operation, sql, values) == 1

    def find_row(
        self, table: str, *, columns: Sequence[str], where: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Project one row by an equality key."""
        if not columns or not where or any(value is None for value in where.values()):
            raise ValueError("Select columns and specify a non-null equality key")
        sql = _point_select(table, tuple(columns), tuple(where))
        with self._lock:
            self._ensure_connected()
            with closing(self._conn.execute(sql, tuple(where.values()))) as cur:
                row = cur.fetchone()
        return None if row is None else dict(zip(columns, row, strict=True))

    def upsert_row(
        self, table: str, *, key: Mapping[str, Any], values: Mapping[str, Any]
    ) -> None:
        """Insert a row or replace its projected values under a unique key."""
        if not key or not values or key.keys() & values.keys():
            raise ValueError("Specify a unique key and separate columns to update")
        sql = _upsert(table, tuple(key), tuple(values))
        with self._row_write(table, "upsert") as cur:
            cur.execute(sql, (*key.values(), *values.values()))

    def query_rows(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        where_in: Mapping[str, Sequence[Any]] | None = None,
        order_by: Sequence[str] | None = None,
        limit: int | None = None,
        distinct: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the selected physical columns, including JSON text as stored."""
        names, rows = self._select_rows(
            table,
            None,
            columns,
            where,
            where_in,
            None,
            order_by,
            limit,
            distinct=distinct,
        )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def catalog_owner_gaps(self) -> list[dict[str, str]]:
        """Find live field pointers whose parameter or aggregate is missing."""
        gaps = []
        with self._lock:
            self._ensure_connected()
            for kind, table in (
                ("parameter", "parameters"),
                ("aggregate", "aggregates"),
            ):
                query = (
                    "SELECT f.obj_uid, f.field FROM live_field_values AS f "  # nosec B608
                    "WHERE f.obj_kind = ? AND NOT EXISTS ("
                    f"SELECT 1 FROM {table} AS owner WHERE owner.uid = f.obj_uid) "
                    "ORDER BY f.obj_uid, f.field"
                )
                with closing(self._conn.execute(query, (kind,))) as cur:
                    gaps.extend(
                        {"obj_kind": kind, "obj_uid": uid, "field": field}
                        for uid, field in cur
                    )
        return gaps

    def uids_missing_related_values(
        self,
        table: str,
        join: JoinSpec,
        *,
        column: str,
        required: Sequence[str],
        where: Mapping[str, Any],
    ) -> list[str]:
        """Select owners missing any required distinct value in related rows."""
        required = tuple(dict.fromkeys(required))
        if not required:
            return []
        with self._lock, ExitStack() as stack:
            self._ensure_connected()
            cur = stack.enter_context(closing(self._conn.cursor()))
            filters, args = self._row_filters(
                stack,
                cur,
                join.table,
                ("base",),
                where,
                {column: required},
                extra_params=1,
            )
            match = (
                f" FROM {_identifier(join.table)} AS base{filters} "
                f"AND base.{_identifier(join.right_column)} = "
                f"owner.{_identifier(join.left_column)}"
            )
            query = f"SELECT owner.uid FROM {_identifier(table)} AS owner WHERE "  # nosec B608 - validated identifiers
            if len(required) == 1:
                query += f"NOT EXISTS (SELECT 1{match})"
            else:
                query += (
                    f"(SELECT COUNT(DISTINCT base.{_identifier(column)}){match}) < ?"
                )
                args.append(len(required))
            return [uid for (uid,) in cur.execute(query, args)]

    def query_joined_rows(
        self,
        table: str,
        join: JoinSpec,
        *,
        columns: Sequence[str],
        where: Mapping[str, Any] | None = None,
        where_in: Mapping[str, Sequence[Any]] | None = None,
        order_by: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read an inner join keyed by the qualified projection names."""
        names, rows = self._select_rows(
            table, join, columns, where, where_in, None, order_by, limit
        )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def _select_rows(
        self,
        table: str,
        join: JoinSpec | None,
        columns: Sequence[str] | None,
        where: Mapping[str, Any] | None,
        where_in: Mapping[str, Sequence[Any]] | None,
        where_like: Mapping[str, str] | None,
        order_by: Sequence[str] | None,
        limit: int | None,
        *,
        distinct: bool = False,
        legacy: bool = False,
    ) -> tuple[list[str], list[tuple]]:
        aliases = ("base",) if join is None else ("base", join.alias)
        source = f"{_identifier(table)} AS base"
        if join is not None:
            if join.alias.casefold() == "base":
                raise ValueError("The joined table must have its own alias")
            alias = _identifier(join.alias)
            source += (
                f" JOIN {_identifier(join.table)} AS {alias}"
                f" ON base.{_identifier(join.left_column)}"
                f" = {alias}.{_identifier(join.right_column)}"
            )
        if columns is not None and not columns:
            raise ValueError("Select at least one column")
        projection = (
            "base.*"
            if columns is None
            else ", ".join(f'{_column(name, aliases)} AS "{name}"' for name in columns)
        )
        with self._lock, ExitStack() as stack:
            self._ensure_connected()
            cur = stack.enter_context(closing(self._conn.cursor()))
            filters, params = self._row_filters(
                stack,
                cur,
                table,
                aliases,
                where,
                where_in,
                where_like,
                extra_params=int(limit is not None),
                legacy=legacy,
            )
            modifier = "DISTINCT " if distinct else ""
            sql = f"SELECT {modifier}{projection} FROM {source}"  # nosec B608 - validated identifiers
            sql += filters
            if order_by:
                sql += " ORDER BY " + ", ".join(
                    _sort_column(name, aliases) for name in order_by
                )
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            cur.execute(sql, params)
            return [desc[0] for desc in cur.description], cur.fetchall()

    def _row_filters(
        self,
        stack: ExitStack,
        cur: sqlite3.Cursor,
        table: str,
        aliases: tuple[str, ...],
        where: Mapping[str, Any] | None,
        where_in: Mapping[str, Sequence[Any]] | None,
        where_like: Mapping[str, str] | None = None,
        *,
        extra_params: int = 0,
        legacy: bool = False,
    ) -> tuple[str, list[Any]]:
        """Build predicates from validated identifiers and separately bound values."""
        if not legacy and (
            any(value is None for value in (where or {}).values())
            or any(
                value is None
                for values in (where_in or {}).values()
                for value in values
            )
        ):
            raise ValueError("None is not supported in metadata predicates")
        conditions = []
        params: list[Any] = []
        for name, value in (where or {}).items():
            conditions.append(f"{_column(name, aliases)} = ?")
            params.append(value)
        for name, pattern in (where_like or {}).items():
            conditions.append(f"{_column(name, aliases)} LIKE ?")
            params.append(pattern)
        for name, values in (where_in or {}).items():
            column = _column(name, aliases)
            if len(values) == 0:
                return " WHERE 0", []
            if self._can_inline_values(
                len(values), existing_params=extra_params + len(params)
            ):
                placeholders = ", ".join("?" for _ in values)
                conditions.append(f"{column} IN ({placeholders})")
                params.extend(values)
            else:
                tmp = self._create_temp_values_table(cur, list(values), prefix=table)
                stack.callback(self._drop_temp_table, cur, tmp)
                conditions.append(f"{column} IN (SELECT value FROM {tmp})")  # nosec B608 - generated temporary table
        return (" WHERE " + " AND ".join(conditions) if conditions else ""), params

    def _execute_row_write(
        self, table: str, operation: str, sql: str, values: Sequence[Any]
    ) -> int:
        if not self.in_write_context:
            with self._row_write(table, operation) as cursor:
                cursor.execute(sql, values)
                return cursor.rowcount
        try:
            with closing(self._conn.execute(sql, values)) as cursor:
                if cursor.rowcount:
                    self._record_write(table, operation)
                return cursor.rowcount
        except sqlite3.IntegrityError as exc:
            raise constraint_error(exc, table) from exc

    @contextmanager
    def _row_write(
        self, table: str, operation: str, *, multi_statement: bool = False
    ) -> Iterator[sqlite3.Cursor]:
        has_writer = self.in_write_context
        savepoint = multi_statement and has_writer
        try:
            with (
                nullcontext() if has_writer else self,
                closing(self._conn.cursor()) as cur,
            ):
                if savepoint:
                    cur.execute("SAVEPOINT metadata_row_write")
                try:
                    yield cur
                except BaseException:
                    if savepoint and self._conn.in_transaction:
                        cur.execute("ROLLBACK TO metadata_row_write")
                        cur.execute("RELEASE metadata_row_write")
                    raise
                if cur.rowcount != 0:
                    self._record_write(table, operation)
                if savepoint:
                    cur.execute("RELEASE metadata_row_write")
        except sqlite3.IntegrityError as cause:
            raise constraint_error(cause, table) from cause

    def insert_rows(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        on_conflict: Literal["raise", "ignore"] = "raise",
        conflict_target: Sequence[str] | None = None,
    ) -> int:
        """Insert rows atomically, optionally ignoring conflicts on one named key."""
        name = _identifier(table)
        if on_conflict not in {"raise", "ignore"}:
            raise ValueError(f"Unknown uniqueness conflict policy: {on_conflict!r}")
        if on_conflict == "ignore" and not conflict_target:
            raise ValueError("Ignoring duplicates requires a conflict_target")
        if on_conflict == "raise" and conflict_target is not None:
            raise ValueError("conflict_target requires on_conflict='ignore'")
        target = ", ".join(_identifier(column) for column in conflict_target or ())
        suffix = f" ON CONFLICT ({target}) DO NOTHING" if target else ""
        if len(rows) == 1:
            row = rows[0]
            columns = tuple(row)
            if not columns:
                raise ValueError("Each row must specify at least one column")
            sql = _insert_record(table, columns) + suffix
            values = tuple(row[column] for column in columns)
            return self._execute_row_write(table, "insert", sql, values)

        count = 0
        with self._row_write(table, "insert", multi_statement=len(rows) > 1) as cur:
            for columns, group in groupby(rows, key=tuple):
                if not columns:
                    raise ValueError("Each row must specify at least one column")
                names = ", ".join(_identifier(column) for column in columns)
                placeholders = ", ".join("?" for _ in columns)
                prefix = f"INSERT INTO {name} ({names}) VALUES "  # nosec B608 - validated identifiers
                values = (tuple(row[c] for c in columns) for row in group)
                if len(rows) < _MULTIROW_INSERT_MIN_ROWS:
                    cur.executemany(f"{prefix}({placeholders}){suffix}", values)
                    count += cur.rowcount
                    continue
                limit = cur.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
                batch_size = max(1, min(128, limit // len(columns)))
                for batch in batched(values, batch_size):
                    slots = ", ".join([f"({placeholders})"] * len(batch))
                    cur.execute(
                        prefix + slots + suffix, tuple(v for row in batch for v in row)
                    )
                    count += cur.rowcount
            if count and cur.rowcount == 0:
                self._record_write(table, "insert")
        return count

    def update_rows(
        self,
        table: str,
        *,
        where: Mapping[str, Any],
        values: Mapping[str, Any],
        where_null: Sequence[str] = (),
    ) -> int:
        """Update the projected values of matching rows and return their count."""
        count, _ = self._update_rows(table, where, {}, where_null, values, ())
        return count

    def update_rows_returning(
        self,
        table: str,
        *,
        where: Mapping[str, Any],
        where_in: Mapping[str, Sequence[Any]],
        where_null: Sequence[str],
        values: Mapping[str, Any],
        columns: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return the projected columns of the rows matched by the update."""
        if not columns:
            raise ValueError("Specify at least one return column")
        _, rows = self._update_rows(table, where, where_in, where_null, values, columns)
        return rows

    def _update_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        where_in: Mapping[str, Sequence[Any]],
        where_null: Sequence[str],
        values: Mapping[str, Any],
        columns: Sequence[str],
    ) -> tuple[int, list[dict[str, Any]]]:
        if not (where or where_in):
            raise ValueError("Specify an update predicate")
        if not values:
            raise ValueError("Specify at least one column to update")
        assignments = ", ".join(f"{_identifier(name)} = ?" for name in values)
        target = _identifier(table)
        with self._row_write(table, "update") as cur, ExitStack() as stack:
            filters, params = self._row_filters(
                stack, cur, table, ("base",), where, where_in, extra_params=len(values)
            )
            for column in where_null:
                filters += f" AND {_identifier(column)} IS NULL"
            sql = f"UPDATE {target} AS base SET {assignments}{filters}"  # nosec B608 - validated identifiers
            if columns:
                sql += " RETURNING " + ", ".join(_identifier(name) for name in columns)
            cur.execute(sql, [*values.values(), *params])
            rows = (
                [dict(zip(columns, row, strict=True)) for row in cur] if columns else []
            )
            return cur.rowcount, rows

    def delete_rows(
        self,
        table: str,
        *,
        where: Mapping[str, Any] | None = None,
        where_in: Mapping[str, Sequence[Any]] | None = None,
    ) -> int:
        """Delete matching rows, checking references at the end of the transaction."""
        if not where and not where_in:
            raise ValueError("Specify a delete predicate or use clear(table)")
        target = _identifier(table)
        with self._row_write(table, "delete") as cur, ExitStack() as stack:
            filters, params = self._row_filters(
                stack, cur, table, ("base",), where, where_in
            )
            cur.execute("PRAGMA defer_foreign_keys=ON")
            sql = f"DELETE FROM {target} AS base{filters}"  # nosec B608 - validated identifiers
            return cur.execute(sql, params).rowcount

    def delete(self, table: str, uid: str) -> None:
        """Delete one record by uid."""
        self.delete_rows(table, where={"uid": uid})

    def delete_batch(self, table: str, uids: list[str]) -> None:
        """Delete the selected records in one transaction."""
        if uids:
            self.delete_rows(table, where_in={"uid": uids})

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """Count records in table."""
        self._ensure_connected()

        conditions: list[str] = []
        params: list[Any] = []

        if where:
            for col, val in where.items():
                conditions.append(f"{col} = ?")
                params.append(self._serialize_value(val))

        sql = f"SELECT COUNT(*) FROM {table}"  # nosec B608 - table is a repo constant
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"

        with closing(self._conn.cursor()) as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def distinct(self, table: str, column: str) -> list[Any]:
        """Get distinct values for a column."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT DISTINCT {_identifier(column)} FROM {_identifier(table)}"  # nosec B608 - validated identifiers
            )
            return [row[0] for row in cur.fetchall()]

    def list_uids(self, table: str) -> list[str]:
        """List all UIDs in a table."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(f"SELECT uid FROM {table}")  # nosec B608 - repo table constant
            return [row[0] for row in cur.fetchall()]

    def clear(self, table: str | None = None) -> None:
        """Clear data from index.

        The schema metadata and UID migration audit are preserved.

        Args:
            table: Name of the data table to clear. If None, clear all data tables.
        """
        protected = {SCHEMA_META_TABLE, "uid_map"}
        if table is not None and table.lower() in protected:
            return
        with self, closing(self._conn.cursor()) as cur:
            cur.execute("PRAGMA defer_foreign_keys=ON")
            tables = (
                [table]
                if table is not None
                else [
                    row[0]
                    for row in cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    if not row[0].startswith("sqlite_") and row[0] not in protected
                ]
            )
            for name in tables:
                quoted = '"' + name.replace('"', '""') + '"'
                cur.execute(f"DELETE FROM {quoted}")  # nosec B608 - quoted identifier
                if cur.rowcount:
                    self._record_write(name, "delete")

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for storage."""
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    def _row_to_dict(
        self, description: tuple[tuple[str, ...], ...], row: tuple[Any, ...]
    ) -> dict[str, Any]:
        """Convert a database row to dictionary."""
        result: dict[str, Any] = {}
        for i, col_info in enumerate(description):
            col_name = col_info[0]
            value = row[i]

            if col_name == "json_data" and value:
                # Merge json_data into result
                result.update(json.loads(value))
            elif col_name != "json_data":
                result[col_name] = value

        return result

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._in_destructor = True
        with suppress(Exception):
            self.close()
