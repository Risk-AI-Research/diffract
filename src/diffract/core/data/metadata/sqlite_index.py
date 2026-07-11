"""SQLite-based metadata index implementation.

This module provides a SQLite implementation of the metadata index
with optimized querying and large-filter handling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import closing, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import types

logger = logging.getLogger(__name__)

# Mapping from Python types to SQLite types
_TYPE_MAP: dict[type, str] = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    bool: "INTEGER",
}

_DEFAULT_MAX_SQL_VARIABLES = 999
_SQL_VARIABLE_SAFETY_MARGIN = 32
_DEFAULT_IN_TEMP_TABLE_THRESHOLD = 512


class SQLiteMetadataIndex:
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
        if ":memory:" not in path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._close_in_destructor_only = False
        else:
            self._close_in_destructor_only = True
        self._in_destructor = False

    def connect(self) -> None:
        """Connect to database and apply pragmas."""
        if self._conn is not None:
            return

        self._conn = sqlite3.connect(
            self._path if ":memory:" not in self._path else ":memory:",
            check_same_thread=False,
            timeout=self._timeout,
            isolation_level=None,
        )

        with closing(self._conn.cursor()) as cur:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(f"PRAGMA cache_size=-{self._cache_size_kb}")
            cur.execute("PRAGMA temp_store=MEMORY")
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
        if self._context_depth == 0:
            self._ensure_connected()
            self._conn.execute("BEGIN IMMEDIATE")
        self._context_depth += 1
        return self

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
                if exc_type is None:
                    self._conn.execute("COMMIT")
                else:
                    self._conn.execute("ROLLBACK")
        finally:
            self._lock.release()

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
        index_name = f"{table_name}_value_idx"
        cur.execute(f"CREATE TEMP TABLE {table_name} (value)")
        deduplicated = self._deduplicate_values(values)
        cur.executemany(
            f"INSERT INTO {table_name} (value) VALUES (?)",
            ((value,) for value in deduplicated),
        )
        cur.execute(f"CREATE INDEX {index_name} ON {table_name}(value)")
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
            f"INSERT INTO {table_name} (seq, value) VALUES (?, ?)",
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
        self._ensure_connected()

        # Build column definitions
        col_defs = ["uid TEXT PRIMARY KEY"]
        for col_name, col_type in columns.items():
            sql_type = _TYPE_MAP.get(col_type, "TEXT")
            col_defs.append(f"{col_name} {sql_type}")

        # Add json_data column for extensibility
        col_defs.append("json_data TEXT")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)})"

        with self._lock, closing(self._conn.cursor()) as cur:
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
        self._ensure_connected()

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

        placeholders = ", ".join("?" * len(col_names))
        sql = f"INSERT INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})"

        with self._lock, closing(self._conn.cursor()) as cur:
            cur.execute(sql, col_values)

    def update(self, table: str, uid: str, **fields: Any) -> None:
        """Update an existing record."""
        self._ensure_connected()

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
        sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE uid = ?"

        with self._lock, closing(self._conn.cursor()) as cur:
            cur.execute(sql, values)

    def upsert(self, table: str, uid: str, **fields: Any) -> None:
        """Insert or update a record."""
        self._ensure_connected()

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

        placeholders = ", ".join("?" * len(col_names))
        sql = (
            f"INSERT OR REPLACE INTO {table} "
            f"({', '.join(col_names)}) VALUES ({placeholders})"
        )

        with self._lock, closing(self._conn.cursor()) as cur:
            cur.execute(sql, col_values)

    def get(self, table: str, uid: str) -> dict[str, Any] | None:
        """Get a single record by uid."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(f"SELECT * FROM {table} WHERE uid = ?", (uid,))
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
                sql = f"SELECT * FROM {table} WHERE uid IN ({placeholders})"
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
                    f"SELECT f.seq, t.* FROM {tmp_table} AS f "
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
        self._ensure_connected()

        conditions: list[str] = []
        params: list[Any] = []
        temp_where_in: dict[str, list[Any]] = {}
        table_alias = "base"

        if where:
            for col, val in where.items():
                conditions.append(f"{table_alias}.{col} = ?")
                params.append(self._serialize_value(val))

        if where_like:
            for col, pattern in where_like.items():
                conditions.append(f"{table_alias}.{col} LIKE ?")
                params.append(pattern)

        if where_in:
            for col, raw_values in where_in.items():
                if len(raw_values) == 0:
                    # SQL semantics: x IN () should always produce no rows.
                    return []

                values = [self._serialize_value(v) for v in raw_values]
                if self._can_inline_values(len(values), existing_params=len(params)):
                    placeholders = ", ".join("?" * len(values))
                    conditions.append(f"{table_alias}.{col} IN ({placeholders})")
                    params.extend(values)
                else:
                    temp_where_in[col] = values

        with self._lock, closing(self._conn.cursor()) as cur:
            temp_tables: dict[str, str] = {}
            try:
                for col, values in temp_where_in.items():
                    temp_tables[col] = self._create_temp_values_table(
                        cur, values, prefix=f"{table}_{col}_in"
                    )

                from_clause = f"{table} AS {table_alias}"
                uid_temp = temp_tables.get("uid")
                if uid_temp is not None:
                    from_clause += (
                        f" JOIN {uid_temp} AS uid_filter "
                        f"ON uid_filter.value = {table_alias}.uid"
                    )

                for col, tmp_table in temp_tables.items():
                    if col == "uid":
                        continue
                    conditions.append(
                        f"{table_alias}.{col} IN (SELECT value FROM {tmp_table})"
                    )

                sql = f"SELECT {table_alias}.uid FROM {from_clause}"
                if conditions:
                    sql += f" WHERE {' AND '.join(conditions)}"
                if order_by:
                    sql += f" ORDER BY {', '.join(order_by)}"
                if limit is not None:
                    sql += f" LIMIT {limit}"

                cur.execute(sql, params)
                return [row[0] for row in cur.fetchall()]
            finally:
                for tmp_table in temp_tables.values():
                    self._drop_temp_table(cur, tmp_table)

    def delete(self, table: str, uid: str) -> None:
        """Delete a record by uid."""
        self._ensure_connected()

        with self._lock, closing(self._conn.cursor()) as cur:
            cur.execute(f"DELETE FROM {table} WHERE uid = ?", (uid,))

    def delete_batch(self, table: str, uids: list[str]) -> None:
        """Delete multiple records by uids."""
        if not uids:
            return

        self._ensure_connected()

        with self._lock, closing(self._conn.cursor()) as cur:
            if self._can_inline_values(len(uids)):
                placeholders = ", ".join("?" * len(uids))
                sql = f"DELETE FROM {table} WHERE uid IN ({placeholders})"
                cur.execute(sql, uids)
                return

            serialized = [self._serialize_value(uid) for uid in uids]
            tmp_table = self._create_temp_values_table(
                cur, serialized, prefix=f"{table}_delete_batch"
            )
            try:
                cur.execute(
                    f"DELETE FROM {table} WHERE uid IN (SELECT value FROM {tmp_table})"
                )
            finally:
                self._drop_temp_table(cur, tmp_table)

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """Count records in table."""
        self._ensure_connected()

        conditions: list[str] = []
        params: list[Any] = []

        if where:
            for col, val in where.items():
                conditions.append(f"{col} = ?")
                params.append(self._serialize_value(val))

        sql = f"SELECT COUNT(*) FROM {table}"
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"

        with closing(self._conn.cursor()) as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def distinct(self, table: str, column: str) -> list[Any]:
        """Get distinct values for a column."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(f"SELECT DISTINCT {column} FROM {table}")
            return [row[0] for row in cur.fetchall()]

    def list_uids(self, table: str) -> list[str]:
        """List all UIDs in a table."""
        self._ensure_connected()

        with closing(self._conn.cursor()) as cur:
            cur.execute(f"SELECT uid FROM {table}")
            return [row[0] for row in cur.fetchall()]

    def clear(self, table: str | None = None) -> None:
        """Clear data from index."""
        self._ensure_connected()

        with self._lock, closing(self._conn.cursor()) as cur:
            if table is None:
                # Get all tables and delete from each
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cur.fetchall()]
                for t in tables:
                    if not t.startswith("sqlite_"):
                        cur.execute(f"DELETE FROM {t}")
            else:
                cur.execute(f"DELETE FROM {table}")

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
