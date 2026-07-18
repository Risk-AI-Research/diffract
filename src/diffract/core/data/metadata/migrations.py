"""Schema versioning and migrations for the SQLite metadata index.

The index stamps ``PRAGMA user_version`` so a database's schema generation
is readable without inspecting tables. Fresh databases are created directly
at the current version and record their producer in the singleton
``schema_meta`` table. Databases written at an older schema version are
refused at open time with instructions for the explicit upgrade entry point
(:func:`upgrade_metadata_index`); nothing is migrated implicitly on open.

The mechanics follow the established embedded-SQLite discipline (Android's
``SQLiteOpenHelper``, MLflow's store versioning): migrations are numbered,
append-only steps, each applied inside a single transaction so a failing
step leaves both the schema and the version stamp untouched. A shipped step
is never rewritten; schema changes append new steps. Adding a uniqueness
constraint to an existing table goes through :func:`unique_index_step`,
because SQLite's ``ALTER TABLE`` cannot add constraints in place.

Two policies bind future steps. Derived bookkeeping (counters, trace
metadata) migrates by dropping the stored values and letting them repopulate
on first read; historical observations are never fabricated. Producer
stamps for stores that predate versioning are recorded as unknown rather
than guessed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

SCHEMA_META_TABLE = "schema_meta"

# Producer stamp for stores created before producer recording existed.
UNKNOWN_PRODUCER = "unknown (store predates schema versioning)"

_UPGRADE_ENTRY_POINT = "diffract.upgrade_metadata_index"

_BACKUP_ADVICE = (
    "Back up the database file first: each migration step commits "
    "separately, so a failure between steps leaves the store at an "
    "intermediate version."
)

_SCHEMA_META_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_meta ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "created_by TEXT NOT NULL)"
)


class SchemaVersionError(Exception):
    """Raised when a metadata index cannot be opened at its schema version.

    Attributes:
        path: Database location.
        found: Schema version recorded in the database.
        expected: Schema version this library operates on.
    """

    def __init__(self, message: str, *, path: str, found: int, expected: int) -> None:
        super().__init__(message)
        self.path = path
        self.found = found
        self.expected = expected


class MigrationError(Exception):
    """Raised when applying a migration step fails after rolling it back."""


class UniqueIndexDuplicatesError(MigrationError):
    """Raised when duplicate rows block the creation of a unique index.

    Attributes:
        table: Table the index was to be created on.
        columns: Column names forming the unique key.
        duplicates: Key tuples held by more than one row.
    """

    _SHOWN = 10

    def __init__(
        self,
        *,
        table: str,
        columns: Sequence[str],
        duplicates: Sequence[tuple[Any, ...]],
    ) -> None:
        self.table = table
        self.columns = tuple(columns)
        self.duplicates = list(duplicates)
        shown = ", ".join(repr(key) for key in self.duplicates[: self._SHOWN])
        overflow = len(self.duplicates) - self._SHOWN
        suffix = f" (and {overflow} more)" if overflow > 0 else ""
        super().__init__(
            f"cannot create a unique index on {table} "
            f"({', '.join(self.columns)}): {len(self.duplicates)} key(s) are "
            f"held by more than one row after deduplication: {shown}{suffix}. "
            "The step was rolled back; resolve the listed rows and rerun the "
            "upgrade."
        )


@dataclass(frozen=True)
class MigrationStep:
    """One numbered schema migration.

    Attributes:
        version: Schema version this step migrates the store to; the step
            runs when the store is at ``version - 1``.
        description: Short human-readable summary used in logs and errors.
        apply: Callable receiving the open connection. It runs inside the
            step's transaction and must not commit, roll back, or close it.
    """

    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]


def _create_schema_meta(conn: sqlite3.Connection, created_by: str) -> None:
    conn.execute(_SCHEMA_META_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (id, created_by) VALUES (1, ?)",
        (created_by,),
    )


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    _create_schema_meta(conn, UNKNOWN_PRODUCER)


# Shipped migration chain. Append-only: never rewrite or renumber a step.
MIGRATIONS: tuple[MigrationStep, ...] = (
    MigrationStep(
        version=1,
        description="record the store producer in the schema_meta table",
        apply=_migrate_to_v1,
    ),
)

CURRENT_SCHEMA_VERSION = MIGRATIONS[-1].version


def _producer_stamp() -> str:
    from diffract import __version__

    return f"diffract-core {__version__}"


def _read_user_version(conn: sqlite3.Connection) -> int:
    with closing(conn.cursor()) as cur:
        return int(cur.execute("PRAGMA user_version").fetchone()[0])


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA cannot bind parameters; the value is an int owned by this module.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def _rollback_if_active(conn: sqlite3.Connection) -> None:
    # SQLite auto-rolls-back on some failures (ON CONFLICT ROLLBACK, disk
    # full); an unconditional ROLLBACK would then raise and mask the error
    # being handled.
    if conn.in_transaction:
        with suppress(sqlite3.Error):
            conn.execute("ROLLBACK")


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    with closing(conn.cursor()) as cur:
        row = cur.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
    return row is not None


def _initialize_at_head(conn: sqlite3.Connection) -> None:
    _create_schema_meta(conn, _producer_stamp())
    _set_user_version(conn, CURRENT_SCHEMA_VERSION)


def _version_mismatch(*, path: str, found: int, expected: int) -> SchemaVersionError:
    if found > expected:
        message = (
            f"Metadata index at '{path}' has schema version {found}, but "
            f"this library supports up to version {expected}: the store was "
            "written by a newer release. Upgrade the library to open it."
        )
    else:
        message = (
            f"Metadata index at '{path}' has schema version {found}; the "
            f"current version is {expected}. Stores are never migrated "
            f"implicitly on open. {_BACKUP_ADVICE} Then upgrade with: "
            f"{_UPGRADE_ENTRY_POINT}('{path}')"
        )
    return SchemaVersionError(message, path=path, found=found, expected=expected)


def _validate_chain(steps: Sequence[MigrationStep]) -> None:
    for position, step in enumerate(steps, start=1):
        if step.version != position:
            raise MigrationError(
                f"migration chain is not contiguous: step at position "
                f"{position} declares version {step.version}; steps are "
                "numbered 1..N in order, and shipped steps are never "
                "renumbered"
            )


def _verify_step_precondition(
    step: MigrationStep, *, version: int, target: int, path: str
) -> None:
    """Validate the store version re-read inside a step's transaction.

    Raises:
        SchemaVersionError: If the store moved past the chain head.
        MigrationError: If the store is not at exactly the step's predecessor
            version (a concurrent writer changed it unexpectedly).
    """
    if version > target:
        raise _version_mismatch(path=path, found=version, expected=target)
    if step.version != version + 1:
        raise MigrationError(
            f"store version changed to {version} while migrating {path}; "
            f"expected {step.version - 1} before applying version {step.version}"
        )


def ensure_schema_version(conn: sqlite3.Connection, *, path: str) -> None:
    """Verify that a metadata database is at the current schema version.

    Fresh (empty) databases are initialized directly at
    ``CURRENT_SCHEMA_VERSION``: the version stamp is written and the
    producing library version is recorded in ``schema_meta``. Databases at
    any other version are refused; upgrading is an explicit, user-invoked
    action, never a side effect of opening the store.

    Args:
        conn: Open connection in autocommit mode (``isolation_level=None``).
        path: Database location, used in error messages.

    Raises:
        SchemaVersionError: If the store is at an older version (the message
            names the upgrade entry point and recommends a backup) or was
            written by a newer library release.
    """
    version = _read_user_version(conn)
    if version == CURRENT_SCHEMA_VERSION:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        version = _read_user_version(conn)
        fresh = version == 0 and not _has_user_tables(conn)
        if fresh:
            _initialize_at_head(conn)
    except BaseException:
        _rollback_if_active(conn)
        raise
    else:
        conn.execute("COMMIT")

    if fresh:
        logger.info(
            "Initialized metadata index %s at schema version %d",
            path,
            CURRENT_SCHEMA_VERSION,
        )
        return
    if version == CURRENT_SCHEMA_VERSION:
        return
    raise _version_mismatch(path=path, found=version, expected=CURRENT_SCHEMA_VERSION)


def run_migrations(
    conn: sqlite3.Connection,
    *,
    steps: Sequence[MigrationStep],
    path: str = "<metadata index>",
) -> list[int]:
    """Apply pending migration steps, each inside its own transaction.

    A failing step rolls back atomically -- both its schema changes and its
    ``user_version`` stamp -- leaving the store exactly at the pre-step
    version. Steps applied earlier in the same call remain committed. The
    version is re-read inside each step's write transaction, so a
    concurrent upgrader of the same store cannot re-apply a step or move
    the stamp backwards; steps another writer already applied are skipped.

    Args:
        conn: Open connection in autocommit mode (``isolation_level=None``).
        steps: Migration chain, numbered contiguously from 1.
        path: Database location, used in logs and error messages.

    Returns:
        Versions applied by this call, in ascending order; empty when
        already current.

    Raises:
        SchemaVersionError: If the store version exceeds the chain head.
        MigrationError: If the chain is malformed or a step fails; the
            failing step is rolled back.
    """
    _validate_chain(steps)
    target = steps[-1].version if steps else 0
    version = _read_user_version(conn)
    if version > target:
        raise _version_mismatch(path=path, found=version, expected=target)

    applied: list[int] = []
    for step in steps:
        try:
            conn.execute("BEGIN IMMEDIATE")
            version = _read_user_version(conn)
            if step.version <= version:
                conn.execute("COMMIT")
                continue
            _verify_step_precondition(step, version=version, target=target, path=path)
            step.apply(conn)
            _set_user_version(conn, step.version)
            conn.execute("COMMIT")
        except (MigrationError, SchemaVersionError):
            _rollback_if_active(conn)
            raise
        except BaseException as exc:
            _rollback_if_active(conn)
            raise MigrationError(
                f"migration to version {step.version} ({step.description}) "
                f"failed and was rolled back; the store is left at version "
                f"{version}"
            ) from exc
        version = step.version
        applied.append(step.version)
        logger.info("Migrated metadata index %s to schema version %d", path, version)
    return applied


def upgrade_metadata_index(path: str | Path) -> list[int]:
    """Upgrade a metadata index database to the current schema version.

    The explicit entry point named by the error raised when an older store
    is opened. Applies the pending shipped migration steps in order, each
    inside its own transaction. Back up the database file first: steps
    commit independently, so a failure between steps leaves the store at an
    intermediate version. An existing empty database is initialized at the
    current version, matching what opening it would produce.

    Args:
        path: Filesystem path of the metadata index database; a leading
            ``~`` expands to the user home.

    Returns:
        Schema versions applied, in ascending order; empty when the store
        was already current or freshly initialized.

    Raises:
        FileNotFoundError: If no database file exists at ``path`` (an
            in-memory store cannot be upgraded from outside its session).
        SchemaVersionError: If the store was written by a newer release.
        MigrationError: If a step fails; the failing step is rolled back.
    """
    location = str(Path(path).expanduser())
    if not Path(location).is_file():
        raise FileNotFoundError(
            f"no metadata index database at '{location}'; nothing to upgrade"
        )

    conn = sqlite3.connect(location, isolation_level=None)
    try:
        version = _read_user_version(conn)
        if version == CURRENT_SCHEMA_VERSION:
            logger.info(
                "Metadata index %s is already at schema version %d",
                location,
                version,
            )
            return []
        if version == 0 and not _has_user_tables(conn):
            conn.execute("BEGIN IMMEDIATE")
            try:
                fresh = _read_user_version(conn) == 0 and not _has_user_tables(conn)
                if fresh:
                    _initialize_at_head(conn)
            except BaseException:
                _rollback_if_active(conn)
                raise
            conn.execute("COMMIT")
            if fresh:
                logger.info(
                    "Initialized empty metadata index %s at schema version %d",
                    location,
                    CURRENT_SCHEMA_VERSION,
                )
                return []
        return run_migrations(conn, steps=MIGRATIONS, path=location)
    finally:
        conn.close()


def unique_index_step(
    *,
    version: int,
    table: str,
    columns: Sequence[str],
    index_name: str,
    description: str,
    dedup: Callable[[sqlite3.Connection, list[tuple[Any, ...]]], None] | None = None,
) -> MigrationStep:
    """Build a migration step that adds a unique index over existing rows.

    SQLite's ``ALTER TABLE`` cannot add constraints, so uniqueness lands on
    an existing table as ``CREATE UNIQUE INDEX``. Pre-existing duplicates
    would make that statement fail without naming any rows, so the step
    surfaces them first: it collects key tuples held by more than one row,
    offers them to the ``dedup`` hook for resolution, and re-checks. Keys
    still duplicated after the hook abort the step with
    :class:`UniqueIndexDuplicatesError` listing them, and the transaction
    rolls back to the pre-step state. Keys containing NULL are never
    reported as duplicates: SQLite unique indexes treat each NULL as
    distinct, so such rows do not block the index.

    Args:
        version: Schema version the step migrates the store to.
        table: Table receiving the index; an identifier owned by the
            migration author, never user input.
        columns: Column names forming the unique key, in index order.
        index_name: Name of the index to create.
        description: Short human-readable step summary.
        dedup: Optional hook called with the connection and the duplicated
            key tuples; it resolves duplicates by deleting or rewriting
            rows inside the step transaction.

    Returns:
        The migration step.
    """
    column_list = ", ".join(columns)
    non_null_keys = " AND ".join(f"{column} IS NOT NULL" for column in columns)

    def _find_duplicates(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
        with closing(conn.cursor()) as cur:
            cur.execute(
                f"SELECT {column_list} "  # nosec B608 - identifiers are migration-authored constants
                f"FROM {table} WHERE {non_null_keys} "
                f"GROUP BY {column_list} HAVING COUNT(*) > 1"
            )
            return [tuple(row) for row in cur.fetchall()]

    def _apply(conn: sqlite3.Connection) -> None:
        duplicates = _find_duplicates(conn)
        if duplicates and dedup is not None:
            dedup(conn, duplicates)
            duplicates = _find_duplicates(conn)
        if duplicates:
            raise UniqueIndexDuplicatesError(
                table=table, columns=columns, duplicates=duplicates
            )
        conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table} ({column_list})")

    return MigrationStep(version=version, description=description, apply=_apply)


def schema_fingerprint(conn: sqlite3.Connection) -> str:
    """Return a stable digest of the database schema and version stamp.

    Covers the normalized DDL of every user table and index from
    ``sqlite_master`` plus ``PRAGMA user_version``; row contents are not
    included. Freezing the fingerprint of each shipped version in a test
    guards the append-only rule: editing a shipped migration step changes
    the fingerprint its version produces, while appending a new step does
    not affect the fingerprints of prior versions.

    Args:
        conn: Open connection to the database to fingerprint.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    with closing(conn.cursor()) as cur:
        rows = cur.execute(
            "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    entries = [
        [entry_type, name, tbl_name, " ".join(sql.split())]
        for entry_type, name, tbl_name, sql in rows
    ]
    payload = json.dumps(
        {"user_version": _read_user_version(conn), "schema": entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
