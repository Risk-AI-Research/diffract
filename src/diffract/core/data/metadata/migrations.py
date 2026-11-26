"""Schema versioning and migrations for the SQLite metadata index.

The index stamps ``PRAGMA user_version`` so a database's schema generation
is readable without inspecting tables. Fresh databases run the migration
chain and record their producer in the singleton
``schema_meta`` table. Databases written at an older schema version are
refused at open time with instructions for guarded migration.

Store identity is checked before opening or migrating an index. The SQLite
application ID identifies a marked database. Unmarked files are checked by
:func:`_probe_expected_tables`.

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
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diffract.core.constants import TABLE_AGGREGATES, TABLE_PARAMETERS
from diffract.core.data.metadata.runs import (
    install_generation_edges,
    install_run_schema,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

SCHEMA_META_TABLE = "schema_meta"

# Frozen SQLite application ID, the big-endian ASCII bytes b"DIFF".
DIFFRACT_APPLICATION_ID = 0x44494646

# SQLite reports zero for an unset application ID.
_UNSTAMPED_APPLICATION_ID = 0

# Tables a diffract metadata index defines, with the columns each carries.
# Data tables are created by `SQLiteMetadataIndex.define_table`, which always
# adds `uid` and `json_data`; a foreign table that borrows a name is rejected
# on shape.
_EXPECTED_TABLES: dict[str, frozenset[str]] = {
    SCHEMA_META_TABLE: frozenset({"id", "created_by"}),
    TABLE_PARAMETERS: frozenset({"uid", "json_data"}),
    TABLE_AGGREGATES: frozenset({"uid", "json_data"}),
}

# Producer stamp for stores created before producer recording existed.
UNKNOWN_PRODUCER = "unknown (store predates schema versioning)"

_INSTANCE_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 3
_GUARDED_MIGRATION = "python -m diffract.migrate"
MIGRATION_SENTINEL = 0x4449464D

_BACKUP_ADVICE = (
    "Back up the value store and capture the metadata index with "
    "sqlite3.Connection.backup() or VACUUM INTO before upgrading."
)

_IDENTITY_ADVICE = (
    "Check the path: it must name the diffract metadata index (the "
    "'[metadata.sqlite] path' of your config), not another application's "
    "database. The file was left unchanged."
)

_SCHEMA_META_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_meta ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "created_by TEXT NOT NULL)"
)
_SCHEMA_META_V1_SQL = _SCHEMA_META_DDL.replace(" IF NOT EXISTS", "")
_IDENTITY_META_COLUMNS = (
    "hash_scheme_version INTEGER NOT NULL DEFAULT 1",
    "values_require_migration INTEGER NOT NULL DEFAULT 0 "
    "CHECK(values_require_migration IN (0, 1))",
)
_SCHEMA_META_V2_SQL = (
    _SCHEMA_META_V1_SQL[:-1] + ", " + ", ".join(_IDENTITY_META_COLUMNS) + ")"
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


class StoreIdentityError(Exception):
    """Raised when a database is not recognizable as a diffract metadata index.

    Attributes:
        path: Database location.
        check: Identity check that failed -- ``"application_id"`` when the
            file header is claimed by another application, or
            ``"expected_tables"`` when its tables are not diffract's.
    """

    def __init__(self, message: str, *, path: str, check: str) -> None:
        super().__init__(message)
        self.path = path
        self.check = check


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
        foreign_key_tables: Child tables whose references the step creates
            or rebuilds. Their constraints are checked before commit.
    """

    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]
    foreign_key_tables: tuple[str, ...] = ()


def _create_schema_meta(conn: sqlite3.Connection, created_by: str) -> None:
    conn.execute(_SCHEMA_META_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (id, created_by) VALUES (1, ?)",
        (created_by,),
    )


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    _create_schema_meta(conn, UNKNOWN_PRODUCER)


def verify_legacy_schema_meta(conn: sqlite3.Connection) -> bool:
    """Check the released singleton record before the migration stamps it."""
    definition = conn.execute(
        "SELECT sql FROM sqlite_schema WHERE name='schema_meta' COLLATE NOCASE"
    ).fetchone()
    if definition is None:
        return _read_user_version(conn) == 0
    return definition[0] == _SCHEMA_META_V1_SQL and list(
        conn.execute("SELECT id FROM schema_meta")
    ) == [(1,)]


_IDENTITY_OBJECT_DDL = {
    "instances": (
        "CREATE TABLE instances (instance_uid TEXT PRIMARY KEY NOT NULL, "
        "name TEXT NOT NULL UNIQUE, display_label TEXT, created_at INTEGER NOT NULL, "
        "json_data TEXT NOT NULL DEFAULT '{}')"
    ),
    "uid_map": (
        "CREATE TABLE uid_map (entity_kind TEXT NOT NULL "
        "CHECK(entity_kind IN ('parameter','aggregate','instance')), "
        "old_uid TEXT NOT NULL, new_uid TEXT NOT NULL, "
        "PRIMARY KEY(entity_kind, old_uid)) WITHOUT ROWID"
    ),
    "parameters": (
        "CREATE TABLE parameters (uid TEXT PRIMARY KEY NOT NULL, "
        "instance_uid TEXT NOT NULL REFERENCES instances(instance_uid), "
        "step INTEGER NOT NULL DEFAULT 0 CHECK(step >= 0), name TEXT NOT NULL, "
        "ptype TEXT NOT NULL DEFAULT 'UNKNOWN', json_data TEXT NOT NULL DEFAULT '{}')"
    ),
    "ix_param_name": "CREATE INDEX ix_param_name ON parameters(name)",
    "ix_param_ptype": "CREATE INDEX ix_param_ptype ON parameters(ptype)",
    "uq_param_natural": (
        "CREATE UNIQUE INDEX uq_param_natural "
        "ON parameters(instance_uid, step, name COLLATE BINARY)"
    ),
    "step_labels": (
        "CREATE TABLE step_labels (instance_uid TEXT NOT NULL "
        "REFERENCES instances(instance_uid), "
        "step INTEGER NOT NULL CHECK(step >= 0), label TEXT NOT NULL, "
        "PRIMARY KEY(instance_uid, step)) WITHOUT ROWID"
    ),
    "aggregates": (
        "CREATE TABLE aggregates (uid TEXT PRIMARY KEY NOT NULL, "
        "field_name TEXT NOT NULL, json_data TEXT NOT NULL DEFAULT '{}')"
    ),
    "ix_agg_field": "CREATE INDEX ix_agg_field ON aggregates(field_name)",
    "aggregate_members": (
        "CREATE TABLE aggregate_members ("
        "aggregate_uid TEXT NOT NULL REFERENCES aggregates(uid), "
        "member_type TEXT NOT NULL DEFAULT 'parameter' "
        "CHECK(member_type IN ('parameter')), "
        "member_uid TEXT NOT NULL REFERENCES parameters(uid), "
        "role TEXT NOT NULL DEFAULT 'member' "
        "CHECK(role IN ('member','source','target')), "
        "ordinal INTEGER, json_data TEXT NOT NULL DEFAULT '{}', "
        "PRIMARY KEY(aggregate_uid,member_type,member_uid,role)) WITHOUT ROWID"
    ),
    "ix_members_by_member": (
        "CREATE INDEX ix_members_by_member "
        "ON aggregate_members(member_uid,aggregate_uid)"
    ),
    "field_values": (
        "CREATE TABLE field_values (obj_uid TEXT NOT NULL, "
        "obj_kind TEXT NOT NULL CHECK(obj_kind IN ('parameter','aggregate')), "
        "field TEXT NOT NULL, backend TEXT NOT NULL, storage_key TEXT NOT NULL, "
        "size_bytes INTEGER, format_version INTEGER NOT NULL DEFAULT 0, "
        "tombstoned_at TEXT, revision TEXT, PRIMARY KEY(obj_uid, field)) WITHOUT ROWID"
    ),
    "ix_values_by_field": "CREATE INDEX ix_values_by_field ON field_values(field)",
    "live_field_values": (
        "CREATE VIEW live_field_values AS SELECT * FROM field_values "
        "WHERE tombstoned_at IS NULL"
    ),
    "assign_value_revision": (
        "CREATE TRIGGER assign_value_revision AFTER INSERT ON field_values "
        "WHEN NEW.revision IS NULL BEGIN "
        "UPDATE field_values SET revision=lower(hex(randomblob(16))) "
        "WHERE obj_uid=NEW.obj_uid AND field=NEW.field; END"
    ),
}


def conflicting_identity_objects(conn: sqlite3.Connection) -> list[str]:
    """List names the target schema needs that survived legacy table removal."""
    reserved = set(_IDENTITY_OBJECT_DDL) | {
        "counters",
        "runs",
        "ix_values_by_run",
        "require_value_generation_on_insert",
        "require_value_generation_on_update",
    }
    rows = conn.execute(
        "SELECT type,name,tbl_name FROM main.sqlite_schema WHERE sql IS NOT NULL"
    )
    return [
        name
        for kind, name, table in rows
        if name.lower() in reserved
        and not (
            table.lower() in {"parameters", "aggregates"} and kind in {"table", "index"}
        )
    ]


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    if {TABLE_PARAMETERS, TABLE_AGGREGATES}.intersection(_user_objects(conn)):
        raise MigrationError(
            f"Existing entity tables require the guarded migration. "
            f"Run {_GUARDED_MIGRATION}."
        )
    for definition in _IDENTITY_META_COLUMNS:
        conn.execute("ALTER TABLE schema_meta ADD COLUMN " + definition)
    for sql in _IDENTITY_OBJECT_DDL.values():
        conn.execute(sql)


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    install_run_schema(conn)
    install_generation_edges(conn, producer=_producer_stamp())


@cache
def _provenance_object_ddl() -> frozenset[str]:
    # SQLite rewrites the stored field_values DDL when columns are appended.
    # Derive the exact expected result from the same numbered migration steps.
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(_IDENTITY_OBJECT_DDL["field_values"])
        _migrate_to_v3(conn)
        return frozenset(
            sql
            for (sql,) in conn.execute(
                "SELECT sql FROM main.sqlite_schema WHERE sql IS NOT NULL"
            )
        )


# Shipped migration chain. Append-only: never rewrite or renumber a step.
MIGRATIONS: tuple[MigrationStep, ...] = (
    MigrationStep(
        version=1,
        description="record the store producer in the schema_meta table",
        apply=_migrate_to_v1,
    ),
    MigrationStep(
        2,
        "create instance, parameter, aggregate and field catalog identities",
        _migrate_to_v2,
        foreign_key_tables=("parameters", "step_labels", "aggregate_members"),
    ),
    MigrationStep(
        PROVENANCE_SCHEMA_VERSION,
        "create the run journal and attach generation edges to values",
        _migrate_to_v3,
        foreign_key_tables=("field_values",),
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


def _read_application_id(conn: sqlite3.Connection) -> int:
    with closing(conn.cursor()) as cur:
        return int(cur.execute("PRAGMA application_id").fetchone()[0])


def _set_application_id(conn: sqlite3.Connection, application_id: int) -> None:
    # PRAGMA cannot bind parameters; the value is an int owned by this module.
    conn.execute(f"PRAGMA application_id = {int(application_id)}")


def _rollback_if_active(conn: sqlite3.Connection) -> None:
    # SQLite auto-rolls-back on some failures (ON CONFLICT ROLLBACK, disk
    # full); an unconditional ROLLBACK would then raise and mask the error
    # being handled.
    if conn.in_transaction:
        with suppress(sqlite3.Error):
            conn.execute("ROLLBACK")


def _user_objects(conn: sqlite3.Connection) -> dict[str, str]:
    with closing(conn.cursor()) as cur:
        rows = cur.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {name.lower(): kind for name, kind in rows}


def _has_user_objects(conn: sqlite3.Connection) -> bool:
    return bool(_user_objects(conn))


def _table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    with closing(conn.cursor()) as cur:
        rows = cur.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()
    return frozenset(name for (name,) in rows)


def _named(names: Iterable[str], *, shown: int = 10) -> str:
    listed = sorted(names)
    overflow = len(listed) - shown
    suffix = f" (and {overflow} more)" if overflow > 0 else ""
    return f"{', '.join(listed[:shown])}{suffix}"


def _probe_expected_tables(conn: sqlite3.Connection) -> str | None:
    """Report a mismatch unless at least one recognized table is present.

    Each recognized table present must contain its required columns. Unrelated
    tables are ignored. Return None when the probe succeeds.
    """
    objects = _user_objects(conn)
    if not objects:
        return (
            "the file holds no tables and carries no diffract marker, so "
            "nothing identifies it as a store (a new store is created by "
            "opening a session at this path)"
        )
    present = sorted(
        name
        for name, object_type in objects.items()
        if object_type == "table" and name in _EXPECTED_TABLES
    )
    if not present:
        return f"it holds no table a diffract store defines (found: {_named(objects)})"
    for name in present:
        missing = _EXPECTED_TABLES[name] - _table_columns(conn, name)
        if missing:
            return (
                f"its '{name}' table is missing the column(s) "
                f"{_named(missing)} that a diffract store defines"
            )
    return None


def verify_store_identity(
    conn: sqlite3.Connection, *, path: str, allow_empty: bool
) -> None:
    """Refuse a database that is not a diffract metadata index.

    Unmarked stores are identified by :func:`_probe_expected_tables`.

    Args:
        conn: Open connection to the database to check.
        path: Database location, used in error messages.
        allow_empty: Whether an unmarked database holding nothing passes.
            True for callers that create the store they open; False for the
            upgrade entry point, which has nothing to upgrade and no
            evidence that the file is a diffract store.

    Raises:
        StoreIdentityError: If the header is claimed by another application
            or the table probe rejects the schema. The database is not
            modified.
    """
    application_id = _read_application_id(conn)
    if application_id == DIFFRACT_APPLICATION_ID:
        return
    if application_id != _UNSTAMPED_APPLICATION_ID:
        raise StoreIdentityError(
            f"Database at '{path}' is not a diffract metadata index: its "
            f"SQLite application_id is {application_id:#x}, and a diffract "
            f"store carries {DIFFRACT_APPLICATION_ID:#x} (failed check: "
            f"application_id). {_IDENTITY_ADVICE}",
            path=path,
            check="application_id",
        )

    if allow_empty and not _has_user_objects(conn):
        return
    reason = _probe_expected_tables(conn)
    if reason is None:
        return
    raise StoreIdentityError(
        f"Database at '{path}' is not a diffract metadata index: {reason} "
        f"(failed check: expected_tables). {_IDENTITY_ADVICE}",
        path=path,
        check="expected_tables",
    )


def verify_identity_layout(conn: sqlite3.Connection, *, path: str) -> None:
    """Verify release-owned identity objects and the store's schema record."""
    actual = {
        sql
        for (sql,) in conn.execute(
            "SELECT sql FROM main.sqlite_schema WHERE sql IS NOT NULL"
        )
    }
    required = set(_IDENTITY_OBJECT_DDL.values())
    if _read_user_version(conn) >= PROVENANCE_SCHEMA_VERSION:
        required.remove(_IDENTITY_OBJECT_DDL["field_values"])
        required.update(_provenance_object_ddl())
    if _SCHEMA_META_V2_SQL in actual and actual.issuperset(required):
        rows = conn.execute(
            "SELECT id,hash_scheme_version,values_require_migration FROM schema_meta"
        ).fetchall()
        if rows in ([(1, 1, 0)], [(1, 1, 1)]):
            return
    raise SchemaVersionError(
        f"Incomplete identity schema at '{path}'. Restore the original index "
        f"and value stores from backup and run {_GUARDED_MIGRATION}.",
        path=path,
        found=_read_user_version(conn),
        expected=CURRENT_SCHEMA_VERSION,
    )


def _apply_step(conn: sqlite3.Connection, step: MigrationStep) -> None:
    step.apply(conn)
    for table in step.foreign_key_tables:
        violation = conn.execute(
            "SELECT parent, fkid FROM pragma_foreign_key_check(?) LIMIT 1", (table,)
        ).fetchone()
        if violation is not None:
            parent, constraint = violation
            raise MigrationError(
                f"migration to version {step.version}: {table} contains a row "
                f"with no matching {parent} for foreign key {constraint}"
            )
    _set_user_version(conn, step.version)


def _initialize_at_head(conn: sqlite3.Connection, *, path: str) -> None:
    _validate_chain(MIGRATIONS)
    # Hold the file lock across the independent commits so another opener
    # cannot mistake a live initialization for a store requiring migration.
    mode = conn.execute("PRAGMA locking_mode").fetchone()[0]
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    try:
        conn.execute("BEGIN EXCLUSIVE")
        version = _read_user_version(conn)
        if version == 0 and not _has_user_objects(conn):
            verify_store_identity(conn, path=path, allow_empty=True)
            _set_application_id(conn, DIFFRACT_APPLICATION_ID)
            _create_schema_meta(conn, _producer_stamp())
        else:
            verify_store_identity(conn, path=path, allow_empty=False)
            if version != CURRENT_SCHEMA_VERSION:
                raise _version_mismatch(
                    path=path, found=version, expected=CURRENT_SCHEMA_VERSION
                )
            verify_identity_layout(conn, path=path)
        conn.execute("COMMIT")
        run_migrations(conn, steps=MIGRATIONS, path=path)
    finally:
        _rollback_if_active(conn)
        if mode == "normal":
            conn.execute("PRAGMA locking_mode=NORMAL")
            _read_user_version(conn)


def _version_mismatch(*, path: str, found: int, expected: int) -> SchemaVersionError:
    if found == MIGRATION_SENTINEL:
        message = (
            f"Half-migrated store at '{path}': restore the metadata and value "
            "stores from backup, then re-run the migration."
        )
    elif found > expected:
        message = (
            f"Metadata index at '{path}' has schema version {found}, but "
            f"this library supports up to version {expected}: the store was "
            "written by a newer release. Upgrade the library to open it."
        )
    elif found >= _INSTANCE_SCHEMA_VERSION:
        message = (
            f"Metadata index at '{path}' has schema version {found}; this build "
            f"requires version {expected}. {_BACKUP_ADVICE} Run "
            "diffract.upgrade_metadata_index(path) before reopening the store."
        )
    else:
        message = (
            f"Metadata index at '{path}' has schema version {found}; the "
            f"current version is {expected}. Stores are never migrated "
            f"implicitly on open. {_BACKUP_ADVICE} Migrating existing data requires "
            f"the guarded identity migration. Run {_GUARDED_MIGRATION} '{path}' "
            "--dry-run to inspect the plan. If initial "
            "creation was interrupted and you have verified the store contains "
            "no user data, it can be removed and initialized again."
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
    if step.version > version + 1:
        raise MigrationError(
            f"store version changed to {version} while migrating {path}; "
            f"expected {step.version - 1} before applying version {step.version}"
        )


def ensure_schema_version(conn: sqlite3.Connection, *, path: str) -> None:
    """Verify that a metadata database is at the current schema version.

    Fresh databases record their producer in ``schema_meta`` and run the
    migration chain to ``CURRENT_SCHEMA_VERSION``. Databases at
    any other version are refused; upgrading is an explicit, user-invoked
    action, never a side effect of opening the store.

    Args:
        conn: Open connection in autocommit mode (``isolation_level=None``).
        path: Database location, used in error messages.

    Raises:
        StoreIdentityError: If the database is not a diffract metadata index.
            Checked before the version, so a misconfigured path is never
            answered with instructions to migrate the file it names.
        SchemaVersionError: If the version differs or required identity objects
            or the singleton schema record are missing or incompatible.
    """
    if _read_user_version(conn) == 0 and not _has_user_objects(conn):
        _initialize_at_head(conn, path=path)
        return
    verify_store_identity(conn, path=path, allow_empty=False)
    version = _read_user_version(conn)
    if version != CURRENT_SCHEMA_VERSION:
        raise _version_mismatch(
            path=path, found=version, expected=CURRENT_SCHEMA_VERSION
        )
    verify_identity_layout(conn, path=path)


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
            _verify_step_precondition(step, version=version, target=target, path=path)
            if step.version <= version:
                conn.execute("COMMIT")
                continue
            _apply_step(conn, step)
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
    """Apply pending metadata migrations to a store with instance identities.

    Migrating data across the identity change requires a coordinated upgrade
    of the metadata and value stores.

    Args:
        path: Filesystem path of the metadata index database; a leading
            ``~`` expands to the user home.

    Returns:
        Applied version numbers, or an empty list if no upgrade was necessary.

    Raises:
        FileNotFoundError: If no database file exists at ``path`` (an
            in-memory store cannot be upgraded from outside its session).
        StoreIdentityError: If the file is not a diffract metadata index; it
            is left unchanged.
        SchemaVersionError: If the store requires a guarded migration or a
            newer release.
        MigrationError: If a step fails; the failing step is rolled back.
    """
    location = str(Path(path).expanduser())
    if not Path(location).is_file():
        raise FileNotFoundError(
            f"no metadata index database at '{location}'; nothing to upgrade"
        )

    conn = sqlite3.connect(location, isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        verify_store_identity(conn, path=location, allow_empty=False)
        version = _read_user_version(conn)
        if version == CURRENT_SCHEMA_VERSION:
            verify_identity_layout(conn, path=location)
            logger.info(
                "Metadata index %s is already at schema version %d",
                location,
                version,
            )
            return []
        if _INSTANCE_SCHEMA_VERSION <= version < CURRENT_SCHEMA_VERSION:
            verify_identity_layout(conn, path=location)
            return run_migrations(conn, steps=MIGRATIONS, path=location)
        if version == 0 and not _has_user_objects(conn):
            _initialize_at_head(conn, path=location)
            return []
        raise _version_mismatch(
            path=location, found=version, expected=CURRENT_SCHEMA_VERSION
        )
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
