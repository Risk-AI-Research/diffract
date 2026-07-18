"""Tests for metadata index schema versioning and migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from diffract import __version__
from diffract.core.data.metadata.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    MigrationError,
    MigrationStep,
    SchemaVersionError,
    UniqueIndexDuplicatesError,
    run_migrations,
    schema_fingerprint,
    unique_index_step,
    upgrade_metadata_index,
)
from diffract.core.data.metadata.sqlite_index import (
    IN_MEMORY_DATABASE,
    SQLiteMetadataIndex,
)

pytestmark = pytest.mark.unit

# Frozen expectations for SHIPPED behavior. Deliberately literals, not
# imports: an import would follow any rewrite and never fail.
_SHIPPED_FINGERPRINTS = {
    1: "d2598a0bdb25ccbe77d103e2383a90b09fe5d0f96780dbdfdb0c0a5c6a636009",
}
_UNKNOWN_PRODUCER_STAMP = "unknown (store predates schema versioning)"

_APPEND_ONLY_RULE = (
    "shipped migration steps are append-only: never edit or renumber a "
    "shipped step; append a new step and add its fingerprint here"
)


def _connect_raw(path: Path | str) -> sqlite3.Connection:
    return sqlite3.connect(str(path), isolation_level=None)


def _user_version(path: Path | str) -> int:
    with sqlite3.connect(str(path)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _table_names(path: Path | str) -> set[str]:
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {name for (name,) in rows}


def _make_legacy_store(path: Path) -> None:
    """Create a database shaped like a store from before schema versioning:
    repository tables and rows exist, ``user_version`` was never stamped."""
    conn = _connect_raw(path)
    try:
        conn.execute(
            "CREATE TABLE parameters (uid TEXT PRIMARY KEY, name TEXT, "
            "model_id TEXT, ptype TEXT, json_data TEXT)"
        )
        conn.execute(
            "INSERT INTO parameters (uid, name, model_id, ptype) "
            "VALUES ('u1', 'weights', 'model_a', 'DENSE')"
        )
    finally:
        conn.close()


class TestFreshInitialization:
    def test_fresh_index_is_created_at_current_version(self, temp_dir: Path) -> None:
        """A brand-new database is stamped at the current schema version and
        records its producer; an unstamped fresh store would be
        indistinguishable from a legacy one on the next open."""
        db_path = temp_dir / "index.db"

        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
        finally:
            index.close()

        assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        assert rows == [(1, f"diffract-core {__version__}")]

    def test_in_memory_index_is_created_at_current_version(self) -> None:
        """The in-memory sentinel takes the fresh-database path on every
        connect instead of being mistaken for an unversioned store."""
        index = SQLiteMetadataIndex(IN_MEMORY_DATABASE)
        try:
            index.connect()
            version = index._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_SCHEMA_VERSION
        finally:
            index.close()

    def test_reopening_an_initialized_store_keeps_one_producer_row(
        self, temp_dir: Path
    ) -> None:
        """Initialization is idempotent across opens: the singleton producer
        row is written once, not duplicated or overwritten."""
        db_path = temp_dir / "index.db"

        for _ in range(2):
            index = SQLiteMetadataIndex(str(db_path))
            try:
                index.connect()
            finally:
                index.close()

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        assert rows == [(1, f"diffract-core {__version__}")]

    def test_clear_all_preserves_the_schema_meta_record(self, temp_dir: Path) -> None:
        """Clearing the whole index wipes data tables only; the producer
        record is infrastructure, and losing it would falsify provenance."""
        db_path = temp_dir / "index.db"
        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
            index.define_table("parameters", {"name": str})
            index.insert("parameters", "u1", name="weights")

            index.clear(None)

            assert index.count("parameters") == 0
        finally:
            index.close()

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        assert rows == [(1, f"diffract-core {__version__}")]


class TestVersionMismatchOnOpen:
    def test_opening_an_older_store_fails_fast_with_upgrade_instructions(
        self, temp_dir: Path
    ) -> None:
        """A store at version N-1 is refused, and the error is actionable:
        it names the store and current versions, the explicit upgrade entry
        point, and the backup recommendation. Silent auto-migration on open
        is exactly what this framework forbids."""
        db_path = temp_dir / "legacy.db"
        _make_legacy_store(db_path)

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(SchemaVersionError) as excinfo:
            index.connect()

        error = excinfo.value
        # The fixture is a pre-versioning store: user_version was never
        # stamped, so its version is 0 regardless of how many steps ship.
        assert error.found == 0
        assert error.expected == CURRENT_SCHEMA_VERSION
        message = str(error)
        # The full importable dotted path, not just the bare function name:
        # the user must be able to reach it from the message alone.
        entry_point = "diffract.upgrade_metadata_index"
        assert entry_point in message
        assert "Back up" in message
        assert str(db_path) in message

    def test_refused_open_leaves_no_connection_and_no_changes(
        self, temp_dir: Path
    ) -> None:
        """The failed open neither keeps a half-usable connection (a retry
        must re-check, not silently succeed) nor touches the store."""
        db_path = temp_dir / "legacy.db"
        _make_legacy_store(db_path)

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(SchemaVersionError):
            index.connect()
        with pytest.raises(SchemaVersionError):
            index.connect()

        assert index._conn is None
        assert _user_version(db_path) == 0
        assert "schema_meta" not in _table_names(db_path)

    def test_opening_a_newer_store_fails_fast(self, temp_dir: Path) -> None:
        """A store written by a newer release is refused instead of being
        read (and possibly corrupted) with stale schema assumptions."""
        db_path = temp_dir / "newer.db"
        conn = _connect_raw(db_path)
        try:
            conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        finally:
            conn.close()

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(SchemaVersionError) as excinfo:
            index.connect()

        assert excinfo.value.found == CURRENT_SCHEMA_VERSION + 1
        assert "newer release" in str(excinfo.value)


class TestUpgradeEntryPoint:
    def test_upgrade_migrates_a_legacy_store_to_the_current_version(
        self, temp_dir: Path
    ) -> None:
        """The entry point named by the open-time error works end to end:
        the legacy store reaches the current version with its rows intact,
        its producer recorded as unknown (never fabricated), and opens
        normally afterwards."""
        db_path = temp_dir / "legacy.db"
        _make_legacy_store(db_path)

        applied = upgrade_metadata_index(db_path)

        assert applied == list(range(1, CURRENT_SCHEMA_VERSION + 1))
        assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        assert rows == [(1, _UNKNOWN_PRODUCER_STAMP)]

        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
            index.define_table(
                "parameters", {"name": str, "model_id": str, "ptype": str}
            )
            record = index.get("parameters", "u1")
        finally:
            index.close()
        assert record is not None
        assert record["name"] == "weights"

    def test_upgrade_of_a_current_store_applies_nothing(self, temp_dir: Path) -> None:
        """Rerunning the upgrade is a no-op, not a re-application."""
        db_path = temp_dir / "legacy.db"
        _make_legacy_store(db_path)

        assert upgrade_metadata_index(db_path) != []
        assert upgrade_metadata_index(db_path) == []

    def test_upgrade_of_a_missing_file_refuses_and_creates_nothing(
        self, temp_dir: Path
    ) -> None:
        """A mistyped path must not silently materialize an empty database
        where the user believed their store lived."""
        db_path = temp_dir / "absent.db"

        with pytest.raises(FileNotFoundError) as excinfo:
            upgrade_metadata_index(db_path)

        assert str(db_path) in str(excinfo.value)
        assert not db_path.exists()

    def test_upgrade_of_an_empty_existing_file_stamps_the_current_producer(
        self, temp_dir: Path
    ) -> None:
        """An empty database file (created but never populated) is a fresh
        store, not a legacy one: the upgrade initializes it at head and
        records the real producer, matching what opening it would do. The
        unknown-producer stamp is reserved for stores that truly predate
        versioning."""
        db_path = temp_dir / "empty.db"
        _connect_raw(db_path).close()

        applied = upgrade_metadata_index(db_path)

        assert applied == []
        assert _user_version(db_path) == CURRENT_SCHEMA_VERSION
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        assert rows == [(1, f"diffract-core {__version__}")]

    def test_upgrade_of_a_newer_store_refuses(self, temp_dir: Path) -> None:
        """The entry point must refuse a store written by a newer release
        rather than report a successful no-op upgrade on a store the library
        cannot open."""
        db_path = temp_dir / "newer.db"
        conn = _connect_raw(db_path)
        try:
            conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        finally:
            conn.close()

        with pytest.raises(SchemaVersionError) as excinfo:
            upgrade_metadata_index(db_path)

        assert excinfo.value.found == CURRENT_SCHEMA_VERSION + 1


class TestMigrationRunner:
    def test_failing_step_rolls_back_schema_and_version_stamp(
        self, temp_dir: Path
    ) -> None:
        """Fault injection: a step that fails midway leaves both the schema
        and ``user_version`` at the pre-step state. A half-applied step or a
        stamp without its schema would be a corrupt store."""
        db_path = temp_dir / "fault.db"

        def _create_first(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE first_table (x INTEGER)")

        def _fail_midway(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE half_done (y INTEGER)")
            raise RuntimeError("injected fault")

        chain = (
            MigrationStep(1, "create the first table", _create_first),
            MigrationStep(2, "fail after partial work", _fail_midway),
        )

        conn = _connect_raw(db_path)
        try:
            with pytest.raises(MigrationError) as excinfo:
                run_migrations(conn, steps=chain, path=str(db_path))
        finally:
            conn.close()

        assert "version 2" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert _user_version(db_path) == 1
        tables = _table_names(db_path)
        assert "first_table" in tables
        assert "half_done" not in tables

    def test_non_contiguous_chain_is_rejected(self, temp_dir: Path) -> None:
        """A gap in step numbering means a store could reach the head
        version without every shipped step having run."""
        db_path = temp_dir / "gap.db"

        def _noop(conn: sqlite3.Connection) -> None:
            return None

        chain = (MigrationStep(1, "one", _noop), MigrationStep(3, "three", _noop))

        conn = _connect_raw(db_path)
        try:
            with pytest.raises(MigrationError, match="not contiguous"):
                run_migrations(conn, steps=chain, path=str(db_path))
        finally:
            conn.close()

    def test_already_applied_steps_are_not_re_executed(self, temp_dir: Path) -> None:
        """The runner decides per step, inside that step's transaction,
        whether it still needs to run. Re-running a chain whose steps are
        already applied must skip them, never re-execute their bodies -- the
        version is re-read under the write lock so a concurrent upgrader
        cannot cause a double-apply. The steps here are non-idempotent (a
        second CREATE TABLE would raise), so a re-execution would surface."""
        db_path = temp_dir / "reapply.db"
        calls: list[int] = []

        def _make_step(number: int) -> MigrationStep:
            def _apply(conn: sqlite3.Connection) -> None:
                calls.append(number)
                conn.execute(f"CREATE TABLE step_{number} (x INTEGER)")

            return MigrationStep(number, f"create step_{number}", _apply)

        chain = (_make_step(1), _make_step(2))

        conn = _connect_raw(db_path)
        try:
            first = run_migrations(conn, steps=chain, path=str(db_path))
            second = run_migrations(conn, steps=chain, path=str(db_path))
        finally:
            conn.close()

        assert first == [1, 2]
        assert second == []
        assert calls == [1, 2]  # each body ran exactly once
        assert _user_version(db_path) == 2

    def test_autorollback_inside_a_step_still_raises_migration_error(
        self, temp_dir: Path
    ) -> None:
        """When a step's own statement rolls the transaction back (ON
        CONFLICT ROLLBACK, as SQLite also does on disk-full), the runner
        must not raise 'cannot rollback - no transaction is active' from its
        cleanup and mask the contract: the caller still gets a MigrationError
        naming the step, and the store is unchanged."""
        db_path = temp_dir / "autorollback.db"
        conn = _connect_raw(db_path)
        conn.execute("CREATE TABLE u (x INTEGER UNIQUE)")
        conn.execute("INSERT INTO u VALUES (1)")

        def _conflict_rollback(conn: sqlite3.Connection) -> None:
            # ON CONFLICT ROLLBACK ends the transaction itself before the
            # Python exception propagates.
            conn.execute("INSERT OR ROLLBACK INTO u VALUES (1)")

        step = MigrationStep(1, "insert conflicting row", _conflict_rollback)

        try:
            with pytest.raises(MigrationError) as excinfo:
                run_migrations(conn, steps=(step,), path=str(db_path))
            row_count = conn.execute("SELECT COUNT(*) FROM u").fetchone()[0]
        finally:
            conn.close()

        assert "version 1" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, sqlite3.IntegrityError)
        assert _user_version(db_path) == 0
        assert row_count == 1


class TestUniqueIndexStep:
    @staticmethod
    def _make_duplicate_store(path: Path) -> None:
        conn = _connect_raw(path)
        try:
            conn.execute(
                "CREATE TABLE items (uid TEXT PRIMARY KEY, model_id TEXT, name TEXT)"
            )
            conn.executemany(
                "INSERT INTO items VALUES (?, ?, ?)",
                [("a", "m", "w"), ("b", "m", "w"), ("c", "m", "x")],
            )
        finally:
            conn.close()

    def test_remaining_duplicates_abort_the_step_and_are_listed(
        self, temp_dir: Path
    ) -> None:
        """Without a resolving hook the step must surface the offending key
        tuples and roll back; SQLite's own unique-index failure names no
        rows, which is precisely why the framework pre-checks."""
        db_path = temp_dir / "dup.db"
        self._make_duplicate_store(db_path)
        step = unique_index_step(
            version=1,
            table="items",
            columns=("model_id", "name"),
            index_name="uq_items_natural",
            description="enforce the natural key",
        )

        conn = _connect_raw(db_path)
        try:
            with pytest.raises(UniqueIndexDuplicatesError) as excinfo:
                run_migrations(conn, steps=(step,), path=str(db_path))
            row_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        finally:
            conn.close()

        error = excinfo.value
        assert error.duplicates == [("m", "w")]
        assert "items" in str(error)
        assert _user_version(db_path) == 0
        assert row_count == 3
        with sqlite3.connect(db_path) as check:
            index_row = check.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND name = 'uq_items_natural'"
            ).fetchone()
        assert index_row is None

    def test_dedup_hook_receives_duplicates_and_unblocks_the_index(
        self, temp_dir: Path
    ) -> None:
        """The hook is called with exactly the duplicated key tuples before
        index creation; after it resolves them the index is created and the
        version advances."""
        db_path = temp_dir / "dedup.db"
        self._make_duplicate_store(db_path)
        observed: list[list[tuple[Any, ...]]] = []

        def _keep_first(
            conn: sqlite3.Connection, duplicates: list[tuple[Any, ...]]
        ) -> None:
            observed.append(duplicates)
            for model_id, name in duplicates:
                conn.execute(
                    "DELETE FROM items WHERE model_id = ? AND name = ? "
                    "AND rowid NOT IN (SELECT MIN(rowid) FROM items "
                    "WHERE model_id = ? AND name = ?)",
                    (model_id, name, model_id, name),
                )

        step = unique_index_step(
            version=1,
            table="items",
            columns=("model_id", "name"),
            index_name="uq_items_natural",
            description="enforce the natural key",
            dedup=_keep_first,
        )

        conn = _connect_raw(db_path)
        try:
            applied = run_migrations(conn, steps=(step,), path=str(db_path))
            remaining = conn.execute("SELECT uid FROM items ORDER BY uid").fetchall()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO items VALUES ('d', 'm', 'x')")
        finally:
            conn.close()

        assert applied == [1]
        assert observed == [[("m", "w")]]
        assert remaining == [("a",), ("c",)]
        assert _user_version(db_path) == 1

    def test_residual_duplicates_after_a_partial_hook_still_abort(
        self, temp_dir: Path
    ) -> None:
        """A hook that resolves some keys but leaves others must not let a
        broken store through: the step re-checks after the hook and aborts
        listing only the keys still duplicated, so the failure names the
        offending rows instead of a bare unnamed IntegrityError."""
        db_path = temp_dir / "partial.db"
        conn = _connect_raw(db_path)
        conn.execute(
            "CREATE TABLE items (uid TEXT PRIMARY KEY, model_id TEXT, name TEXT)"
        )
        conn.executemany(
            "INSERT INTO items VALUES (?, ?, ?)",
            [("a", "m", "w"), ("b", "m", "w"), ("c", "m", "x"), ("d", "m", "x")],
        )
        conn.close()

        def _resolve_only_w(
            conn: sqlite3.Connection, duplicates: list[tuple[Any, ...]]
        ) -> None:
            # Deliberately fix just the ("m", "w") key, leaving ("m", "x").
            conn.execute("DELETE FROM items WHERE uid = 'b'")

        step = unique_index_step(
            version=1,
            table="items",
            columns=("model_id", "name"),
            index_name="uq_items_natural",
            description="enforce the natural key",
            dedup=_resolve_only_w,
        )

        conn = _connect_raw(db_path)
        try:
            with pytest.raises(UniqueIndexDuplicatesError) as excinfo:
                run_migrations(conn, steps=(step,), path=str(db_path))
        finally:
            conn.close()

        assert excinfo.value.duplicates == [("m", "x")]
        assert _user_version(db_path) == 0
        # The step rolled back: the hook's delete of 'b' is undone too.
        assert _table_names(db_path) == {"items"}
        with sqlite3.connect(db_path) as check:
            count = check.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        assert count == 4

    def test_keys_containing_null_are_not_treated_as_duplicates(
        self, temp_dir: Path
    ) -> None:
        """SQLite unique indexes treat each NULL as distinct, so multiple
        rows whose key contains NULL are legal and CREATE UNIQUE INDEX
        accepts them. The pre-check must agree: it must neither abort the
        step nor hand these rows to a dedup hook that would delete a row the
        index would have kept."""
        db_path = temp_dir / "nullkeys.db"
        conn = _connect_raw(db_path)
        conn.execute(
            "CREATE TABLE items (uid TEXT PRIMARY KEY, model_id TEXT, name TEXT)"
        )
        conn.executemany(
            "INSERT INTO items VALUES (?, ?, ?)",
            [("a", "m", None), ("b", "m", None), ("c", "m", "w")],
        )
        conn.close()

        hook_calls: list[list[tuple[Any, ...]]] = []

        def _record(
            conn: sqlite3.Connection, duplicates: list[tuple[Any, ...]]
        ) -> None:
            hook_calls.append(duplicates)

        step = unique_index_step(
            version=1,
            table="items",
            columns=("model_id", "name"),
            index_name="uq_items_natural",
            description="enforce the natural key",
            dedup=_record,
        )

        conn = _connect_raw(db_path)
        try:
            applied = run_migrations(conn, steps=(step,), path=str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        finally:
            conn.close()

        assert applied == [1]
        assert hook_calls == []  # NULL-bearing rows are not duplicates
        assert count == 3  # no row deleted
        assert _user_version(db_path) == 1


class TestAppendOnlyGuard:
    def test_shipped_chain_is_contiguous_and_defines_the_current_version(
        self,
    ) -> None:
        assert [step.version for step in MIGRATIONS] == list(
            range(1, len(MIGRATIONS) + 1)
        )
        assert set(_SHIPPED_FINGERPRINTS) == set(
            range(1, CURRENT_SCHEMA_VERSION + 1)
        ), _APPEND_ONLY_RULE

    def test_shipped_steps_produce_their_frozen_schemas(self, temp_dir: Path) -> None:
        """Each shipped chain prefix must reproduce the schema fingerprint
        frozen at the time that step shipped. Editing a shipped step changes
        its fingerprint and fails here; appending a new step adds a new
        entry without touching the old ones."""
        for version, expected in sorted(_SHIPPED_FINGERPRINTS.items()):
            db_path = temp_dir / f"chain_v{version}.db"
            conn = _connect_raw(db_path)
            try:
                run_migrations(conn, steps=MIGRATIONS[:version], path=str(db_path))
                actual = schema_fingerprint(conn)
            finally:
                conn.close()
            assert actual == expected, (
                f"schema produced by shipped migrations 1..{version} changed; "
                f"{_APPEND_ONLY_RULE}"
            )

    def test_migrated_producer_stamp_is_frozen(self, temp_dir: Path) -> None:
        """The unknown-producer stamp written by the shipped chain is
        shipped behavior; rewording it would rewrite what migrated stores
        record."""
        db_path = temp_dir / "stamp.db"
        conn = _connect_raw(db_path)
        try:
            run_migrations(conn, steps=MIGRATIONS, path=str(db_path))
            rows = conn.execute("SELECT id, created_by FROM schema_meta").fetchall()
        finally:
            conn.close()
        assert rows == [(1, _UNKNOWN_PRODUCER_STAMP)]

    def test_fresh_initialization_matches_the_migrated_schema(
        self, temp_dir: Path
    ) -> None:
        """Creating a store at head and migrating a store to head must
        converge on the same schema, or the two paths drift apart as steps
        accumulate."""
        fresh_path = temp_dir / "fresh.db"
        index = SQLiteMetadataIndex(str(fresh_path))
        try:
            index.connect()
        finally:
            index.close()

        conn = _connect_raw(fresh_path)
        try:
            fresh_fingerprint = schema_fingerprint(conn)
        finally:
            conn.close()

        assert fresh_fingerprint == _SHIPPED_FINGERPRINTS[CURRENT_SCHEMA_VERSION]
