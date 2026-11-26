"""Tests for metadata index schema versioning and migrations."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from diffract import __version__
from diffract.core.data.metadata import migrations
from diffract.core.data.metadata.migrations import (
    CURRENT_SCHEMA_VERSION,
    DIFFRACT_APPLICATION_ID,
    MIGRATIONS,
    MigrationError,
    MigrationStep,
    SchemaVersionError,
    StoreIdentityError,
    UniqueIndexDuplicatesError,
    run_migrations,
    schema_fingerprint,
    unique_index_step,
    upgrade_metadata_index,
)
from diffract.core.data.metadata.runs import RunJournal
from diffract.core.data.metadata.sqlite_index import (
    IN_MEMORY_DATABASE,
    SQLiteMetadataIndex,
)
from diffract.core.storage.sqlite_manager import SQLiteStorageManager
from diffract.migrate import migrate_store

pytestmark = pytest.mark.unit

# Frozen expectations for SHIPPED behavior. Deliberately literals, not
# imports: an import would follow any rewrite and never fail.
_SHIPPED_FINGERPRINTS = {
    3: "f0d2a288dcd306ec0aaae360c344babdab177aaa33d1a65c78b21cdafa5b230d",
    2: "5865a5f5b30cc42e1f3b4e70c25e0043ee6af3deb5c36c57d0ec72c0127b0b27",
    1: "d2598a0bdb25ccbe77d103e2383a90b09fe5d0f96780dbdfdb0c0a5c6a636009",
}
_UNKNOWN_PRODUCER_STAMP = "unknown (store predates schema versioning)"

_APPEND_ONLY_RULE = (
    "shipped migration steps are append-only: never edit or renumber a "
    "shipped step; append a new step and add its fingerprint here"
)

_ANOTHER_APPLICATION_ID = 0x12345678


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


def _application_id(path: Path | str) -> int:
    with sqlite3.connect(str(path)) as conn:
        return int(conn.execute("PRAGMA application_id").fetchone()[0])


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


def _make_future_store(path: Path) -> None:
    """Create a store as a newer release would leave it: the diffract header
    marker and bookkeeping table are in place, the schema generation is one
    past what this library ships."""
    conn = _connect_raw(path)
    try:
        conn.execute(f"PRAGMA application_id = {DIFFRACT_APPLICATION_ID}")
        conn.execute(
            "CREATE TABLE schema_meta (id INTEGER PRIMARY KEY CHECK (id = 1), "
            "created_by TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO schema_meta VALUES (1, 'diffract-core 9.9.9')")
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
    finally:
        conn.close()


def _make_foreign_store(path: Path, *, user_version: int = 0) -> None:
    """Create an unrelated application's database: its own tables, its own
    ``user_version`` bookkeeping, no diffract markers."""
    conn = _connect_raw(path)
    try:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO customers (name) VALUES ('acme')")
        conn.execute(f"PRAGMA user_version = {user_version}")
    finally:
        conn.close()


class TestFreshInitialization:
    @pytest.mark.parametrize("valid_references", [True, False])
    def test_fresh_and_upgraded_stores_run_the_same_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_references: bool
    ) -> None:
        old = tmp_path / "old.db"
        fresh = tmp_path / "fresh.db"
        with closing(SQLiteMetadataIndex(str(old))) as index:
            index.connect()
        with sqlite3.connect(old) as db:
            db.execute("UPDATE schema_meta SET created_by = 'original writer'")

        def add_membership(conn: sqlite3.Connection) -> None:
            conn.execute(
                "CREATE TABLE members (uid TEXT PRIMARY KEY, producer INTEGER "
                "REFERENCES schema_meta(id)) WITHOUT ROWID"
            )
            conn.execute(
                "INSERT INTO members VALUES ('item', ?)",
                (1 if valid_references else 2,),
            )

        next_version = CURRENT_SCHEMA_VERSION + 1
        step = MigrationStep(next_version, "add members", add_membership, ("members",))
        monkeypatch.setattr(migrations, "MIGRATIONS", (*MIGRATIONS, step))
        monkeypatch.setattr(migrations, "CURRENT_SCHEMA_VERSION", next_version)
        with closing(SQLiteMetadataIndex(str(fresh))) as index:
            if not valid_references:
                with pytest.raises(MigrationError, match=r"members.*schema_meta"):
                    index.connect()
                with (
                    closing(_connect_raw(old)) as db,
                    pytest.raises(MigrationError, match=r"members.*schema_meta"),
                ):
                    run_migrations(db, steps=migrations.MIGRATIONS)
                assert _user_version(old) == CURRENT_SCHEMA_VERSION
                assert "members" not in _table_names(old)
                assert _user_version(fresh) == CURRENT_SCHEMA_VERSION
                assert "members" not in _table_names(fresh)
                with closing(_connect_raw(fresh)) as db:
                    assert db.execute(
                        "SELECT created_by FROM schema_meta"
                    ).fetchone() == (f"diffract-core {__version__}",)
                return
            index.connect()
        with closing(_connect_raw(old)) as db:
            assert run_migrations(db, steps=migrations.MIGRATIONS) == [next_version]
        assert upgrade_metadata_index(old) == []
        with (
            closing(sqlite3.connect(fresh)) as created,
            closing(sqlite3.connect(old)) as upgraded,
        ):
            assert schema_fingerprint(created) == schema_fingerprint(upgraded)
            for db in (created, upgraded):
                assert db.execute("SELECT uid, producer FROM members").fetchall() == [
                    ("item", 1)
                ]
            assert created.execute("SELECT created_by FROM schema_meta").fetchone() == (
                f"diffract-core {__version__}",
            )
            assert upgraded.execute(
                "SELECT created_by FROM schema_meta"
            ).fetchone() == ("original writer",)

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
            index.define_table("samples", {"name": str})
            index.insert("samples", "u1", name="weights")

            index.clear(None)

            assert index.count("samples") == 0
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
        db_path = temp_dir / "O'Brien" / "legacy.db"
        db_path.parent.mkdir()
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
        assert "python -m diffract.migrate" in message
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
        _make_future_store(db_path)

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(SchemaVersionError) as excinfo:
            index.connect()

        assert excinfo.value.found == CURRENT_SCHEMA_VERSION + 1
        assert "newer release" in str(excinfo.value)


class TestUpgradeEntryPoint:
    def test_rebuilding_a_referenced_table_preserves_membership(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy.db"
        _make_legacy_store(path)

        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE members (member TEXT REFERENCES parameters(uid) "
                "ON DELETE CASCADE)"
            )
            conn.execute("INSERT INTO members VALUES ('u1')")

        def rebuild_parameters(conn: sqlite3.Connection) -> None:
            conn.execute(
                "CREATE TABLE next_params (uid TEXT PRIMARY KEY, json_data TEXT)"
            )
            conn.execute(
                "INSERT INTO next_params SELECT uid, json_data FROM parameters"
            )
            conn.execute("DROP TABLE parameters")
            conn.execute("ALTER TABLE next_params RENAME TO parameters")
            assert conn.execute("PRAGMA foreign_key_check(members)").fetchone() is None

        step = MigrationStep(1, "rebuild parameters", rebuild_parameters, ("members",))
        with closing(_connect_raw(path)) as db:
            assert run_migrations(db, steps=(step,)) == [1]
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT member FROM members").fetchall() == [("u1",)]

    @pytest.mark.parametrize("version", [0, 1])
    @pytest.mark.parametrize("stamped", [True, False])
    def test_identity_migration_requires_guarded_upgrade(
        self, tmp_path, version, stamped
    ):
        path = tmp_path / "legacy.db"
        _make_legacy_store(path)
        with closing(_connect_raw(path)) as conn:
            run_migrations(conn, steps=MIGRATIONS[:version])
            if stamped:
                conn.execute(f"PRAGMA application_id = {DIFFRACT_APPLICATION_ID}")
        original = path.read_bytes()
        with pytest.raises(
            SchemaVersionError, match=r"python -m diffract\.migrate"
        ) as error:
            upgrade_metadata_index(path)
        assert (error.value.found, error.value.expected) == (
            version,
            CURRENT_SCHEMA_VERSION,
        )
        assert path.read_bytes() == original
        with (
            closing(SQLiteMetadataIndex(str(path))) as index,
            pytest.raises(SchemaVersionError),
        ):
            index.connect()
        assert path.read_bytes() == original

    @pytest.mark.parametrize("source_version", [0, 1])
    def test_guarded_upgrade_matches_the_fresh_identity_schema(
        self, tmp_path, source_version
    ):
        path = tmp_path / "metadata_index.db"
        _make_legacy_store(path)
        with closing(_connect_raw(path)) as db:
            run_migrations(db, steps=MIGRATIONS[:source_version])
            db.execute(
                "INSERT INTO parameters VALUES ('u2','other','MODEL_A','DENSE',?)",
                ('{"source":"weights"}',),
            )
        with closing(
            SQLiteStorageManager(str(tmp_path / "sqlite_storage.db"))
        ) as store:
            store.set_field("u1", "score", 9, table="parameters")
        plan = migrate_store(path, dry_run=True)
        assert plan["target_version"] == CURRENT_SCHEMA_VERSION
        assert _user_version(path) == source_version
        assert (
            migrate_store(path, backup_dir=tmp_path / "snapshot")["status"]
            == "migrated"
        )
        assert upgrade_metadata_index(path) == []
        with closing(_connect_raw(path)) as db:
            assert _user_version(path) == CURRENT_SCHEMA_VERSION
            assert (
                schema_fingerprint(db) == _SHIPPED_FINGERPRINTS[CURRENT_SCHEMA_VERSION]
            )
            rows = db.execute(
                "SELECT p.uid,i.name,p.step,p.json_data FROM parameters p "
                "JOIN instances i USING (instance_uid) ORDER BY i.name"
            ).fetchall()
            assert [(name, step, info) for _, name, step, info in rows] == [
                ("MODEL_A", 0, '{"source":"weights"}'),
                ("model_a", 0, "{}"),
            ]
            assert len({uid for uid, *_ in rows}) == 2
            assert all(len(bytes.fromhex(uid)) == 16 for uid, *_ in rows)
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
            assert db.execute("SELECT created_by FROM schema_meta").fetchone() == (
                f"diffract-core {__version__}",
            )

    def test_existing_foreign_references_block_guarded_upgrade_before_rekey(
        self, tmp_path
    ):
        path = tmp_path / "metadata_index.db"
        _make_legacy_store(path)
        with closing(_connect_raw(path)) as db:
            run_migrations(db, steps=MIGRATIONS[:1])
            db.execute("CREATE TABLE links(p TEXT REFERENCES parameters(uid))")
            db.execute("INSERT INTO links VALUES ('u1')")
        with closing(
            SQLiteStorageManager(str(tmp_path / "sqlite_storage.db"))
        ) as store:
            store.set_field("u1", "score", 9, table="parameters")
        original = path.read_bytes()
        with pytest.raises(MigrationError, match="links"):
            migrate_store(path, have_backup=True)
        assert path.read_bytes() == original
        with closing(_connect_raw(path)) as db:
            assert db.execute("SELECT p FROM links").fetchall() == [("u1",)]
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    def test_matching_version_with_incomplete_identity_tables_refuses_all_entry_points(
        self, tmp_path
    ):
        path = tmp_path / "metadata.db"
        _make_legacy_store(path)
        with closing(_connect_raw(path)) as db:
            run_migrations(db, steps=MIGRATIONS[:1])
            db.execute(
                "ALTER TABLE schema_meta ADD COLUMN "
                "hash_scheme_version INTEGER DEFAULT 1"
            )
            db.execute(
                "CREATE TABLE instances(instance_uid TEXT PRIMARY KEY, name TEXT)"
            )
            db.execute(f"PRAGMA application_id={DIFFRACT_APPLICATION_ID}")
            db.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION}")
        before = path.read_bytes()
        with closing(SQLiteMetadataIndex(str(path))) as index:
            for operation in (
                index.connect,
                lambda: upgrade_metadata_index(path),
                lambda: migrate_store(path),
            ):
                with pytest.raises(
                    SchemaVersionError, match="Incomplete identity schema"
                ):
                    operation()
            assert path.read_bytes() == before

    @pytest.mark.parametrize(
        "damage",
        [
            "DROP TABLE step_labels",
            "DROP TABLE instances",
            "DROP TABLE uid_map",
            "DROP INDEX uq_param_natural",
            "DROP TRIGGER assign_value_revision",
            "DROP TRIGGER require_value_generation_on_insert",
            "DROP TRIGGER require_value_generation_on_update",
            "DELETE FROM schema_meta",
            "ALTER TABLE schema_meta RENAME TO original_meta; CREATE VIEW schema_meta "
            "AS SELECT * FROM original_meta",
            "ALTER TABLE schema_meta RENAME TO original_meta; CREATE TABLE schema_meta "
            "AS SELECT * FROM original_meta UNION ALL SELECT * FROM original_meta",
            "UPDATE schema_meta SET hash_scheme_version=7",
            "DROP INDEX uq_param_natural; CREATE INDEX uq_param_natural "
            "ON parameters(instance_uid,step,name)",
        ],
    )
    def test_damaged_identity_contract_refuses_every_entry_point(
        self, tmp_path, damage
    ):
        path = tmp_path / "metadata.db"
        with closing(SQLiteMetadataIndex(str(path))) as index:
            index.connect()
        with closing(_connect_raw(path)) as db:
            db.executescript(damage)
        before = path.read_bytes()
        with closing(SQLiteMetadataIndex(str(path))) as index:
            for open_or_upgrade in (
                index.connect,
                lambda: upgrade_metadata_index(path),
                lambda: migrate_store(path),
            ):
                with pytest.raises(
                    SchemaVersionError, match="Incomplete identity schema"
                ):
                    open_or_upgrade()
        assert path.read_bytes() == before

    def test_direct_identity_step_preserves_existing_entities_on_refusal(
        self, tmp_path
    ):
        path = tmp_path / "existing.db"
        _make_legacy_store(path)
        with closing(_connect_raw(path)) as db:
            run_migrations(db, steps=MIGRATIONS[:1])
            db.execute(
                "INSERT INTO parameters SELECT 'u2',name,model_id,ptype,? "
                "FROM parameters",
                ('{"source":"second"}',),
            )
            before = schema_fingerprint(db)
            with pytest.raises(MigrationError, match="guarded migration"):
                run_migrations(db, steps=MIGRATIONS)
            assert schema_fingerprint(db) == before
            assert db.execute(
                "SELECT uid,json_data FROM parameters ORDER BY uid"
            ).fetchall() == [("u1", None), ("u2", '{"source":"second"}')]

    def test_upgrade_of_a_current_store_applies_nothing(self, temp_dir: Path) -> None:
        path = temp_dir / "current.db"
        with closing(SQLiteMetadataIndex(str(path))) as index:
            index.connect()
        original = path.read_bytes()
        assert upgrade_metadata_index(path) == []
        assert path.read_bytes() == original

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

    def test_upgrade_of_an_unclaimed_empty_file_refuses_and_changes_nothing(
        self, temp_dir: Path
    ) -> None:
        """An empty file carries no evidence that it is a diffract store, and
        the upgrade entry point mutates whatever it is pointed at. It must
        refuse rather than claim a file that may belong to another
        application; creating a store is what opening a session does."""
        db_path = temp_dir / "empty.db"
        _connect_raw(db_path).close()

        with pytest.raises(StoreIdentityError) as excinfo:
            upgrade_metadata_index(db_path)

        assert excinfo.value.check == "expected_tables"
        assert _user_version(db_path) == 0
        assert _table_names(db_path) == set()

    def test_upgrade_of_an_empty_file_the_library_claimed_initializes_it(
        self, temp_dir: Path
    ) -> None:
        """An empty file that carries the diffract application id is our own
        store, so the identity gate lets it through and it is initialized at
        head with the real producer -- the unknown-producer stamp stays
        reserved for stores that truly predate versioning."""
        db_path = temp_dir / "claimed.db"
        conn = _connect_raw(db_path)
        try:
            conn.execute(f"PRAGMA application_id = {DIFFRACT_APPLICATION_ID}")
        finally:
            conn.close()

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
        _make_future_store(db_path)

        with pytest.raises(SchemaVersionError) as excinfo:
            upgrade_metadata_index(db_path)

        assert excinfo.value.found == CURRENT_SCHEMA_VERSION + 1


class TestStoreIdentityGate:
    def test_upgrading_a_foreign_database_refuses_and_leaves_it_untouched(
        self, temp_dir: Path
    ) -> None:
        """The entry point mutates the file it is pointed at, and a
        misconfigured metadata path points it at an unrelated database. It
        must refuse before writing: stamping ``user_version`` on a foreign
        store desyncs whatever migration bookkeeping that application keys
        off it."""
        db_path = temp_dir / "app.db"
        _make_foreign_store(db_path)

        with pytest.raises(StoreIdentityError) as excinfo:
            upgrade_metadata_index(db_path)

        error = excinfo.value
        assert error.check == "expected_tables"
        assert str(db_path) in str(error)
        assert "customers" in str(error)
        assert _user_version(db_path) == 0
        assert _table_names(db_path) == {"customers"}
        assert _application_id(db_path) == 0

    def test_upgrading_a_database_claimed_by_another_application_refuses(
        self, temp_dir: Path
    ) -> None:
        """A file stamped with someone else's ``application_id`` is refused on
        that stamp alone -- even when its tables happen to be shaped like a
        diffract store, the owner has declared the format."""
        db_path = temp_dir / "other.db"
        _make_legacy_store(db_path)
        conn = _connect_raw(db_path)
        try:
            conn.execute(f"PRAGMA application_id = {_ANOTHER_APPLICATION_ID}")
        finally:
            conn.close()

        with pytest.raises(StoreIdentityError) as excinfo:
            upgrade_metadata_index(db_path)

        assert excinfo.value.check == "application_id"
        assert _user_version(db_path) == 0
        assert "schema_meta" not in _table_names(db_path)

    def test_a_foreign_table_borrowing_a_diffract_name_is_refused(
        self, temp_dir: Path
    ) -> None:
        """``parameters`` is a plausible table name in an unrelated
        application, so the probe checks shape as well as name: a table
        without the columns every diffract table carries is not ours."""
        db_path = temp_dir / "params.db"
        conn = _connect_raw(db_path)
        try:
            conn.execute("CREATE TABLE parameters (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO parameters VALUES ('retries', '3')")
        finally:
            conn.close()

        with pytest.raises(StoreIdentityError) as excinfo:
            upgrade_metadata_index(db_path)

        error = excinfo.value
        assert error.check == "expected_tables"
        assert "parameters" in str(error)
        assert _user_version(db_path) == 0
        assert _table_names(db_path) == {"parameters"}

    def test_opening_a_foreign_database_refuses_without_upgrade_instructions(
        self, temp_dir: Path
    ) -> None:
        """The open-time refusal for an unversioned store tells the user to
        run the upgrade on that path. When the path is not a diffract store
        at all, following that instruction is exactly the mutation to avoid:
        the message must name the identity check instead."""
        db_path = temp_dir / "app.db"
        _make_foreign_store(db_path)

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(StoreIdentityError) as excinfo:
            index.connect()

        message = str(excinfo.value)
        assert "diffract.upgrade_metadata_index" not in message
        assert index._conn is None
        assert _table_names(db_path) == {"customers"}

    def test_opening_a_foreign_database_at_the_current_version_is_refused(
        self, temp_dir: Path
    ) -> None:
        """``user_version`` is application-defined bookkeeping: an unrelated
        database can already sit at the value diffract uses. Matching that
        number is not identity, so the store must still be refused rather
        than opened and written to."""
        db_path = temp_dir / "collision.db"
        _make_foreign_store(db_path, user_version=CURRENT_SCHEMA_VERSION)

        index = SQLiteMetadataIndex(str(db_path))
        with pytest.raises(StoreIdentityError) as excinfo:
            index.connect()

        assert excinfo.value.check == "expected_tables"
        assert index._conn is None
        assert _table_names(db_path) == {"customers"}

    def test_a_fresh_store_carries_the_application_id(self, temp_dir: Path) -> None:
        """Every store created from now on is identifiable without inspecting
        its tables."""
        db_path = temp_dir / "index.db"

        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
        finally:
            index.close()

        assert _application_id(db_path) == DIFFRACT_APPLICATION_ID

    def test_a_versioned_store_without_the_marker_still_opens(
        self, temp_dir: Path
    ) -> None:
        """Stores written by the release that introduced schema versioning
        carry no ``application_id``. The gate must admit that generation
        through the table probe, or the marker would lock out the very
        stores it was added to protect."""
        db_path = temp_dir / "unmarked.db"
        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
            index.define_table("samples", {"name": str})
            index.insert("samples", "u1", name="weights")
        finally:
            index.close()
        conn = _connect_raw(db_path)
        try:
            conn.execute("PRAGMA application_id = 0")
        finally:
            conn.close()

        reopened = SQLiteMetadataIndex(str(db_path))
        try:
            reopened.connect()
            record = reopened.get("samples", "u1")
        finally:
            reopened.close()

        assert record is not None
        assert record["name"] == "weights"

    def test_unowned_tables_do_not_make_a_store_unrecognizable(
        self, temp_dir: Path
    ) -> None:
        """A file can hold tables this module does not own -- here the SQLite
        value store's, from a file both tiers were pointed at. Such a store
        reaches the table probe whenever it predates the marker, and
        exclusivity buys no identity signal: refusing a file that
        demonstrably holds the index's own tables would lock a real store
        out of its own library."""
        db_path = temp_dir / "shared.db"
        index = SQLiteMetadataIndex(str(db_path))
        try:
            index.connect()
            index.define_table("samples", {"name": str})
            index.insert("samples", "u1", name="weights")
        finally:
            index.close()
        storage = SQLiteStorageManager(str(db_path))
        try:
            storage.connect()
            storage.set_field("u1", "weights", b"payload")
        finally:
            storage.close()
        conn = _connect_raw(db_path)
        try:
            conn.execute("PRAGMA application_id = 0")
        finally:
            conn.close()

        reopened = SQLiteMetadataIndex(str(db_path))
        try:
            reopened.connect()
            record = reopened.get("samples", "u1")
        finally:
            reopened.close()

        assert record is not None
        assert record["name"] == "weights"

    def test_a_refused_upgrade_never_stamps_the_store(self, temp_dir: Path) -> None:
        """Identity admits a store, but the version check can still refuse it.
        The marker is a write, so it must land only on a store the call
        actually migrates -- a refusal leaves the file byte-identical."""
        db_path = temp_dir / "ahead.db"
        _make_legacy_store(db_path)
        conn = _connect_raw(db_path)
        try:
            conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        finally:
            conn.close()

        with pytest.raises(SchemaVersionError):
            upgrade_metadata_index(db_path)

        assert _application_id(db_path) == 0

    def test_application_id_is_frozen(self) -> None:
        """The marker is written into the file header of every store. Changing
        it orphans every database already stamped, so the value is frozen
        here as a literal rather than imported."""
        assert DIFFRACT_APPLICATION_ID == 0x44494646


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


def test_concurrent_open_during_creation_waits_for_complete_schema(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from diffract.core.data.metadata.errors import IndexContentionError

    path = str(tmp_path / "new.db")
    in_step = Event()
    finish = Event()
    last = MIGRATIONS[-1]

    def pause_inside_step(conn):
        in_step.set()
        assert finish.wait(5)
        last.apply(conn)

    step = MigrationStep(last.version, last.description, pause_inside_step)
    monkeypatch.setattr(migrations, "MIGRATIONS", (*MIGRATIONS[:-1], step))
    with ThreadPoolExecutor(1) as pool, closing(SQLiteMetadataIndex(path)) as creator:
        future = pool.submit(creator.connect)
        try:
            assert in_step.wait(5)
            with (
                closing(SQLiteMetadataIndex(path, timeout=0)) as contender,
                pytest.raises(IndexContentionError),
            ):
                contender.connect()
        finally:
            finish.set()
        future.result(timeout=5)
        with closing(SQLiteMetadataIndex(path)) as reader:
            assert reader.query_rows(
                "schema_meta", columns=["hash_scheme_version"]
            ) == [{"hash_scheme_version": 1}]


def test_upgrade_preserves_revisions_and_assigns_new_ones_to_generated_fields(tmp_path):
    path = tmp_path / "metadata.db"
    with closing(_connect_raw(path)) as db:
        run_migrations(db, steps=MIGRATIONS[:2])
        db.execute(f"PRAGMA application_id={DIFFRACT_APPLICATION_ID}")
        db.executemany(
            "INSERT INTO field_values(obj_uid,obj_kind,field,backend,storage_key,"
            "tombstoned_at) VALUES (?,'parameter','f','sqlite',?,?)",
            [("a", "a/f", None), ("b", "b/f", "erased")],
        )
        before = db.execute(
            "SELECT obj_uid,revision,tombstoned_at FROM field_values ORDER BY obj_uid"
        ).fetchall()

    assert upgrade_metadata_index(path) == [3]
    with closing(SQLiteMetadataIndex(str(path))) as index:
        run = RunJournal(index).start("ingest")
        index._conn.execute(
            "INSERT INTO field_values(obj_uid,obj_kind,field,backend,storage_key,"
            "format_version,run_id,output_name) "
            "VALUES ('c','parameter','f','sqlite','c/f',1,?,'f')",
            (run.run_id,),
        )
        after = index._conn.execute(
            "SELECT obj_uid,revision,tombstoned_at FROM field_values ORDER BY obj_uid"
        ).fetchall()

    assert after[:2] == before
    revisions = {revision for _, revision, _ in after}
    assert len(revisions) == 3
    assert all(len(bytes.fromhex(revision)) == 16 for revision in revisions)
