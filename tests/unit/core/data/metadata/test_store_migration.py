"""Guarded migration of data written by a name-addressed store."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from diffract import Session
from diffract.core.data.metadata.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATION_SENTINEL,
    MIGRATIONS,
    MigrationError,
    run_migrations,
)
from diffract.core.storage.hdf5_manager import HDF5StorageManager
from diffract.core.storage.maintenance import check_store, collect_store
from diffract.core.storage.sqlite_manager import SQLiteStorageManager
from diffract.core.storage.zarr_manager import ZarrStorageManager
from diffract.migrate import migrate_store
from diffract.session import IncompatibleStoreError

pytestmark = pytest.mark.unit
MODELS = ("net_step2", "net_step10")
RENAMES = {"parameters": {"weight#0": "weight.0"}}


def _legacy(
    tmp: Path, backend: str = "sqlite", *, duplicate: bool = False
) -> tuple[Path, Path]:
    index = tmp / "metadata_index.db"
    cfg = tmp / "session.ini"
    if backend == "sqlite":
        path = tmp / "sqlite_storage.db"
        store = SQLiteStorageManager(str(path), array_threshold=64)
        option = "path"
    elif backend == "hdf5":
        path = tmp / "hdf5_storage.h5"
        store = HDF5StorageManager(str(path), swmr=False)
        option = "path"
    else:
        pytest.importorskip("zarr")
        path = tmp / "zarr_storage.zarr"
        store = ZarrStorageManager(str(path))
        option = "store_url"
    cfg.write_text(
        f"[metadata]\nbackend=sqlite\n[metadata.sqlite]\npath={index}\n"
        f"[storage]\nbackend={backend}\n[storage.{backend}]\n{option}={path}\n"
        "[cache]\nbackend=simple\n[cache.simple]\nmax_memory_mb=8\n[parallel.thread_pool]\nmax_workers=1\n"
        "[nn.extractor]\nskip_not_implemented_types=true\n"
    )
    with closing(sqlite3.connect(index)) as db, db:
        db.execute(
            "CREATE TABLE parameters(uid TEXT PRIMARY KEY, name TEXT, model_id TEXT, "
            "ptype TEXT, json_data TEXT)"
        )
        db.execute(
            "CREATE TABLE aggregates(uid TEXT PRIMARY KEY, field_name TEXT, "
            "context_models TEXT, context_params TEXT, json_data TEXT)"
        )
        data = [("p1", MODELS[0], 2), ("p2", MODELS[1], 10)]
        if duplicate:
            data.append(("p3", MODELS[0], 20))
        for uid, model, value in data:
            db.execute(
                "INSERT INTO parameters VALUES(?,?,?,?,?)",
                (
                    uid,
                    "weight#0",
                    model,
                    "DENSE",
                    json.dumps({"other_meta": {"origin": value}}),
                ),
            )
            store.set_field(
                uid, "weights", np.eye(8, dtype=np.float32) * value, table="parameters"
            )
            store.set_field(uid, "score", float(value), table="parameters")
        old = "l_overlap@models[net_step10,net_step2]@params[weight#0]"
        db.execute(
            "INSERT INTO aggregates VALUES(?,?,?,?,?)",
            (
                old,
                "l_overlap",
                json.dumps(MODELS),
                '["weight#0"]',
                '{"source":"legacy"}',
            ),
        )
        store.set_field(old, "value", np.arange(3), table="aggregates")
    store.close()
    return index, cfg


def _version(path: Path) -> int:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


@pytest.mark.parametrize("backend", ["sqlite", "hdf5", "zarr"])
def test_real_backends_preserve_fields_and_open_public_session(tmp_path, backend):
    index, cfg = _legacy(tmp_path, backend)
    blocked = migrate_store(index, config=cfg, dry_run=True)
    assert blocked["status"] == "blocked"
    assert "weight#0" in str(blocked["errors"]) or "'#'" in str(blocked["errors"])
    with pytest.raises(MigrationError, match="backup"):
        migrate_store(index, config=cfg, rename=RENAMES)
    assert _version(index) == 0
    before = index.read_bytes()
    plan = migrate_store(
        index, config=cfg, rename=RENAMES, parse_step_suffix=True, dry_run=True
    )
    assert plan["values"] == 5
    assert index.read_bytes() == before
    value_files = {
        "sqlite": tmp_path / "sqlite_storage.db",
        "hdf5": tmp_path / "hdf5_storage.h5",
    }
    value_file = value_files.get(backend)
    previous_bytes = value_file.read_bytes() if value_file is not None else None
    backups = tmp_path / "backup"
    result = migrate_store(
        index, config=cfg, rename=RENAMES, parse_step_suffix=True, backup_dir=backups
    )
    assert result["values"] == 5
    assert result["uid_map"] == 5
    assert result["retained_local_keys"] == (5 if value_file is not None else 0)
    if value_file is not None:
        assert value_file.read_bytes() == previous_bytes
    assert _version(backups / "metadata.db") == 0
    assert _version(index) == CURRENT_SCHEMA_VERSION
    assert not any(check_store(index, config=cfg).values())
    assert not any(
        collect_store(index, config=cfg, dry_run=True, grace_days=0).values()
    )
    with closing(sqlite3.connect(index)) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM live_field_values WHERE format_version=0"
        ).fetchone() == (5,)
        assert set(db.execute("SELECT ptype FROM parameters")) == {("DENSE",)}
        (meta,) = db.execute("SELECT json_data FROM aggregates").fetchone()
        assert json.loads(meta) == {"source": "legacy", "direction_unknown": True}
        markers = db.execute("SELECT role,json_data FROM aggregate_members").fetchall()
        assert [(role, json.loads(j)) for role, j in markers] == [
            ("member", {"direction_unknown": True})
        ] * 2
    with Session(config_path=cfg) as session:
        rows = session.results.export_metrics("score", "weights", export_format="dict")
        observations = sorted(
            (e["metadata"].get("step", r["step"]), e["fields"])
            for r in session.models.parameters.list()
            for uid, e in rows.items()
            if uid == r["uid"]
        )
        for (step, values), wanted in zip(observations, [2, 10], strict=True):
            assert step == wanted
            assert values["score"] == wanted
            np.testing.assert_array_equal(
                values["weights"], np.eye(8, dtype=np.float32) * wanted
            )
        (pair,) = session.results.export_aggregates("l_overlap", export_format="list")
        np.testing.assert_array_equal(pair["value"], np.arange(3))
    assert migrate_store(index)["status"] == "current"


@pytest.mark.parametrize("backend", ["sqlite", "hdf5"])
def test_retained_fields_survive_scoped_erase_and_native_rewrite(tmp_path, backend):
    index, cfg = _legacy(tmp_path, backend)
    migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")

    with Session(config_path=cfg) as session:

        @session.compute.kernel(require_fields=("weights",), produce_fields=("score",))
        def score(weights: np.ndarray) -> float:
            return float(weights.sum())

        def scores():
            return {
                uid: row["fields"]["score"]
                for uid, row in session.results.export_metrics(
                    "score", export_format="dict"
                ).items()
                if "score" in row["fields"]
            }

        original = scores()
        first = next(uid for uid, value in original.items() if value == 2)
        second = next(uid for uid, value in original.items() if value == 10)
        session.results.ingest_metrics({first: {"score": 40}}, force=True)
        assert scores() == {first: 40, second: 10}
        session.filter(param_ids=[first]).results.erase("score")
        assert scores() == {second: 10}
        session.results.ingest_metrics({first: {"score": 42}}, force=False)
        assert scores() == {first: 42, second: 10}

    with Session(config_path=cfg) as reopened:
        rows = reopened.results.export_metrics("score", export_format="dict")
        assert rows[first]["fields"]["score"] == 42
        assert rows[second]["fields"]["score"] == 10

    with closing(sqlite3.connect(index)) as db:
        keys = dict(
            db.execute(
                "SELECT obj_uid,storage_key FROM live_field_values WHERE field='score'"
            )
        )
        assert keys[first] == f"{first}/score"
        assert keys[second] != f"{second}/score"


@pytest.mark.parametrize(
    ("keep", "uid", "value"), [("oldest", "p1", 2), ("newest", "p3", 20)]
)
def test_duplicate_selection_and_provenance(tmp_path, keep, uid, value):
    index, cfg = _legacy(tmp_path, duplicate=True)
    blocked = migrate_store(index, config=cfg, rename=RENAMES, dry_run=True)
    assert blocked["status"] == "blocked"
    (conflict,) = blocked["duplicates"]
    assert [
        (r["old_uid"], r["provenance"]["other_meta"]["origin"])
        for r in conflict["rows"]
    ] == [("p1", 2), ("p3", 20)]
    migrate_store(
        index, config=cfg, rename=RENAMES, keep=keep, backup_dir=tmp_path / "backup"
    )
    with closing(sqlite3.connect(index)) as db:
        target = db.execute(
            "SELECT new_uid FROM uid_map WHERE old_uid=?", (uid,)
        ).fetchone()[0]
        assert db.execute(
            "SELECT new_uid FROM uid_map WHERE old_uid IN ('p1','p3')"
        ).fetchall() == [(target,), (target,)]
    with Session(config_path=cfg) as session:
        rows = session.results.export_metrics("score", export_format="dict")
        assert rows[target]["fields"]["score"] == value
        assert rows[target]["metadata"]["origin"] == value
        assert session.models.list() == sorted(MODELS)


def test_half_migrated_store_refuses_reopen_and_reentry_until_restored(tmp_path):
    index, cfg = _legacy(tmp_path)
    backup = tmp_path / "backup"
    child = """
import os
import sys
from pathlib import Path

from diffract.core.data.metadata import store_migration

move_values = store_migration._move_values


def crash(*args):
    move_values(*args)
    os._exit(78)


store_migration._move_values = crash
store_migration.migrate_store(
    Path(sys.argv[1]),
    config=Path(sys.argv[2]),
    rename={"parameters": {"weight#0": "weight.0"}},
    backup_dir=Path(sys.argv[3]),
)
"""
    result = subprocess.run(  # noqa: S603 - fixed fixture program
        [sys.executable, "-c", child, str(index), str(cfg), str(backup)],
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 78
    assert _version(index) == MIGRATION_SENTINEL
    with pytest.raises(MigrationError, match="restore"):
        migrate_store(index, config=cfg, have_backup=True)
    with (
        pytest.raises(IncompatibleStoreError, match="restore"),
        Session(config_path=cfg),
    ):
        pass
    # The two databases and any external arrays are restored with all handles closed.
    manifest = json.loads((backup / "restore.json").read_text())
    for entry, destination in manifest.items():
        if entry in {"metadata.db", "sqlite"}:
            with (
                closing(sqlite3.connect(backup / entry)) as source,
                closing(sqlite3.connect(destination)) as restored,
            ):
                source.backup(restored)
        else:
            shutil.copy2(backup / entry, destination)
    assert _version(index) == 0
    assert (
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)["values"]
        == 5
    )


@pytest.mark.parametrize("backend", ["sqlite", "hdf5", "zarr"])
def test_restoring_only_index_refuses_mixed_value_generations(tmp_path, backend):
    index, cfg = _legacy(tmp_path, backend)
    backup = tmp_path / "snapshot"
    assert (
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=backup)["values"]
        == 5
    )
    if backend in {"sqlite", "hdf5"}:
        with Session(config_path=cfg) as session:
            session.models.add({"new.weight": np.eye(2)}, model_id="new")
    shutil.copy2(backup / "metadata.db", index)
    for dry_run in (True, False):
        with pytest.raises(
            ValueError, match="Restore the metadata index and every value store"
        ):
            migrate_store(
                index, config=cfg, dry_run=dry_run, have_backup=True, rename=RENAMES
            )
        assert _version(index) == 0


def test_legacy_numeric_scalars_stay_readable_without_rewriting_rows(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(tmp_path / "sqlite_storage.db")) as db, db:
        db.execute(
            "UPDATE storage SET value_type='pickle',value_data=? WHERE field='score'",
            (pickle.dumps(np.float32(1.25), protocol=5),),
        )
    values = tmp_path / "sqlite_storage.db"
    before = values.read_bytes()
    result = migrate_store(
        index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup"
    )
    assert result["legacy_numeric_scalars"] == 2
    assert values.read_bytes() == before
    with Session(config_path=cfg) as session:
        assert [
            e["fields"]["score"]
            for e in session.results.export_metrics(
                "score", export_format="dict"
            ).values()
        ] == [1.25, 1.25]


def test_concurrent_source_change_after_preflight_keeps_legacy_store_openable(
    tmp_path, monkeypatch
):
    import importlib

    migration = importlib.import_module("diffract.core.data.metadata.store_migration")
    index, cfg = _legacy(tmp_path)
    backup = migration._backup

    def concurrent_update(*args):
        backup(*args)
        with closing(sqlite3.connect(index)) as db, db:
            db.execute(
                "UPDATE parameters SET json_data=? WHERE uid='p1'",
                ('{"other_meta":{"new":true}}',),
            )

    monkeypatch.setattr(migration, "_backup", concurrent_update)
    with pytest.raises(MigrationError, match="changed during preflight"):
        migrate_store(
            index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backups"
        )
    assert _version(index) == 0
    with closing(sqlite3.connect(index)) as db:
        assert json.loads(
            db.execute("SELECT json_data FROM parameters WHERE uid='p1'").fetchone()[0]
        )["other_meta"]["new"]


def test_unrecognized_legacy_encoding_refuses_before_sentinel(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(tmp_path / "sqlite_storage.db")) as db, db:
        db.execute(
            "UPDATE storage SET value_type='pickle',value_data=? WHERE field='score'",
            (pickle.dumps(["unsupported"], protocol=5),),
        )
    with pytest.raises(ValueError, match="Unsupported legacy pickle"):
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)
    assert _version(index) == 0


def test_backup_keeps_committed_wal_rows_and_external_arrays(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        with writer:
            writer.execute(
                "UPDATE parameters SET json_data=? WHERE uid='p1'",
                ('{"other_meta":{"wal":"kept"}}',),
            )
        with closing(
            sqlite3.connect(index.as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as base:
            assert (
                "kept"
                not in base.execute(
                    "SELECT json_data FROM parameters WHERE uid='p1'"
                ).fetchone()[0]
            )
        backup = tmp_path / "backup"
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=backup)
    with closing(sqlite3.connect(backup / "metadata.db")) as saved:
        assert json.loads(
            saved.execute("SELECT json_data FROM parameters WHERE uid='p1'").fetchone()[
                0
            ]
        ) == {"other_meta": {"wal": "kept"}}
    manifest = json.loads((backup / "restore.json").read_text())
    arrays = [
        (backup / src, Path(dest))
        for src, dest in manifest.items()
        if src.startswith("sqlite-arrays/")
    ]
    with closing(sqlite3.connect(backup / "sqlite")) as saved:
        expected = {
            Path(row[0])
            for row in saved.execute(
                "SELECT file_path FROM storage WHERE file_path IS NOT NULL"
            )
        }
    assert {original for _, original in arrays} == expected
    assert len(expected) == 3
    for snapshot, original in arrays:
        assert snapshot.read_bytes() == original.read_bytes()


def test_remote_store_retains_the_exact_legacy_object(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    index, cfg = _legacy(tmp_path, backend="zarr")
    url = "memory://diffract-" + tmp_path.name
    old = ZarrStorageManager(url)
    local = ZarrStorageManager(str(tmp_path / "zarr_storage.zarr"), readonly=True)
    try:
        for table in ("parameters", "aggregates"):
            for field in local.list_fields(table=table):
                for uid in local.list_objs_has_field(field, table=table):
                    old.set_field(
                        uid,
                        field,
                        local.get_field(uid, field, table=table),
                        table=table,
                    )
    finally:
        old.close()
        local.close()
    cfg.write_text(cfg.read_text().replace(str(tmp_path / "zarr_storage.zarr"), url))
    with pytest.raises(ValueError, match="remote"):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert _version(index) == 0
    report = migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)
    assert report["retained_remote_keys"] == 5
    with closing(sqlite3.connect(index)) as db:
        key, version = db.execute(
            "SELECT storage_key,format_version FROM field_values "
            "WHERE obj_kind='aggregate'"
        ).fetchone()
    assert version == 0
    assert key.startswith("aggregates/value/l_overlap%40models%5B")
    with Session(config_path=cfg) as session:
        (pair,) = session.results.export_aggregates("l_overlap", export_format="list")
        np.testing.assert_array_equal(pair["value"], np.arange(3))


def test_configured_index_and_session_relative_paths(tmp_path):
    from diffract import containers

    index, cfg = _legacy(tmp_path)
    root = Path(containers.__file__).resolve().parents[2]
    other = tmp_path / "other"
    other.mkdir()
    _, wrong = _legacy(other)
    with pytest.raises(ValueError, match="configured metadata index"):
        migrate_store(
            index, config=wrong, rename=RENAMES, backup_dir=tmp_path / "backup"
        )
    assert _version(index) == 0
    assert not (tmp_path / "backup").exists()
    cfg.write_text(
        cfg.read_text().replace(str(tmp_path), os.path.relpath(tmp_path, root))
    )
    assert (
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)["values"]
        == 5
    )
    with Session(config_path=cfg) as session:
        assert sorted(
            r["fields"]["score"]
            for r in session.results.export_metrics(
                "score", export_format="dict"
            ).values()
        ) == [2, 10]


def test_file_uri_backs_up_local_zarr_including_spaces(tmp_path):
    directory = tmp_path / "store with space"
    directory.mkdir()
    index, cfg = _legacy(directory, "zarr")
    path = directory / "zarr_storage.zarr"
    cfg.write_text(cfg.read_text().replace(str(path), path.as_uri()))
    backup = tmp_path / "backup"
    result = migrate_store(index, config=cfg, rename=RENAMES, backup_dir=backup)
    assert result["retained_remote_keys"] == 0
    assert (backup / "zarr").is_dir()
    with Session(config_path=cfg) as session:
        assert len(session.results.export_metrics("weights", export_format="dict")) == 2


def test_sqlite_accepts_slashes_in_old_uid_and_preserves_global_scope(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index)) as metadata, metadata:
        metadata.execute("UPDATE parameters SET name='weight/0'")
        metadata.execute(
            "UPDATE aggregates SET uid=replace(uid,'weight#0','weight/0'),"
            "context_params='[\"weight/0\"]'"
        )
        metadata.execute(
            "INSERT INTO aggregates VALUES('global','global','[]','[]','{}')"
        )
    with closing(sqlite3.connect(tmp_path / "sqlite_storage.db")) as values, values:
        values.execute(
            "UPDATE storage SET obj_uid=replace(obj_uid,'weight#0','weight/0') "
            "WHERE table_name='aggregates'"
        )
    migrate_store(
        index,
        config=cfg,
        rename={"parameters": {"weight/0": "weight.0"}},
        have_backup=True,
    )
    with closing(sqlite3.connect(index)) as db:
        uid, flags = db.execute(
            "SELECT uid,json_data FROM aggregates WHERE field_name='global'"
        ).fetchone()
        assert "direction_unknown" not in json.loads(flags)
        assert not db.execute(
            "SELECT * FROM aggregate_members WHERE aggregate_uid=?", (uid,)
        ).fetchall()
    with Session(config_path=cfg) as session:
        (pair,) = session.results.export_aggregates("l_overlap", export_format="list")
        np.testing.assert_array_equal(pair["value"], np.arange(3))


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE TABLE annotations(ref REFERENCES parameters(uid) "
        "ON DELETE CASCADE, note)",
        "ALTER TABLE parameters ADD COLUMN application_label TEXT",
        "CREATE TRIGGER on_parameter AFTER DELETE ON parameters BEGIN SELECT 1; END",
    ],
)
def test_unmapped_schema_extensions_block_before_backup(tmp_path, ddl):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index)) as db, db:
        db.execute(ddl)
    report = migrate_store(index, config=cfg, rename=RENAMES, dry_run=True)
    assert report["status"] == "blocked"
    assert report["errors"]
    with pytest.raises(MigrationError):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert not (tmp_path / "backup").exists()
    assert _version(index) == 0


@pytest.mark.parametrize(
    ("ddl", "object_name"),
    [
        ("CREATE TABLE Instances(note)", "Instances"),
        ("CREATE UNIQUE INDEX app_natural ON parameters(model_id,name)", "app_natural"),
        ("CREATE INDEX uq_param_natural ON app_events(id)", "uq_param_natural"),
        ("CREATE VIEW app_view AS SELECT model_id FROM parameters", "app_view"),
        (
            "CREATE TRIGGER app_hook AFTER UPDATE ON schema_meta BEGIN "
            "UPDATE parameters SET name='changed'; END",
            "app_hook",
        ),
        (
            "CREATE TRIGGER app_hook AFTER UPDATE OF id ON app_events BEGIN "
            "INSERT INTO app_events SELECT model_id FROM parameters; END",
            "app_hook",
        ),
    ],
)
def test_schema_extension_is_reported_before_either_store_changes(
    tmp_path, ddl, object_name
):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index, isolation_level=None)) as db:
        run_migrations(db, steps=MIGRATIONS[:1])
        db.execute("CREATE TABLE app_events(id)")
        db.execute(ddl)
    paths = (index, tmp_path / "sqlite_storage.db")
    before = [p.read_bytes() for p in paths]
    report = migrate_store(index, config=cfg, rename=RENAMES, dry_run=True)
    assert report["status"] == "blocked"
    assert object_name in str(report["errors"])
    with pytest.raises(MigrationError, match=object_name):
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)
    assert [p.read_bytes() for p in paths] == before


def test_malformed_legacy_schema_record_refuses_before_stamping(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index, isolation_level=None)) as db:
        run_migrations(db, steps=MIGRATIONS[:1])
        db.execute("ALTER TABLE schema_meta RENAME TO original_meta")
        db.execute("CREATE TABLE schema_meta AS SELECT * FROM original_meta")
    before = index.read_bytes()
    report = migrate_store(index, config=cfg, rename=RENAMES, dry_run=True)
    assert report["status"] == "blocked"
    assert "schema_meta" in str(report["errors"])
    with pytest.raises(MigrationError, match="schema_meta"):
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)
    assert index.read_bytes() == before


def test_case_insensitive_legacy_tables_and_nullable_type_migrate_with_values(tmp_path):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index)) as db, db:
        db.execute("ALTER TABLE aggregates RENAME TO staging")
        db.execute("ALTER TABLE staging RENAME TO AGGREGATES")
        db.execute("UPDATE parameters SET ptype=NULL WHERE uid='p1'")
        db.execute("CREATE TABLE app_notes(note)")
        db.execute("CREATE VIEW app_view AS SELECT note FROM app_notes")
        db.execute(
            "CREATE TRIGGER app_hook AFTER INSERT ON app_notes BEGIN SELECT 1; END"
        )
        db.execute("INSERT INTO app_notes VALUES('keep')")
    report = migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)
    assert (report["parameters"], report["aggregates"], report["values"]) == (2, 1, 5)
    with Session(config_path=cfg) as session:
        (pair,) = session.results.export_aggregates("l_overlap", export_format="list")
        np.testing.assert_array_equal(pair["value"], np.arange(3))
        params = session.results.export_metrics("score", export_format="dict")
        assert sorted(
            (r["metadata"]["parameter_type"], r["fields"]["score"])
            for r in params.values()
        ) == [("DENSE", 10), ("UNKNOWN", 2)]
    with closing(sqlite3.connect(index)) as db:
        assert db.execute("SELECT note FROM app_view").fetchall() == [("keep",)]
        assert db.execute(
            "SELECT name FROM sqlite_schema WHERE name='app_hook'"
        ).fetchone()


@pytest.mark.parametrize("column", ["uid", "model_id"])
def test_null_legacy_coordinate_is_reported_before_mutation(tmp_path, column):
    index, cfg = _legacy(tmp_path)
    with closing(sqlite3.connect(index)) as db, db:
        db.execute(f"UPDATE parameters SET {column}=NULL WHERE uid='p1'")  # nosec B608
    before = index.read_bytes()
    report = migrate_store(
        index, config=cfg, rename=RENAMES, dry_run=True, parse_step_suffix=True
    )
    assert report["status"] == "blocked"
    assert report["errors"]
    with pytest.raises(MigrationError):
        migrate_store(
            index, config=cfg, rename=RENAMES, have_backup=True, parse_step_suffix=True
        )
    assert index.read_bytes() == before


def test_parameter_admission_normalizes_and_rejects_empty_components(tmp_path):
    index, cfg = _legacy(tmp_path)
    res = migrate_store(
        index,
        config=cfg,
        rename={"parameters": {"weight#0": "bad..path"}},
        dry_run=True,
    )
    assert res["status"] == "blocked"
    assert "empty path component" in str(res["errors"])
    migrate_store(
        index,
        config=cfg,
        rename={"parameters": {"weight#0": "module.tensor"}},
        have_backup=True,
    )
    with closing(sqlite3.connect(index)) as db:
        assert db.execute("SELECT DISTINCT name FROM parameters").fetchall() == [
            ("tensor",)
        ]


@pytest.mark.parametrize("mode", ["insert", "overwrite"])
def test_value_updates_after_backup_leave_legacy_values_reachable(
    tmp_path, monkeypatch, mode
):
    from diffract.core.data.metadata import store_migration

    index, cfg = _legacy(tmp_path)
    values = tmp_path / "sqlite_storage.db"
    with closing(sqlite3.connect(values)) as db, db:
        db.execute(
            "UPDATE storage SET value_type='pickle',value_data=? "
            "WHERE obj_uid='p1' AND field='score'",
            (pickle.dumps(np.float32(1)),),
        )
    backup = store_migration._backup

    def update(*args):
        backup(*args)
        store = SQLiteStorageManager(str(values))
        try:
            store.set_field(
                "p1", "later" if mode == "insert" else "score", 41, table="parameters"
            )
        finally:
            store.close()

    monkeypatch.setattr(store_migration, "_backup", update)
    with pytest.raises(MigrationError, match="value store changed"):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert _version(index) == 0
    with closing(sqlite3.connect(values)) as db:
        assert (
            json.loads(
                db.execute(
                    "SELECT value_data FROM storage WHERE obj_uid='p1' AND field=?",
                    ("later" if mode == "insert" else "score",),
                ).fetchone()[0]
            )
            == 41
        )


def test_precommit_corruption_does_not_publish_catalog(tmp_path, monkeypatch):
    import h5py

    from diffract.core.data.metadata import store_migration

    index, cfg = _legacy(tmp_path, "hdf5")
    move = store_migration._move_values

    def corrupt(*args):
        move(*args)
        with h5py.File(tmp_path / "hdf5_storage.h5", "r+") as file:
            key = "root/parameters/weights/p1"
            file.move(key, "original_dataset")
            file.create_dataset(key, data=np.zeros((9, 9)))

    monkeypatch.setattr(store_migration, "_move_values", corrupt)
    with pytest.raises(ValueError, match="Verification failed"):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert _version(index) == MIGRATION_SENTINEL


def test_unsupported_zarr_codec_rejected_before_writing(tmp_path):
    from diffract.core.constants import STORAGE_ATTR_TYPE

    index, cfg = _legacy(tmp_path, "zarr")
    store = ZarrStorageManager(str(tmp_path / "zarr_storage.zarr"))
    try:
        store._get_root()["parameters/score/p1"].attrs[STORAGE_ATTR_TYPE] = (
            "unsupported"
        )
    finally:
        store.close()
    with pytest.raises(ValueError, match="Unsupported Zarr serialization"):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert _version(index) == 0
    assert not (tmp_path / "backup").exists()


def test_hybrid_heavy_sqlite_preserves_structured_array(tmp_path):
    index, cfg = _legacy(tmp_path, "hdf5")
    light = HDF5StorageManager(str(tmp_path / "hdf5_storage.h5"), swmr=False)
    heavy_path = tmp_path / "sqlite_storage.db"
    heavy = SQLiteStorageManager(str(heavy_path))
    record = np.array([(2, 1.25), (7, 2.5)], dtype=[("index", "i4"), ("score", "f8")])
    try:
        light.set_field("p1", "record", "__HEAVY__", table="parameters")
        heavy.set_field("p1", "record", record, table="parameters")
    finally:
        light.close()
        heavy.close()
    cfg.write_text(
        cfg.read_text().replace("[storage]\nbackend=hdf5", "[storage]\nbackend=hybrid")
        + f"\n[storage.sqlite]\npath={heavy_path}\n"
        "[storage.hybrid]\nlight=hdf5\nheavy=sqlite\n"
    )
    result = migrate_store(
        index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup"
    )
    assert result["values"] == 6
    with closing(sqlite3.connect(index)) as db:
        assert db.execute(
            "SELECT backend,size_bytes FROM field_values WHERE field='record'"
        ).fetchone() == ("heavy/sqlite", record.nbytes)
    with Session(config_path=cfg) as session:
        rows = session.results.export_metrics("record", export_format="dict")
        (stored,) = [
            r["fields"]["record"] for r in rows.values() if "record" in r["fields"]
        ]
        np.testing.assert_array_equal(stored, record)


@pytest.mark.parametrize("backend", ["hdf5", "zarr"])
def test_changed_file_store_before_sentinel_preserves_late_value(
    tmp_path, monkeypatch, backend
):
    from diffract.core.data.metadata import store_migration

    index, cfg = _legacy(tmp_path, backend)
    backup = store_migration._backup

    def update(*args):
        backup(*args)
        _, (spec,) = store_migration.resolve_store_layout(index, cfg)
        store = spec.open(readonly=False)
        try:
            store.set_field("p1", "late", 41, table="parameters")
        finally:
            store.close()

    monkeypatch.setattr(store_migration, "_backup", update)
    with pytest.raises(MigrationError, match="value store changed"):
        migrate_store(index, config=cfg, rename=RENAMES, backup_dir=tmp_path / "backup")
    assert _version(index) == 0
    _, (spec,) = store_migration.resolve_store_layout(index, cfg)
    store = spec.open(readonly=True)
    try:
        assert store.get_field("p1", "late", table="parameters") == 41
    finally:
        store.close()


def test_merged_coordinates_do_not_create_an_unknown_pair(tmp_path):
    index, cfg = _legacy(tmp_path)
    rename = RENAMES | {"models": dict.fromkeys(MODELS, "shared")}
    migrate_store(index, config=cfg, rename=rename, keep="newest", have_backup=True)
    with closing(sqlite3.connect(index)) as db:
        assert db.execute("SELECT COUNT(*) FROM aggregate_members").fetchone() == (1,)
        assert "direction_unknown" not in json.loads(
            db.execute("SELECT json_data FROM aggregates").fetchone()[0]
        )


def test_configured_tilde_matches_existing_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "~"
    directory.mkdir()
    index, cfg = _legacy(directory)
    cfg.write_text(cfg.read_text().replace(str(directory), "~"))
    assert (
        migrate_store(index, config=cfg, rename=RENAMES, have_backup=True)["values"]
        == 5
    )
    with Session(config_path=cfg) as session:
        assert len(session.results.export_metrics("score", export_format="dict")) == 2


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.parametrize("point", ["index", "value"])
def test_crashed_migration_requires_complete_restore_at_each_boundary(tmp_path, point):
    import zarr  # noqa: F401 - this verification cell requires the actual backend

    index, cfg = _legacy(tmp_path, "zarr")
    backup = tmp_path / "backup"
    program = """
import os, sys
from pathlib import Path
from diffract.core.data.metadata import store_migration as migration
from diffract.core.storage.zarr_manager import ZarrStorageManager
if sys.argv[1] == "index":
    original = migration._insert
    def stop_after_parameters(db, table, rows):
        original(db, table, rows)
        if table == "parameters":
            assert db.execute("select count(*) from parameters").fetchone() == (2,)
            os._exit(71)
    migration._insert = stop_after_parameters
else:
    original = ZarrStorageManager.move_blob
    def stop_after_one(self, old, new):
        original(self, old, new)
        os._exit(72)
    ZarrStorageManager.move_blob = stop_after_one
migration.migrate_store(Path(sys.argv[2]), config=Path(sys.argv[3]),
    backup_dir=Path(sys.argv[4]), rename={"parameters": {"weight#0":"weight.0"}})
"""
    crash = subprocess.run(  # noqa: S603 - fixed child process crash fixture
        [sys.executable, "-c", program, point, str(index), str(cfg), str(backup)],
        check=False,
    )
    assert crash.returncode == {"index": 71, "value": 72}[point]
    assert _version(index) == MIGRATION_SENTINEL

    def legacy_keys():
        store = ZarrStorageManager(str(tmp_path / "zarr_storage.zarr"), readonly=True)
        try:
            return {
                (table, name, uid)
                for table in ("parameters", "aggregates")
                for name in store.list_fields(table=table)
                for uid in store.list_objs_has_field(name, table=table)
            }
        finally:
            store.close()

    assert len(legacy_keys()) == {"index": 5, "value": 4}[point]
    with pytest.raises(MigrationError, match="restore"):
        migrate_store(index, config=cfg, have_backup=True)
    with (
        pytest.raises(IncompatibleStoreError, match="restore"),
        Session(config_path=cfg),
    ):
        pass
    if point == "value":
        with (
            closing(sqlite3.connect(backup / "metadata.db")) as source,
            closing(sqlite3.connect(index)) as target,
        ):
            source.backup(target)
        with pytest.raises(ValueError, match="Restore the metadata index"):
            migrate_store(index, config=cfg, have_backup=True, rename=RENAMES)
        assert _version(index) == 0
        assert len(legacy_keys()) == 4
    manifest = json.loads((backup / "restore.json").read_text())
    for entry, path in manifest.items():
        source, target = backup / entry, Path(path)
        if source.is_dir():
            shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    assert len(legacy_keys()) == 5
    assert _version(index) == 0
    with closing(sqlite3.connect(index)) as db:
        assert db.execute("select count(*) from parameters").fetchone() == (2,)
        assert db.execute("select count(*) from aggregates").fetchone() == (1,)
    report = migrate_store(index, config=cfg, have_backup=True, rename=RENAMES)
    assert report["values"] == 5
    with Session(config_path=cfg) as session:
        records = session.results.export_metrics(
            "weights", "score", export_format="dict"
        )
        assert len(records) == 2
        for record in records.values():
            expected = {MODELS[0]: 2, MODELS[1]: 10}[record["metadata"]["model_id"]]
            assert record["fields"]["score"] == expected
            np.testing.assert_array_equal(
                record["fields"]["weights"], np.eye(8) * expected
            )
        (pair,) = session.results.export_aggregates("l_overlap", export_format="list")
        np.testing.assert_array_equal(pair["value"], np.arange(3))
