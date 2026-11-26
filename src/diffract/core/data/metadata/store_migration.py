"""Offline upgrade for persistent Diffract stores with name-addressed values."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from diffract import __version__
from diffract.containers import _parse_ini_config, resolve_config_paths
from diffract.core.data.metadata.legacy import (
    Keep,
    LegacyPlan,
    metadata_signature,
    plan_legacy,
)
from diffract.core.data.metadata.migrations import (
    _INSTANCE_SCHEMA_VERSION,
    CURRENT_SCHEMA_VERSION,
    DIFFRACT_APPLICATION_ID,
    MIGRATION_SENTINEL,
    MIGRATIONS,
    MigrationError,
    SchemaVersionError,
    StoreIdentityError,
    _apply_step,
    verify_identity_layout,
    verify_store_identity,
)
from diffract.core.storage.catalog import OBJECT_KIND_BY_TABLE
from diffract.core.storage.legacy_values import (
    LegacyValue,
    StoreSpec,
    inventory,
    legacy_keys,
    local_file_versions,
    verify_values,
)
from diffract.core.storage.sqlite_manager import (
    external_array_paths,
)
from diffract.core.storage.zarr_manager import local_zarr_path

MIGRATION_CLI_ERRORS = (
    OSError,
    sqlite3.Error,
    ValueError,
    MigrationError,
    StoreIdentityError,
    SchemaVersionError,
)
_HYBRID_STORES = 2


def resolve_store_layout(
    path: Path, config: Path | None
) -> tuple[Path, list[StoreSpec]]:
    """Locate an existing metadata index and its configured value stores."""
    path = path.expanduser().resolve()
    index = path / "metadata_index.db" if path.is_dir() else path
    if not index.is_file():
        raise FileNotFoundError(index)
    if config is not None:
        config = config.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        tree = (
            _parse_ini_config(config)
            if config.suffix.lower() == ".ini"
            else json.loads(config.read_text())
        )
        resolve_config_paths(tree)
        metadata = tree.get("metadata", {})
        configured_index = metadata.get("sqlite", {}).get("path")
        if metadata.get("backend", "sqlite") != "sqlite" or (
            configured_index is not None and Path(configured_index).resolve() != index
        ):
            raise ValueError(
                "The configured metadata index differs from the supplied index."
            )
        storage = tree["storage"]
        root = Path.cwd()
    else:
        root = index.parent
        files = {
            "sqlite": "sqlite_storage.db",
            "hdf5": "hdf5_storage.h5",
            "zarr": "zarr_storage.zarr",
        }
        present = [name for name, file in files.items() if (root / file).exists()]
        if (
            not present
            or len(present) > _HYBRID_STORES
            or (len(present) == _HYBRID_STORES and "sqlite" not in present)
        ):
            raise ValueError(
                "Specify --config to identify the value store for this index."
            )
        storage = {name: {"path": files[name]} for name in present}
        if len(present) == _HYBRID_STORES:
            storage.update(
                backend="hybrid", hybrid={"light": "sqlite", "heavy": present[1]}
            )
        else:
            storage["backend"] = present[0]
    primary = storage["backend"]
    routes = (
        [("", primary)]
        if primary != "hybrid"
        else [(slot + "/", storage["hybrid"][slot]) for slot in ("light", "heavy")]
    )
    specs = []
    for prefix, backend in routes:
        if backend not in {"hdf5", "sqlite", "zarr"}:
            raise ValueError(f"Unsupported persistent backend {backend!r}")
        options = dict(storage[backend])
        location = options.pop("store_url", None) or options.pop("path")
        local = local_zarr_path(location) if backend == "zarr" else Path(location)
        if local is not None:
            location = str((root / local).resolve())
            if not Path(location).exists():
                raise FileNotFoundError(location)
        specs.append(StoreSpec(prefix + backend, backend, location, options))
    if len({spec.location for spec in specs} | {str(index)}) != len(specs) + 1:
        raise ValueError("The metadata and value stores must use distinct locations.")
    return index, specs


def _sqlite_backup(source: Path, dest: Path) -> None:
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader,
        closing(sqlite3.connect(dest)) as target,
    ):
        reader.backup(target)
        if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise MigrationError(f"SQLite backup did not pass verification: {dest}")


def _backup(root: Path, index: Path, specs: list[StoreSpec]) -> None:
    if any(s.remote for s in specs):
        raise ValueError(
            "Back up the remote object store and index, then pass --i-have-a-backup."
        )
    root = root.expanduser().resolve()
    if any(Path(s.location) == root or Path(s.location) in root.parents for s in specs):
        raise ValueError("The backup must be outside the value store.")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    _sqlite_backup(index, root / "metadata.db")
    manifest["metadata.db"] = str(index)
    for spec in specs:
        name = spec.name.replace("/", "-")
        path = Path(spec.location)
        dest = root / name
        if spec.backend == "sqlite":
            _sqlite_backup(path, dest)
            for recorded in external_array_paths(dest):
                source = path.parent / recorded
                relative = Path(name + "-arrays") / str(len(manifest))
                (root / relative).parent.mkdir(exist_ok=True)
                shutil.copy2(source, root / relative)
                manifest[str(relative)] = str(source)
        elif path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)
        manifest[name] = str(path)
    (root / "restore.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _changed_during_preflight() -> None:
    raise MigrationError(
        "The metadata index changed during preflight. Re-run the migration."
    )


def _stamp_sentinel(db: sqlite3.Connection, expected: int, signature: bytes) -> None:
    db.execute("BEGIN IMMEDIATE")
    try:
        if (
            db.execute("PRAGMA user_version").fetchone()[0] != expected
            or metadata_signature(db) != signature
        ):
            _changed_during_preflight()
        MIGRATIONS[0].apply(db)
        db.execute(
            "UPDATE schema_meta SET created_by='migration in progress' WHERE id=1"
        )
        db.execute(f"PRAGMA user_version = {MIGRATION_SENTINEL}")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _insert(db: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = tuple(rows[0])
    slots = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({slots})"  # nosec B608
    db.executemany(sql, ([row[column] for column in columns] for row in rows))


def _write_metadata(db: sqlite3.Connection, plan: LegacyPlan) -> None:
    db.execute("DROP TABLE IF EXISTS aggregates")
    db.execute("DROP TABLE IF EXISTS parameters")
    for step in MIGRATIONS[:_INSTANCE_SCHEMA_VERSION]:
        step.apply(db)
    created = int(time.time())
    _insert(
        db,
        "instances",
        [
            {"instance_uid": uid, "name": name, "created_at": created}
            for name, uid in plan.instances.items()
        ],
    )
    _insert(db, "parameters", plan.parameters)
    _insert(db, "aggregates", plan.aggregates)
    _insert(db, "aggregate_members", plan.members)
    db.executemany(
        "INSERT INTO uid_map(entity_kind,old_uid,new_uid) VALUES (?,?,?)",
        ((kind, old, new) for (kind, old), new in plan.uid_map.items()),
    )


def _move_values(specs: list[StoreSpec], values: list[LegacyValue]) -> None:
    for spec in specs:
        if spec.keeps_legacy_keys:
            continue
        store = spec.open(readonly=False)
        try:
            with store:
                for value in values:
                    if value.backend == spec.name:
                        store.move_blob(value.old_key, value.new_key)
        finally:
            store.close()


def _verify_identity(
    db: sqlite3.Connection, plan: LegacyPlan, values: list[LegacyValue]
) -> None:
    """Compare target natural keys, membership and value pointers to the input plan."""
    if list(db.execute("PRAGMA foreign_key_check")):
        raise MigrationError("Foreign-key verification failed")
    expected = {
        (r["uid"], r["instance_uid"], r["step"], r["name"], r["ptype"], r["json_data"])
        for r in plan.parameters
    }
    if (
        set(
            db.execute(
                "SELECT uid,instance_uid,step,name,ptype,json_data FROM parameters"
            )
        )
        != expected
    ):
        raise MigrationError("Parameter coordinate verification failed")
    if set(db.execute("SELECT name,instance_uid FROM instances")) != set(
        plan.instances.items()
    ):
        raise MigrationError("Model name verification failed")
    if set(db.execute("SELECT entity_kind,old_uid,new_uid FROM uid_map")) != {
        (kind, old, new) for (kind, old), new in plan.uid_map.items()
    }:
        raise MigrationError("Identity map verification failed")
    if set(db.execute("SELECT uid,field_name,json_data FROM aggregates")) != {
        (r["uid"], r["field_name"], r["json_data"]) for r in plan.aggregates
    }:
        raise MigrationError("Aggregate field verification failed")
    if set(
        db.execute(
            "SELECT aggregate_uid,member_uid,role,json_data FROM aggregate_members"
        )
    ) != {
        (r["aggregate_uid"], r["member_uid"], r["role"], r["json_data"])
        for r in plan.members
    }:
        raise MigrationError("Aggregate membership verification failed")
    if set(
        db.execute(
            "SELECT obj_kind,obj_uid,field,backend,storage_key,format_version "
            "FROM live_field_values"
        )
    ) != {
        (OBJECT_KIND_BY_TABLE[v.table], v.new_uid, v.field, v.backend, v.new_key, 0)
        for v in values
    }:
        raise MigrationError("Catalog verification failed")


def migrate_store(
    path: Path,
    *,
    config: Path | None = None,
    dry_run: bool = False,
    backup_dir: Path | None = None,
    have_backup: bool = False,
    keep: Keep = "abort",
    rename: dict | None = None,
    parse_step_suffix: bool = False,
) -> dict[str, Any]:
    """Inspect or upgrade a closed metadata index and its persistent value stores."""
    index = path.expanduser().resolve()
    if index.is_dir():
        index = index / "metadata_index.db"
    with closing(sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)) as reader:
        reader.execute("BEGIN")
        verify_store_identity(reader, path=str(index), allow_empty=False)
        (version,) = reader.execute("PRAGMA user_version").fetchone()
        if version == MIGRATION_SENTINEL:
            raise MigrationError(
                "Half-migrated store: restore the metadata and value stores "
                "from backup, then re-run the migration."
            )
        if version == CURRENT_SCHEMA_VERSION:
            verify_identity_layout(reader, path=str(index))
            (blocked,) = reader.execute(
                "SELECT values_require_migration FROM schema_meta WHERE id=1"
            ).fetchone()
            if blocked:
                raise MigrationError(
                    "This development index has already changed its identities. "
                    "Restore the original index and value store for migration."
                )
            return {"status": "current", "version": version}
        if version not in {0, 1}:
            raise MigrationError(f"Unsupported input schema version {version}")
        index, specs = resolve_store_layout(index, config)
        plan = plan_legacy(
            reader,
            keep=keep,
            rename={} if rename is None else rename,
            parse_step_suffix=parse_step_suffix,
        )
        if not plan.errors and not dry_run:
            signature = metadata_signature(reader)
    report = {
        "status": "planned",
        "source_version": version,
        "target_version": CURRENT_SCHEMA_VERSION,
        "instances": len(plan.instances),
        "parameters": len(plan.parameters),
        "aggregates": len(plan.aggregates),
        "uid_map": len(plan.uid_map),
        "memberships": len(plan.members),
        "duplicate_policy": keep,
        "parse_step_suffix": parse_step_suffix,
        "rename_map": rename or {},
        "duplicates": plan.conflicts,
        "errors": plan.errors,
    }
    if plan.errors or (plan.conflicts and keep == "abort"):
        report["status"] = "blocked"
        if dry_run:
            return report
        raise MigrationError(json.dumps(report, ensure_ascii=True))
    with ExitStack() as stack:
        versions = {}
        locked = {}
        if not dry_run:
            for spec in specs:
                if spec.backend == "sqlite":
                    conn = stack.enter_context(
                        closing(
                            sqlite3.connect(
                                Path(spec.location).as_uri() + "?mode=rw", uri=True
                            )
                        )
                    )
                    versions[spec.name] = conn.execute("PRAGMA data_version").fetchone()
                    locked[spec.name] = conn
        if not dry_run:
            nodes = legacy_keys(specs)
            initial_files = local_file_versions(specs)
        values = inventory(specs, plan.retained, plan.uid_map)
        report["values"] = len(values)
        counts = Counter(v.backend for v in values)
        report["retained_remote_keys"] = sum(
            counts[spec.name] for spec in specs if spec.remote
        )
        report["retained_local_keys"] = sum(
            counts[spec.name]
            for spec in specs
            if spec.keeps_legacy_keys and not spec.remote
        )
        report["legacy_numeric_scalars"] = sum(v.legacy_scalar for v in values)
        report["value_actions"] = [
            {
                "backend": spec.name,
                "action": "retain_key" if spec.keeps_legacy_keys else "rekey",
                "values": counts[spec.name],
            }
            for spec in specs
        ]
        if dry_run:
            return report
        if backup_dir is None and not have_backup:
            raise MigrationError(
                "Provide --backup-dir or --i-have-a-backup before migrating."
            )
        if backup_dir is not None:
            start = time.monotonic()
            _backup(backup_dir, index, specs)
            report["backup_seconds"] = time.monotonic() - start

        started = time.monotonic()
        with closing(sqlite3.connect(index)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            for name, writer in locked.items():
                writer.execute("BEGIN IMMEDIATE")
                if writer.execute("PRAGMA data_version").fetchone() != versions[name]:
                    raise MigrationError(
                        "A value store changed during preflight. Re-run the migration."
                    )
            if (
                local_file_versions(specs) != initial_files
                or legacy_keys(specs) != nodes
            ):
                raise MigrationError(
                    "A value store changed during preflight. Re-run the migration."
                )
            _stamp_sentinel(db, version, signature)
            db.execute("BEGIN IMMEDIATE")
            _write_metadata(db, plan)
            _move_values(specs, values)
            db.executemany(
                "INSERT INTO field_values(obj_uid,obj_kind,field,backend,"
                "storage_key,size_bytes,format_version) VALUES (?,?,?,?,?,?,0)",
                (
                    (
                        v.new_uid,
                        OBJECT_KIND_BY_TABLE[v.table],
                        v.field,
                        v.backend,
                        v.new_key,
                        v.size,
                    )
                    for v in values
                ),
            )
            moved = {
                (v.backend, v.table, v.field, v.old_uid)
                for v in values
                if v.old_key != v.new_key
            }
            if legacy_keys(specs) != nodes - moved:
                raise MigrationError(
                    "Legacy keys changed during migration. Restore the complete backup."
                )
            verify_values(specs, values)
            _verify_identity(db, plan, values)
            for step in MIGRATIONS[_INSTANCE_SCHEMA_VERSION:]:
                _apply_step(db, step)
            db.execute(
                "UPDATE schema_meta SET created_by=? WHERE id=1",
                (f"diffract-core {__version__}",),
            )
            db.execute(f"PRAGMA application_id={DIFFRACT_APPLICATION_ID}")
            db.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION}")
            db.commit()
        return report | {
            "status": "migrated",
            "transform_seconds": time.monotonic() - started,
        }
