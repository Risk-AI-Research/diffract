"""Physical value rekeying and recovery at the catalog commit boundary."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from diffract.core.data.metadata.migrations import (
    DIFFRACT_APPLICATION_ID,
    MIGRATIONS,
    SchemaVersionError,
    run_migrations,
    upgrade_metadata_index,
)
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.storage.catalog import ValuePointer
from diffract.core.storage.catalog_manager import (
    CatalogStorageManager,
)
from diffract.core.storage.hdf5_manager import HDF5StorageManager
from diffract.core.storage.hybrid_manager import HybridStorageManager
from diffract.core.storage.keys import value_key
from diffract.core.storage.ram_manager import RAMStorageManager
from diffract.core.storage.sqlite_manager import SQLiteStorageManager
from diffract.core.storage.zarr_manager import ZarrStorageManager

pytestmark = pytest.mark.unit
UID = "2" * 32
DEATH_EXIT = 73


def _seed_v2_pointer(path: Path, field: str, pointer: ValuePointer) -> None:
    """Create the pre-provenance catalog row and use the normal upgrade."""
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        run_migrations(db, steps=MIGRATIONS[:2])
        db.execute(f"PRAGMA application_id={DIFFRACT_APPLICATION_ID}")
        db.execute(
            "INSERT INTO field_values "
            "(obj_uid,obj_kind,field,backend,storage_key,size_bytes,format_version) "
            "VALUES (?,'parameter',?,?,?,?,0)",
            (UID, field, pointer.backend, pointer.storage_key, pointer.size_bytes),
        )
    assert upgrade_metadata_index(path) == [3]


@pytest.mark.parametrize("backend", ["ram", "sqlite", "hdf5", "zarr"])
def test_native_rekey_and_collision(tmp_path: Path, backend: str) -> None:
    """Rekey the actual legacy blob, keeping both sides of a collision intact."""
    if backend == "zarr":
        pytest.importorskip("zarr")
    target = value_key(UID, "weights")
    source = "parameters/weights/legacy"
    path = tmp_path / backend
    store = {
        "ram": RAMStorageManager,
        "sqlite": lambda: SQLiteStorageManager(str(path)),
        "hdf5": lambda: HDF5StorageManager(str(path)),
        "zarr": lambda: ZarrStorageManager(str(path)),
    }[backend]()
    weights = np.arange(64 * 1024, dtype="float32").reshape(256, 256)
    try:
        store.set_field("legacy", "weights", weights, table="parameters")
        store.put_blob(target, b"other")
        with pytest.raises(FileExistsError):
            store.move_blob(source, target)
        assert store.read_blob(target) == b"other"
        np.testing.assert_array_equal(store.read_blob(source), weights)
        store.delete_blob(target)
        if backend == "zarr":
            inode = (path / "root" / source).stat().st_ino
        if backend == "hdf5":
            import h5py

            with h5py.File(path, "r") as file:
                address = h5py.h5o.get_info(file["root/" + source].id).addr
        store.move_blob(source, target)
        store.move_blob(target, target)
        np.testing.assert_array_equal(store.read_blob(target), weights)
        assert not store.has_field("legacy", "weights", table="parameters")
        if backend == "zarr":
            assert (path / "root" / target).stat().st_ino == inode
        if backend == "hdf5":
            with h5py.File(path, "r") as file:
                assert h5py.h5o.get_info(file["root/" + target].id).addr == address
        with pytest.raises(KeyError):
            store.move_blob(source, target)
    finally:
        if hasattr(store, "close"):
            store.close()


def test_hybrid_reads_retained_key_and_failed_replacement(tmp_path: Path) -> None:
    """Pointers select the original tier and node even after routing changes."""
    meta = tmp_path / "metadata.db"
    kept = ValuePointer("heavy/hdf5", "parameters/field/legacy", format_version=0)
    _seed_v2_pointer(meta, "field", kept)
    with closing(SQLiteMetadataIndex(str(meta))) as index:
        light = SQLiteStorageManager(str(tmp_path / "light.db"))
        heavy = HDF5StorageManager(str(tmp_path / "heavy.h5"))
        with closing(
            CatalogStorageManager(HybridStorageManager(light, heavy), index)
        ) as store:
            heavy.set_field("legacy", "field", b"original", table="parameters")
            assert store.get_field(UID, "field", table="parameters") == b"original"
            key = value_key(UID, "field")
            heavy.put_blob(key, b"previous")
            with pytest.raises(ValueError):
                heavy.put_blob(key, object())
            assert heavy.read_blob(key) == b"previous"


def test_initial_field_write_discards_failed_binding(tmp_path: Path) -> None:
    """A caught HDF5 failure does not commit an address or replace a live value."""
    table = "parameters"
    with (
        closing(SQLiteMetadataIndex(str(tmp_path / "metadata.db"))) as index,
        closing(
            CatalogStorageManager(
                HDF5StorageManager(str(tmp_path / "values.h5")), index
            )
        ) as store,
    ):
        with store, pytest.raises(ValueError):
            store.publish_initial_field(UID, "weights", object(), table=table)
        assert not index.query_rows("field_values")

        value = np.arange(6).reshape(2, 3)
        store.publish_initial_field(UID, "weights", value, table=table)
        with store, pytest.raises(ValueError):
            store.publish_initial_field(UID, "weights", object(), table=table)
        np.testing.assert_array_equal(
            store.get_field(UID, "weights", table=table), value
        )


@pytest.mark.parametrize("table", ["parameters", "aggregates"])
def test_legacy_catalog_requires_original_value_store(
    tmp_path: Path, table: str
) -> None:
    """A metadata-only upgrade cannot present pre-catalog data as empty."""
    meta = tmp_path / "metadata.db"
    with closing(sqlite3.connect(meta, isolation_level=None)) as db:
        run_migrations(db, steps=MIGRATIONS[:1])
        db.execute(
            "CREATE TABLE parameters(uid TEXT PRIMARY KEY,name TEXT,model_id TEXT,"
            "ptype TEXT,json_data TEXT)"
        )
        db.execute(
            "CREATE TABLE aggregates(uid TEXT PRIMARY KEY,field_name TEXT,"
            "context_models TEXT,context_params TEXT,json_data TEXT)"
        )
        db.execute(f"INSERT INTO {table}(uid,json_data) VALUES(?, '{{}}')", (UID,))  # nosec B608
    with closing(SQLiteStorageManager(str(tmp_path / "values.db"))) as raw:
        raw.set_field(UID, "weights", b"original", table=table)
        with pytest.raises(SchemaVersionError, match="guarded"):
            upgrade_metadata_index(meta)
        with (
            closing(SQLiteMetadataIndex(str(meta))) as index,
            pytest.raises(SchemaVersionError),
        ):
            CatalogStorageManager(raw, index)
        assert raw.get_field(UID, "weights", table=table) == b"original"


def test_process_crash_between_value_and_catalog_commit(tmp_path: Path) -> None:
    """First publication interrupted after the blob commit leaves a retryable orphan."""
    meta = tmp_path / "metadata.db"
    data = tmp_path / "values.db"
    code = textwrap.dedent(
        """
        import os, sys
        from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
        from diffract.core.storage.catalog_manager import CatalogStorageManager
        from diffract.core.storage.sqlite_manager import SQLiteStorageManager

        class ExitAfterCommit(SQLiteStorageManager):
            def __exit__(self, *args):
                super().__exit__(*args)
                if not self._context_depth and args[0] is None:
                    os._exit(int(sys.argv[4]))

        index = SQLiteMetadataIndex(sys.argv[1])
        store = CatalogStorageManager(ExitAfterCommit(sys.argv[2]), index)
        store.set_field(sys.argv[3], "payload", b"old", table="parameters")
        """
    )
    child = subprocess.run(  # noqa: S603 - fixed test program in an isolated store
        [sys.executable, "-c", code, str(meta), str(data), UID, str(DEATH_EXIT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[4] / "src"),
        },
    )
    assert child.returncode == DEATH_EXIT, child.stderr
    with closing(SQLiteMetadataIndex(str(meta))) as index:
        raw = SQLiteStorageManager(str(data))
        with closing(CatalogStorageManager(raw, index)) as store:
            assert raw.read_blob(value_key(UID, "payload")) == b"old"
            assert store.list_fields(UID, table="parameters") == []
            store.set_field(UID, "payload", b"recovered", table="parameters")
            assert store.get_field(UID, "payload", table="parameters") == b"recovered"


def test_remote_zarr_retains_legacy_node(tmp_path: Path) -> None:
    """A non-local fsspec store reads a retained node through its catalog pointer."""
    pytest.importorskip("zarr")
    from uuid import uuid4

    raw = ZarrStorageManager(f"memory://catalog-{uuid4().hex}/store")
    source = "parameters/weights/legacy"
    meta = tmp_path / "metadata.db"
    _seed_v2_pointer(meta, "weights", ValuePointer("zarr", source, format_version=0))
    with (
        closing(SQLiteMetadataIndex(str(meta))) as index,
        closing(CatalogStorageManager(raw, index)) as store,
    ):
        raw.set_field("legacy", "weights", b"kept", table="parameters")
        with pytest.raises(NotImplementedError, match="retain"):
            raw.move_blob(source, value_key(UID, "weights"))
        assert store.get_field(UID, "weights", table="parameters") == b"kept"
        assert raw.has_field("legacy", "weights", table="parameters")
    with closing(SQLiteMetadataIndex(str(tmp_path / "metadata.db"))) as reopened:
        assert reopened.query_rows(
            "field_values", columns=("format_version", "storage_key")
        ) == [{"format_version": 0, "storage_key": source}]


@pytest.mark.parametrize(
    "mid_write", [False, True], ids=["after-flush", "during-write"]
)
def test_hdf5_process_loss_and_closed_backup(tmp_path: Path, mid_write: bool) -> None:
    """Check pointer visibility and recover even if the HDF5 container cannot reopen."""
    import h5py

    meta = tmp_path / "index.db"
    payload = tmp_path / "values.h5"
    backup = tmp_path / "values.backup.h5"
    with (
        closing(SQLiteMetadataIndex(str(meta))) as index,
        closing(
            CatalogStorageManager(HDF5StorageManager(str(payload)), index)
        ) as store,
    ):
        store.set_field(UID, "previous", b"committed", table="parameters")
    shutil.copyfile(payload, backup)
    code = textwrap.dedent(
        """
        import os, sys
        import numpy as np
        from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
        from diffract.core.storage.catalog_manager import CatalogStorageManager
        from diffract.core.storage.hdf5_manager import HDF5StorageManager

        mid_write = sys.argv[4] == "1"
        exit_code = int(sys.argv[5])

        class Interrupted(HDF5StorageManager):
            def _write_ndarray_dataset(self, *args):
                super()._write_ndarray_dataset(*args)
                if mid_write:
                    os._exit(exit_code)

            def put_blob(self, key, value):
                super().put_blob(key, value)
                if mid_write:
                    raise AssertionError("Expected interruption inside the writer")
                os._exit(exit_code)

        index = SQLiteMetadataIndex(sys.argv[1])
        backend = Interrupted(sys.argv[2], keep_file_open=False)
        store = CatalogStorageManager(backend, index)
        store.set_field(sys.argv[3], "next", np.arange(32), table="parameters")
        """
    )
    child = subprocess.run(  # noqa: S603 - fixed test program on a temporary HDF5 file
        [
            sys.executable,
            "-c",
            code,
            str(meta),
            str(payload),
            UID,
            str(int(mid_write)),
            str(DEATH_EXIT),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == DEATH_EXIT, child.stderr
    with closing(SQLiteMetadataIndex(str(meta))) as index:
        assert not index.query_rows("live_field_values", where={"field": "next"})
        if mid_write:
            # A killed HDF5 writer can leave a superblock write flag or corrupt
            # the entire file. Restore the closed backup before retrying.
            shutil.copyfile(backup, payload)
        else:
            with h5py.File(payload, "r") as file:
                np.testing.assert_array_equal(
                    file[f"root/{UID}/next"][:], np.arange(32)
                )
        with closing(
            CatalogStorageManager(HDF5StorageManager(str(payload)), index)
        ) as store:
            assert store.get_field(UID, "previous", table="parameters") == b"committed"
            store.set_field(UID, "next", np.arange(32), table="parameters")
            np.testing.assert_array_equal(
                store.get_field(UID, "next", table="parameters"), np.arange(32)
            )


@pytest.mark.parametrize("legacy", [False, True], ids=["current", "migrated"])
@pytest.mark.parametrize("during", ["before", "after"])
def test_interrupted_overwrite_matches_durable_array(
    tmp_path: Path, during: str, legacy: bool
) -> None:
    """An in-place replacement recovers the descriptor for the surviving array."""
    meta, path = tmp_path / "index.db", tmp_path / "values.db"
    initial = np.arange(3, dtype="int16")
    if legacy:
        key = value_key(UID, "weights")
        with closing(SQLiteStorageManager(str(path))) as raw:
            raw.put_blob(key, initial)
        _seed_v2_pointer(
            meta, "weights", ValuePointer("sqlite", key, initial.nbytes, 0)
        )
    else:
        with (
            closing(SQLiteMetadataIndex(str(meta))) as index,
            closing(
                CatalogStorageManager(SQLiteStorageManager(str(path)), index)
            ) as store,
        ):
            store.set_field(UID, "weights", initial, table="parameters")
    code = textwrap.dedent(
        """
        import sys, os, numpy as np
        from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
        from diffract.core.storage.catalog_manager import CatalogStorageManager
        from diffract.core.storage.sqlite_manager import SQLiteStorageManager
        class Interrupted(SQLiteStorageManager):
            def put_blob(self, key, value):
                if sys.argv[4] == "before":
                    os._exit(int(sys.argv[5]))
                super().put_blob(key, value)
            def __exit__(self,*args):
                super().__exit__(*args)
                if not self._context_depth and args[0] is None:
                    os._exit(int(sys.argv[5])+1)
        index = SQLiteMetadataIndex(sys.argv[1])
        store = CatalogStorageManager(Interrupted(sys.argv[2]), index)
        new = np.arange(5, dtype="int64")
        if sys.argv[6] == "legacy":
            new = np.arange(3, dtype="int16") + 10
        store.set_field(sys.argv[3], "weights", new, table="parameters")
        """
    )
    result = subprocess.run(  # noqa: S603 - isolated subprocess with fixed inputs
        [
            sys.executable,
            "-c",
            code,
            str(meta),
            str(path),
            UID,
            during,
            str(DEATH_EXIT),
            "legacy" if legacy else "current",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == DEATH_EXIT + int(during == "after"), result.stderr
    expected = (
        np.arange(5, dtype="int64")
        if during == "after"
        else np.arange(3, dtype="int16")
    )
    if during == "after" and legacy:
        expected = np.arange(3, dtype="int16") + 10
    with (
        closing(SQLiteMetadataIndex(str(meta))) as reopened,
        closing(
            CatalogStorageManager(SQLiteStorageManager(str(path)), reopened)
        ) as store,
    ):
        pointer = store.catalog.get("parameter", UID, "weights")
        np.testing.assert_array_equal(
            store.get_field(UID, "weights", table="parameters"), expected
        )
        assert pointer.size_bytes == expected.nbytes
        assert pointer.format_version == int(not legacy or during == "after")
        if legacy and during == "before":
            assert pointer.generation is None
        else:
            assert pointer.generation is not None
            assert reopened.find_row(
                "runs", columns=("status",), where={"run_id": pointer.generation.run_id}
            ) == {"status": "START"}
    assert not list(Path(str(meta) + ".pending").glob("*.json"))


def test_sqlite_native_addresses_have_no_table_segment(tmp_path: Path) -> None:
    """Inspect both entity kinds directly in the SQL value table."""
    blob = tmp_path / "values.db"
    aggregate = "f" * 64
    field = "metric@axis[x]"
    with (
        closing(SQLiteMetadataIndex(str(tmp_path / "index.db"))) as index,
        closing(CatalogStorageManager(SQLiteStorageManager(str(blob)), index)) as store,
    ):
        store.set_field(UID, field, 1, table="parameters")
        store.set_field(aggregate, field, 2, table="aggregates")
    with closing(sqlite3.connect(blob)) as raw:
        assert raw.execute(
            "SELECT table_name,field,obj_uid FROM storage ORDER BY obj_uid"
        ).fetchall() == [
            ("", "metric%40axis%5Bx%5D", UID),
            ("", "metric%40axis%5Bx%5D", aggregate),
        ]
