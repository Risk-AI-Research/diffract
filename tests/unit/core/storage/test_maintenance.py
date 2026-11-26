"""Exercise catalog checks and collection against on-disk stores."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from diffract import Session
from diffract.core.data.metadata.store_migration import resolve_store_layout
from diffract.core.storage.keys import value_key
from diffract.core.storage.maintenance import check_store, collect_store

pytestmark = pytest.mark.unit


def _seed(
    tmp_path: Path, backend: str, *, external_arrays: bool = False
) -> tuple[Path, Path, str]:
    if backend == "zarr":
        pytest.importorskip("zarr")
    index = tmp_path / "metadata_index.db"
    config = tmp_path / "maintenance.ini"
    config.write_text(
        f"""
[metadata]
backend = "sqlite"
[metadata.sqlite]
path = {json.dumps(str(index))}
[storage]
backend = "{backend}"
[storage.sqlite]
path = {json.dumps(str(tmp_path / "sqlite_storage.db"))}
array_threshold = {1 if external_arrays else 1_048_576}
[storage.hdf5]
path = {json.dumps(str(tmp_path / "hdf5_storage.h5"))}
[storage.zarr]
store_url = {json.dumps(str(tmp_path / "zarr_storage.zarr"))}
[storage.hybrid]
light = "sqlite"
heavy = "hdf5"
array_threshold = 1
[cache]
backend = "none"
[nn.extractor]
skip_not_implemented_types = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with Session(config_path=config) as session:
        session.models.add({"weight": np.eye(2)}, model_id="net")
        session.compute.apply("frob_norm")
        (row,) = session.models.parameters.list()
    return index, config, row["uid"]


@pytest.mark.parametrize("backend", ["sqlite", "hdf5", "zarr", "hybrid"])
def test_gc_dry_run_matches_actual_collection(tmp_path, backend):
    index, config, uid = _seed(tmp_path, backend)
    assert not any(check_store(index, config=config).values())
    with Session(config_path=config) as session:
        session.results.erase("frob_norm")
    _, specs = resolve_store_layout(index, config)
    orphan = value_key(uid, "orphan")
    with closing(specs[0].open(readonly=False)) as physical:
        physical.put_blob(orphan, 17)

    report = check_store(index, config=config)
    assert report["orphan_blobs"] == [{"backend": specs[0].name, "storage_key": orphan}]
    assert not report["missing_blobs"]
    sidecar = index.with_name(index.name + ".gc.json")
    assert collect_store(index, config=config, dry_run=True) == {
        "tombstones": [],
        "blobs": [],
    }
    assert not sidecar.exists()
    assert collect_store(index, config=config) == {"tombstones": [], "blobs": []}
    assert sidecar.exists()
    planned = collect_store(index, config=config, dry_run=True, grace_days=0)
    assert len(planned["tombstones"]) == 1
    assert len(planned["blobs"]) == 2
    assert report == check_store(index, config=config)
    assert collect_store(index, config=config, grace_days=0) == planned
    assert not sidecar.exists()
    assert not any(check_store(index, config=config).values())
    with Session(config_path=config) as session:
        session.compute.apply("frob_norm")
        row = session.results.export_metrics("frob_norm", export_format="dict")
        assert row[uid]["fields"]["frob_norm"] == pytest.approx(np.sqrt(2))


def test_fsck_reports_broken_pointer_owner_and_run(tmp_path):
    index, config, uid = _seed(tmp_path, "sqlite")
    _, (spec,) = resolve_store_layout(index, config)
    with closing(spec.open(readonly=False)) as physical:
        physical.delete_blob(value_key(uid, "weights"))
    with closing(sqlite3.connect(index)) as db, db:
        db.execute(
            "UPDATE runs SET status='FAILED', config_hash='broken' "
            "WHERE kernel_name='frob_norm'"
        )
        db.execute("UPDATE field_values SET built_at_seq=999 WHERE field='frob_norm'")
        owner_fields = {
            field
            for (field,) in db.execute(
                "SELECT field FROM live_field_values WHERE obj_uid=?", (uid,)
            )
        }
        db.execute("DELETE FROM parameters WHERE uid=?", (uid,))

    report = check_store(index, config=config)
    assert {r["field"] for r in report["missing_blobs"]} == {"weights"}
    assert {r["field"] for r in report["unknown_owners"]} == owner_fields
    assert {r["field"] for r in report["incomplete_runs"]} == {"frob_norm"}
    assert {r["field"] for r in report["invalid_generations"]} == {"frob_norm"}
    assert len(report["invalid_run_configs"]) == 1
    with pytest.raises(ValueError, match="missing catalog data"):
        collect_store(index, config=config, grace_days=0)


def test_gc_keeps_a_resurrected_result_and_ages_orphans(tmp_path):
    index, config, uid = _seed(tmp_path, "sqlite")
    with Session(config_path=config) as session:
        session.results.erase("frob_norm")
        session.compute.apply("frob_norm")
    _, (spec,) = resolve_store_layout(index, config)
    orphan = value_key(uid, "orphan")
    with closing(spec.open(readonly=False)) as physical:
        physical.put_blob(orphan, 13)
    assert not collect_store(index, config=config)["blobs"]

    sidecar = index.with_name(index.name + ".gc.json")
    old = datetime.now(UTC) - timedelta(days=15)
    sidecar.write_text(
        json.dumps([[spec.name, orphan, old.isoformat()]]), encoding="utf-8"
    )
    plan = collect_store(index, config=config, dry_run=True)
    assert plan == {
        "tombstones": [],
        "blobs": [{"backend": spec.name, "storage_key": orphan}],
    }
    assert collect_store(index, config=config) == plan
    assert not any(check_store(index, config=config).values())
    with Session(config_path=config) as session:
        row = session.results.export_metrics("frob_norm", export_format="dict")
        assert row[uid]["fields"]["frob_norm"] == pytest.approx(np.sqrt(2))


def test_fsck_detects_a_missing_external_sqlite_value(tmp_path):
    index, config, uid = _seed(tmp_path, "sqlite", external_arrays=True)
    assert not any(check_store(index, config=config).values())
    with closing(sqlite3.connect(tmp_path / "sqlite_storage.db")) as values:
        (path,) = values.execute(
            "SELECT file_path FROM storage WHERE field='weights' AND obj_uid=?", (uid,)
        ).fetchone()
    Path(path).unlink()
    report = check_store(index, config=config)
    assert [row["storage_key"] for row in report["missing_blobs"]] == [
        value_key(uid, "weights")
    ]


_CRASH_AFTER_BLOBS = """
import os
import signal
import sys
from diffract import Session
from diffract.core.storage.sqlite_manager import SQLiteStorageManager

original_put = SQLiteStorageManager.put_blob
original_exit = SQLiteStorageManager.__exit__

def put(self, key, value):
    original_put(self, key, value)
    if key.endswith('/frob_norm'):
        self._crash_when_committed = True

def exit_after_commit(self, exc_type, exc_val, traceback):
    original_exit(self, exc_type, exc_val, traceback)
    if exc_type is None and getattr(self, '_crash_when_committed', False):
        if os.name == 'nt':
            os._exit(79)
        os.kill(os.getpid(), signal.SIGKILL)

with Session(config_path=sys.argv[1]) as session:
    SQLiteStorageManager.put_blob = put
    SQLiteStorageManager.__exit__ = exit_after_commit
    session.compute.apply("frob_norm")
"""


def test_process_death_between_blob_and_index_commits_heals_via_recompute(tmp_path):
    index, config, uid = _seed(tmp_path, "sqlite")
    with Session(config_path=config) as session:
        session.results.erase("frob_norm")
    collect_store(index, config=config, grace_days=0)
    crashed = subprocess.run(  # noqa: S603 - fixed fixture program
        [sys.executable, "-c", _CRASH_AFTER_BLOBS, str(config)],
        capture_output=True,
        check=False,
        timeout=45,
    )
    assert crashed.returncode == (79 if os.name == "nt" else -signal.SIGKILL), (
        crashed.stderr
    )
    report = check_store(index, config=config)
    assert not report["missing_blobs"]
    assert report["orphan_blobs"] == [
        {"backend": "sqlite", "storage_key": value_key(uid, "frob_norm")}
    ]
    with Session(config_path=config) as session:
        session.compute.apply("frob_norm")
        row = session.results.export_metrics("frob_norm", export_format="dict")
        assert row[uid]["fields"]["frob_norm"] == pytest.approx(np.sqrt(2))
    checked = subprocess.run(  # noqa: S603 - fixed module and test store
        [
            sys.executable,
            "-m",
            "diffract.store",
            "fsck",
            str(index),
            "--config",
            str(config),
        ],
        capture_output=True,
        check=True,
        timeout=45,
    )
    assert not any(json.loads(checked.stdout).values())
