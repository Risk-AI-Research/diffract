"""Exercise identical workflows in the released and migrated session implementations."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from diffract.migrate import migrate_store

pytestmark = [pytest.mark.integration, pytest.mark.stress]

# Public source of the v0.3.0 release. An installed reference venv also works.
REFERENCE_COMMIT = "0f151dcc8eccb734947b32211829d825ee46bc64"
ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = Path(__file__).with_name("migration_workload.py")


def _reference(tmp_path):
    if executable := os.environ.get("DIFFRACT_V030_PYTHON"):
        return executable, ""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "Install a 0.3.0 reference with DIFFRACT_V030_PYTHON or provide git."
        )
    archive = subprocess.check_output(  # noqa: S603 - pinned release content
        [git, "archive", REFERENCE_COMMIT, "src/diffract"], cwd=ROOT
    )
    destination = tmp_path / "reference"
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")
    return sys.executable, str(destination / "src")


def _run(directory, executable, source, action, version):
    directory.mkdir(exist_ok=True)
    subprocess.run(  # noqa: S603 - local test workload and configured interpreter
        [executable, str(WORKLOAD), action, version],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": source, "TI_SKIP_VERSION_CHECK": "ON"},
        check=True,
    )
    if not action.startswith("edge-"):
        return json.loads((directory / f"{action}.json").read_text())
    return None


def _key(key):
    return tuple(tuple(item) if isinstance(item, list) else item for item in key)


def _array(item):
    return np.frombuffer(base64.b64decode(item["bytes"]), dtype=item["dtype"]).reshape(
        item["shape"]
    )


def _coordinates(rows):
    result = {_key(key): value for key, value in rows}
    assert len(result) == len(rows)
    return result


def _compare(before, after, *, fresh):
    for category in ("metrics", "viz"):
        old, new = (_coordinates(snapshot[category]) for snapshot in (before, after))
        assert old.keys() == new.keys()
        assert len(old) == {"metrics": 18 if fresh else 12, "viz": 4}[category]
        for coordinate, expected in old.items():
            recomputed = category == "metrics" and (
                coordinate[-1] == "weights_svals" or coordinate[0] == "donor"
            )
            if fresh and recomputed:
                assert new[coordinate]["dtype"] == expected["dtype"]
                assert new[coordinate]["shape"] == expected["shape"]
                np.testing.assert_allclose(
                    _array(expected), _array(new[coordinate]), rtol=1e-6, atol=1e-7
                )
            else:
                assert expected == new[coordinate], coordinate

    legacy = {_key(row["key"]): row["value"] for row in before["aggregates"]}
    assert len(legacy) == len(before["aggregates"]) == (6 if fresh else 4)
    current = defaultdict(list)
    for row in after["aggregates"]:
        current[_key(row["key"])].append(row)
    assert current.keys() == legacy.keys()
    for coord, expected in legacy.items():
        rows = current[coord]
        expected_members = {
            (model, 0, parameter) for model in coord[1] for parameter in coord[2]
        }
        for row in rows:
            assert len(row["members"]) == len(expected_members)
            assert {tuple(member) for member in row["members"]} == expected_members
        if fresh and coord[:2] == ("l_overlap", ("alpha", "donor")):
            assert {tuple(row["roles"]) for row in rows} == {
                ("donor", "alpha"),
                ("alpha", "donor"),
            }
            assert len(rows) == 2
            metrics = _coordinates(after["metrics"])
            bases = {}
            overlap_atol = 0.0
            for model in coord[1]:
                weight = _array(metrics[model, 0, coord[2][0], "weights"])
                u, singular, _ = np.linalg.svd(
                    weight.astype(np.float64), full_matrices=False
                )
                bases[model] = u[:, np.argsort(singular)]
                # A singular vector's sensitivity scales with the inverse spectral gap.
                overlap_atol += (
                    np.finfo(weight.dtype).eps * singular[0] / min(-np.diff(singular))
                )
            for row in rows:
                assert not row["unknown"]
                assert row["steps"] == [0, 0]
                assert [member[0] for member in row["members"]] == row["roles"]
                assert row["value"]["dtype"] == expected["dtype"]
                assert row["value"]["shape"] == expected["shape"]
                left, right = row["roles"]
                np.testing.assert_allclose(
                    _array(row["value"]),
                    np.abs(bases[left].T @ bases[right]),
                    rtol=0,
                    atol=overlap_atol,
                )
            a, b = (_array(row["value"]) for row in rows)
            np.testing.assert_allclose(a, b.T, rtol=1e-6, atol=1e-7)
            assert not np.allclose(a, a.T)
            assert any(
                np.allclose(_array(expected), x, rtol=0, atol=overlap_atol)
                for x in (a, b)
            )
        else:
            (row,) = rows
            assert row["value"] == expected, coord
            if coord[0] == "l_overlap":
                assert row["unknown"]
                assert row["roles"] == [None, None]
                assert row["steps"] == [None, None]


def test_migrated_session_matches_the_released_workload(tmp_path):
    executable, reference = _reference(tmp_path)
    original = tmp_path / "original"
    migrated = tmp_path / "migrated"
    before = _run(original, executable, reference, "seed", "legacy")
    shutil.copytree(original, migrated)
    migrate_store(migrated / ".diffract" / "sqlite", backup_dir=tmp_path / "backup")
    current = str(ROOT / "src")
    after = _run(migrated, sys.executable, current, "inspect", "current")
    _compare(before, after, fresh=False)
    before = _run(original, executable, reference, "exercise", "legacy")
    after = _run(migrated, sys.executable, current, "exercise", "current")
    _compare(before, after, fresh=True)


def test_apply_verifies_inherited_spectra_and_reuses_the_recorded_runs(tmp_path):
    from diffract import Session

    executable, reference = _reference(tmp_path)
    original = tmp_path / "source"
    _run(original, executable, reference, "seed", "legacy")
    directory = original / ".diffract" / "sqlite"
    migrate_store(directory, backup_dir=tmp_path / "backup")

    with (
        closing(sqlite3.connect(directory / "metadata_index.db")) as index,
        pytest.MonkeyPatch.context() as patch,
    ):
        query = (
            "SELECT obj_kind,obj_uid,field,backend,storage_key,revision,format_version "
            "FROM live_field_values ORDER BY obj_kind,obj_uid,field"
        )
        before = index.execute(query).fetchall()
        patch.chdir(original)
        with Session(profile="local") as session:
            session.compute.configure_kernel("weights_svd", allow_cuda=False)
            inherited = session.results.status(
                "param_norm", "frob_norm", "weights_svals"
            )
            assert len(inherited) == 10
            assert {(row.field, row.status, row.reason) for row in inherited} == {
                ("weights_svals", "stale", "spectral execution settings are unknown"),
                ("param_norm", "fresh", "pre-provenance"),
                ("frob_norm", "fresh", "pre-provenance"),
            }
            assert index.execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)
            session.compute.apply("param_norm")
            recorded = index.execute(
                "SELECT COUNT(*) FROM runs WHERE status='COMPLETE'"
            ).fetchone()
            verified = index.execute(query).fetchall()
            session.compute.apply("param_norm")
            assert {
                (row.status, row.reason)
                for row in session.results.status(
                    "param_norm", "frob_norm", "weights_svals"
                )
            } == {("fresh", None)}

        assert recorded == (4,)
        assert index.execute("SELECT COUNT(*) FROM runs").fetchone() == recorded
        assert index.execute(query).fetchall() == verified
        assert [r for r in verified if r[2] == "weights"] == [
            r for r in before if r[2] == "weights"
        ]
        old_spectra = {r[1]: r[5] for r in before if r[2] == "weights_svals"}
        new_spectra = {r[1]: r[5] for r in verified if r[2] == "weights_svals"}
        assert len(old_spectra) == 4
        assert new_spectra.keys() == old_spectra.keys()
        assert all(
            new_spectra[uid] != revision for uid, revision in old_spectra.items()
        )
        assert index.execute(
            "SELECT r.kernel_name, COUNT(*) FROM live_field_values f JOIN runs r "
            "USING(run_id) WHERE f.built_at_seq IS NOT NULL "
            "AND f.changed_at_seq IS NOT NULL "
            "GROUP BY r.kernel_name ORDER BY r.kernel_name"
        ).fetchall() == [("frob_norm", 4), ("param_norm", 2), ("weights_svd", 12)]


@pytest.mark.parametrize(
    "case", ["empty", "ingested", "orphan", "duplicate", "reserved", "partial"]
)
def test_generated_released_stores_preserve_or_report_edge_cases(
    tmp_path, case, monkeypatch
):
    import sqlite3
    from contextlib import closing

    from diffract import Session
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager

    executable, reference = _reference(tmp_path)
    original = tmp_path / "source"
    _run(original, executable, reference, "edge-" + case, "legacy")
    monkeypatch.chdir(original)
    store = original / ".diffract" / "sqlite"
    sqlite = store / "metadata_index.db"
    blobs = store / "sqlite_storage.db"
    renames = {"models": {"legacy@,[]": "renamed"}} if case == "reserved" else {}
    if case == "orphan":
        with closing(SQLiteStorageManager(str(blobs))) as backend:
            backend.set_field("missing-owner", "observed", 1234, table="parameters")
        with pytest.raises(ValueError, match=r"unknown.*owner"):
            migrate_store(store, have_backup=True)
        with closing(sqlite3.connect(sqlite)) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (0,)
        with closing(SQLiteStorageManager(str(blobs))) as backend:
            assert (
                backend.get_field("missing-owner", "observed", table="parameters")
                == 1234
            )
        return
    if case == "duplicate":
        with closing(sqlite3.connect(sqlite)) as db, db:
            row = db.execute(
                "SELECT name,model_id,ptype,json_data FROM parameters "
                "ORDER BY uid LIMIT 1"
            ).fetchone()
            db.execute(
                "INSERT INTO parameters(uid,name,model_id,ptype,json_data) "
                "VALUES(?,?,?,?,?)",
                ("duplicate", *row),
            )
        assert migrate_store(store, dry_run=True)["status"] == "blocked"
        with closing(sqlite3.connect(sqlite)) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (0,)
            assert db.execute("SELECT count(*) FROM parameters").fetchone() == (9,)
    if case == "reserved":
        assert migrate_store(store, dry_run=True)["status"] == "blocked"
    with closing(sqlite3.connect(sqlite)) as db:
        coords = dict(db.execute("SELECT uid,name FROM parameters"))
    with closing(SQLiteStorageManager(str(blobs))) as backend:
        expected = {
            name: backend.get_field(uid, "observed", table="parameters")
            for uid, name in coords.items()
            if uid != "duplicate"
        }
    options = {"keep": "oldest"} if case == "duplicate" else {}
    result = migrate_store(
        store, rename=renames, backup_dir=tmp_path / "backup", **options
    )
    assert result["status"] == "migrated"
    assert result["parameters"] == (0 if case == "empty" else 8)
    with Session(profile="local") as session:
        if case == "empty":
            assert session.models.list() == []
            return
        assert session.models.list() == (["renamed"] if case == "reserved" else ["net"])
        rows = session.results.export_metrics("observed", export_format="dict")
        assert {
            r["metadata"]["name"]: r["fields"]["observed"] for r in rows.values()
        } == expected
        if case == "partial":
            listed = session.models.parameters.list(verbose=True)
            assert all(
                "weights_svals" not in p["available_fields"]
                and "weights_lsvs" in p["available_fields"]
                for p in listed
            )
            session.compute.apply("weights_svals")
            svd = session.results.export_metrics(
                "weights", "weights_svals", export_format="dict"
            )
            for entry in svd.values():
                w = entry["fields"]["weights"]
                np.testing.assert_allclose(
                    entry["fields"]["weights_svals"],
                    np.sort(np.linalg.svd(w, compute_uv=False)),
                )
