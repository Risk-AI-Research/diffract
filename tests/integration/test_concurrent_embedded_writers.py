"""Exercise multiple independent SQLite sessions writing one store."""

from __future__ import annotations

import multiprocessing
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from diffract import Session

if TYPE_CHECKING:
    from multiprocessing.synchronize import Barrier

pytestmark = [pytest.mark.integration, pytest.mark.stress]

_WRITERS = 4
_ROUNDS = 8
_SHAPE = (3, 3)
_METRIC = "observed_sum"


def _write_checkpoints(cfg: Path, worker: int, barrier: Barrier) -> None:
    with Session(config_path=cfg) as session:

        @session.compute.kernel(require_fields=("weights",), produce_fields=(_METRIC,))
        def observed_sum(weights: np.ndarray) -> int:
            return int(weights.sum())

        for turn in range(_ROUNDS):
            barrier.wait(timeout=45)
            step = turn * _WRITERS + worker
            session.models.add(
                {
                    "left.weight": np.full(_SHAPE, step + 1, dtype=np.float32),
                    "right.weight": np.full(_SHAPE, step + 2, dtype=np.float32),
                },
                model_id="shared",
                step=step,
            )
            session.filter(model_ids=["shared"], steps=step).compute.apply(_METRIC)


def _check_concurrent_writes(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    values = tmp_path / "values.db"
    cfg = tmp_path / "session.ini"
    cfg.write_text(
        f"""
[metadata]
backend = sqlite
[metadata.sqlite]
path = {index}
[storage]
backend = sqlite
[storage.sqlite]
path = {values}
[cache]
backend = "none"
[parallel.thread_pool]
max_workers = 1
[nn.extractor]
skip_not_implemented_types = true
""".strip()
        + "\n"
    )
    with Session(config_path=cfg) as session:
        assert session.models.list() == []

    spawn = multiprocessing.get_context("spawn")
    barrier = spawn.Barrier(_WRITERS)
    workers = [
        spawn.Process(target=_write_checkpoints, args=(cfg, worker, barrier))
        for worker in range(_WRITERS)
    ]
    started = []
    try:
        for process in workers:
            process.start()
            started.append(process)
        deadline = time.monotonic() + 120
        for process in started:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in started] == [0] * _WRITERS
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
            process.close()

    steps = range(_WRITERS * _ROUNDS)
    expected = {
        (step, name): step + offset
        for step in steps
        for name, offset in (("left.weight", 1), ("right.weight", 2))
    }
    with Session(config_path=cfg) as reopened:
        assert reopened.models.list(layout="coordinates") == [
            ("shared", step) for step in steps
        ]
        observations = reopened.results.export_metrics(
            "weights", _METRIC, export_format="dict"
        )
        assert len(observations) == len(expected)
        for row in observations.values():
            meta, fields = row["metadata"], row["fields"]
            marker = expected.pop((meta["step"], meta["name"]))
            np.testing.assert_array_equal(fields["weights"], np.full(_SHAPE, marker))
            assert fields[_METRIC] == marker * _SHAPE[0] * _SHAPE[1]
        assert not expected

    with closing(sqlite3.connect(index)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        expected_runs = 2 * _WRITERS * _ROUNDS
        assert db.execute(
            "SELECT count(*), count(DISTINCT seq), min(seq), max(seq) FROM runs"
        ).fetchone() == (expected_runs, expected_runs, 1, expected_runs)
        assert db.execute(
            "SELECT count(*) FROM live_field_values AS f "
            "JOIN runs AS r USING(run_id) "
            "WHERE r.status='COMPLETE' AND f.changed_at_seq=r.seq"
        ).fetchone() == (2 * len(observations),)
        catalog = set(db.execute("SELECT obj_uid,field FROM live_field_values"))
        assert catalog == {
            (uid, field) for uid in observations for field in ("weights", _METRIC)
        }

    with closing(sqlite3.connect(values)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
        physical = set(
            db.execute("SELECT obj_uid,field FROM storage WHERE table_name=''")
        )
        assert physical == catalog


def test_four_processes_ingest_and_apply_into_one_sqlite_store(tmp_path: Path) -> None:
    # Session initialization wires providers at module scope. Keep the entire
    # scenario, including the verification session, outside the pytest process.
    subprocess.run(  # noqa: S603 - run this test module with the active interpreter
        [sys.executable, __file__, str(tmp_path)],
        check=True,
        timeout=240,
    )


if __name__ == "__main__":
    _check_concurrent_writes(Path(sys.argv[1]))
