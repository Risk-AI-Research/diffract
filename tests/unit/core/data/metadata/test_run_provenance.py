"""Exercise persistent run sequencing and the generation constraints."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from diffract import Session
from diffract.core.compute.config import KernelConfig
from diffract.core.data.metadata.migrations import (
    DIFFRACT_APPLICATION_ID,
    MIGRATIONS,
    run_migrations,
    upgrade_metadata_index,
)
from diffract.core.data.metadata.runs import Run, RunJournal
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.storage.catalog_manager import CatalogStorageManager
from diffract.core.storage.sqlite_manager import SQLiteStorageManager

pytestmark = pytest.mark.unit


def test_run_start_and_outcome_are_visible_to_an_independent_reader(tmp_path) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.connect()
        journal = RunJournal(index)
        config = KernelConfig(scale=2).identity("sum", "1")
        first, second = journal.start_batch("kernel", 2, config)
        ingestion = journal.start("ingest")
        assert (first.seq, second.seq, ingestion.seq) == (1, 2, 3)

        with closing(sqlite3.connect(path)) as reader:
            assert reader.execute(
                "SELECT kind, status, kernel_version, config_hash "
                "FROM runs ORDER BY seq"
            ).fetchall() == [
                ("kernel", "START", "1", config.config_hash),
                ("kernel", "START", "1", config.config_hash),
                ("ingest", "START", None, None),
            ]
            journal.finish(first, "COMPLETE")
            journal.finish(first, "COMPLETE")
            journal.finish(second, "FAILED")
            with pytest.raises(ValueError, match="Cannot mark run"):
                journal.finish(first, "FAILED")
            assert reader.execute(
                "SELECT status, finished_at_ns IS NOT NULL FROM runs ORDER BY seq"
            ).fetchall() == [("COMPLETE", 1), ("FAILED", 1), ("START", 0)]


def test_write_exception_records_failure_and_restores_the_active_run(tmp_path) -> None:
    with closing(SQLiteMetadataIndex(str(tmp_path / "metadata.db"))) as index:
        index.connect()
        journal = RunJournal(index)
        with journal.activity("ingest") as ingestion:
            assert journal.active == ingestion
        with (
            pytest.raises(RuntimeError, match="write failed"),
            journal.activity("merge") as run,
        ):
            raise RuntimeError("write failed")
        assert journal.active is None
        assert index.find_row(
            "runs", columns=("status",), where={"run_id": run.run_id}
        ) == {"status": "FAILED"}


def test_finishing_many_runs_batches_updates_and_rolls_back_a_mismatched_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        journal = RunJournal(index)
        first, second, third = journal.start_batch("ingest", 3)
        journal.finish(third, "COMPLETE")

        with pytest.raises(ValueError, match="Cannot mark run"):
            journal.finish_batch((first, Run(first.run_id, second.seq)), "COMPLETE")
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute(
                "SELECT status FROM runs ORDER BY seq"
            ).fetchall() == [
                ("START",),
                ("START",),
                ("COMPLETE",),
            ]

        statements: list[str] = []
        index._conn.set_trace_callback(statements.append)
        try:
            journal.finish_batch((first, second, third), "COMPLETE")
        finally:
            index._conn.set_trace_callback(None)
        assert (
            sum(
                statement.upper().startswith('UPDATE "RUNS"')
                for statement in statements
            )
            == 1
        )
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT DISTINCT status FROM runs").fetchall() == [
                ("COMPLETE",)
            ]


def test_generation_constraints_cover_inserts_and_updates(tmp_path) -> None:
    path = tmp_path / "metadata.db"
    with closing(sqlite3.connect(path, isolation_level=None)) as legacy:
        run_migrations(legacy, steps=MIGRATIONS[:2])
        legacy.execute(f"PRAGMA application_id={DIFFRACT_APPLICATION_ID}")
        legacy.execute(
            "INSERT INTO field_values "
            "(obj_uid,obj_kind,field,backend,storage_key,format_version) VALUES "
            "('legacy','parameter','weights','ram','legacy/weights',0), "
            "('pretrace','parameter','weights','ram','pretrace/weights',1)"
        )
    assert upgrade_metadata_index(path) == [3]
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.connect()
        conn = index._conn

        assert conn.execute(
            "SELECT obj_uid, kind, built_at_seq, changed_at_seq FROM field_values "
            "LEFT JOIN runs USING(run_id) ORDER BY obj_uid"
        ).fetchall() == [
            ("legacy", None, None, None),
            ("pretrace", "migration", None, None),
        ]
        conn.execute(
            "UPDATE field_values SET tombstoned_at='now' WHERE obj_uid='legacy'"
        )
        for change in ("format_version=1", "changed_at_seq=99"):
            with pytest.raises(sqlite3.IntegrityError, match="generation requires"):
                conn.execute(f"UPDATE field_values SET {change} WHERE obj_uid='legacy'")

        run = RunJournal(index).start("ingest")
        insert = (
            "INSERT INTO field_values (obj_uid,obj_kind,field,backend,storage_key,"
            "format_version,run_id,output_name) "
            "VALUES ('new','parameter',?,'ram',?,1,?,?)"
        )
        with pytest.raises(sqlite3.IntegrityError, match="generation requires"):
            conn.execute(insert, ("x", "new/x", None, "x"))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(insert, ("x", "new/x", "unknown run", "x"))
        conn.execute(insert, ("x", "new/x", run.run_id, "x"))
        conn.execute(insert, ("y", "new/y", run.run_id, "y"))
        with pytest.raises(sqlite3.IntegrityError, match="generation requires"):
            conn.execute("UPDATE field_values SET run_id=NULL WHERE obj_uid='new'")
        with pytest.raises(sqlite3.IntegrityError, match="generation requires"):
            conn.execute(
                insert + " ON CONFLICT(obj_uid,field) DO UPDATE SET run_id = NULL",
                ("x", "new/x", run.run_id, "x"),
            )
        assert conn.execute(
            "SELECT run_id, output_name FROM field_values "
            "WHERE obj_uid='new' ORDER BY output_name"
        ).fetchall() == [(run.run_id, "x"), (run.run_id, "y")]


def test_generation_round_trips_through_the_storage_writer(tmp_path) -> None:
    metadata = tmp_path / "metadata.db"
    values = tmp_path / "values.db"
    with (
        closing(SQLiteMetadataIndex(str(metadata))) as index,
        closing(
            CatalogStorageManager(SQLiteStorageManager(str(values)), index)
        ) as storage,
    ):
        storage.set_field("owner", "direct", 1, table="parameters")
        direct = storage.live_pointer("owner", "direct", "parameters").generation
        assert direct is not None
        assert index.find_row(
            "runs", columns=("kind", "status"), where={"run_id": direct.run_id}
        ) == {"kind": "ingest", "status": "COMPLETE"}
        config = KernelConfig(scale=2).identity("total", "2")
        assert storage.journal is not None
        with storage.journal.activity("kernel", config) as run, storage:
            storage.set_field("owner", "sum", 7, table="parameters")
            storage.set_field("owner", "count", 3, table="parameters")
        for field, value in (("sum", 7), ("count", 3)):
            pointer = storage.live_pointer("owner", field, "parameters")
            assert pointer.generation is not None
            assert pointer.generation.run_id == run.run_id
            assert pointer.generation.output_name == field
            assert pointer.generation.changed_at_seq == run.seq
            assert storage.get_field("owner", field, table="parameters") == value

    with (
        closing(SQLiteMetadataIndex(str(metadata))) as reader,
        closing(
            CatalogStorageManager(SQLiteStorageManager(str(values)), reader)
        ) as storage,
    ):
        assert reader.find_row(
            "runs",
            columns=("status", "kernel_version", "config_hash"),
            where={"run_id": run.run_id},
        ) == {
            "status": "COMPLETE",
            "kernel_version": "2",
            "config_hash": config.config_hash,
        }
        assert storage.get_field("owner", "sum", table="parameters") == 7
        assert storage.get_field("owner", "count", table="parameters") == 3


@pytest.fixture
def persistent_run_config(tmp_path: Path) -> Path:
    metadata = tmp_path / "metadata.db"
    config = tmp_path / "session.ini"
    config.write_text(
        f"""[metadata]
backend = sqlite
[metadata.sqlite]
path = {metadata}
[storage]
backend = sqlite
[storage.sqlite]
path = {tmp_path / "values.db"}
[cache]
backend = "none"
[nn.extractor]
skip_not_implemented_types = true
"""
    )
    return config


def test_session_records_ingestion_and_one_run_for_all_parameter_outputs(
    persistent_run_config: Path,
) -> None:
    config = persistent_run_config
    metadata = config.parent / "metadata.db"
    with Session(config_path=config) as session:
        for step in range(2):
            session.models.add(
                {"w.weight": np.eye(2, dtype=np.float32) * (step + 1)},
                model_id="model",
                step=step,
            )
        uid = session.models.parameters.list()[0]["uid"]
        session.results.ingest_metrics({uid: {"external": 7}})
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        session.compute.apply("weights_svals")
        session.compute.apply("avg_l_agreement")

    with Session(config_path=config) as reopened:
        exported = reopened.results.export_metrics("external", export_format="dict")
        assert exported[uid]["fields"]["external"] == 7
    with closing(sqlite3.connect(metadata)) as reader:
        assert reader.execute(
            "SELECT count(*) FROM live_field_values "
            "WHERE run_id IS NULL OR changed_at_seq IS NULL"
        ).fetchone() == (0,)
        assert reader.execute(
            "SELECT COUNT(*), COUNT(DISTINCT r.run_id), MIN(r.config_json) "
            "FROM live_field_values AS f JOIN runs AS r USING(run_id) "
            "WHERE r.kernel_name='weights_svd' AND r.status='COMPLETE'"
        ).fetchone() == (6, 1, '{"allow_cuda":false,"precision":"preserve"}')
        assert reader.execute(
            "SELECT COUNT(*), COUNT(DISTINCT r.run_id) "
            "FROM live_field_values AS f JOIN runs AS r USING(run_id) "
            "WHERE f.output_name='l_overlap' AND r.status='COMPLETE'"
        ).fetchone() == (2, 2)
        assert reader.execute(
            "SELECT f.output_name, r.kind, f.changed_at_seq = r.seq "
            "FROM live_field_values AS f JOIN runs AS r USING(run_id) "
            "WHERE f.output_name='external'"
        ).fetchone() == ("external", "ingest", 1)


def test_crash_after_catalog_commit_leaves_the_published_run_detectable(
    persistent_run_config: Path,
) -> None:
    config = persistent_run_config
    with Session(config_path=config) as session:
        session.models.add({"w.weight": np.eye(2, dtype=np.float32)}, model_id="model")

    child = """
from os import _exit
from sys import argv

from diffract import Session
from diffract.core.data.metadata.runs import RunJournal

with Session(config_path=argv[1]) as session:
    session.compute.configure_kernel("weights_svd", allow_cuda=False)

    def crash_after_publication(self, runs, outcome):
        if outcome == "COMPLETE":
            _exit(73)
        _exit(74)

    RunJournal.finish_batch = crash_after_publication
    session.compute.apply("weights_svals")
"""
    result = subprocess.run(  # noqa: S603 - fixed fixture code with a temporary store
        [sys.executable, "-c", child, str(config)],
        check=False,
        timeout=60,
    )
    assert result.returncode == 73

    metadata = config.parent / "metadata.db"
    with closing(sqlite3.connect(metadata)) as reader:
        assert reader.execute(
            "SELECT r.status, COUNT(*), COUNT(DISTINCT r.run_id) "
            "FROM live_field_values AS f JOIN runs AS r USING(run_id) "
            "WHERE r.kernel_name='weights_svd' GROUP BY r.status"
        ).fetchall() == [("START", 3, 1)]
    with Session(config_path=config) as reopened:
        results = reopened.results.export_metrics("weights_svals", export_format="dict")
        (row,) = results.values()
        np.testing.assert_array_equal(row["fields"]["weights_svals"], [1.0, 1.0])


def test_reingestion_and_merge_publish_under_distinct_runs(
    persistent_run_config: Path,
) -> None:
    target_config = persistent_run_config
    source_directory = target_config.parent / "source"
    source_directory.mkdir()
    source_config = source_directory / "session.ini"
    source_config.write_text(
        target_config.read_text().replace(
            str(target_config.parent), str(source_directory)
        )
    )

    source = Session(config_path=source_config)
    with source:
        source.models.add({"w": np.eye(2)}, model_id="source")
        (source_parameter,) = source.models.parameters.list()
        source.results.ingest_metrics({source_parameter["uid"]: {"copied": 7}})

    metadata = target_config.parent / "metadata.db"
    with (
        Session(config_path=target_config) as target,
        closing(sqlite3.connect(metadata)) as reader,
    ):
        target.models.add({"w": np.eye(2)}, model_id="target")
        (target_parameter,) = target.models.parameters.list()
        uid = target_parameter["uid"]
        target.results.ingest_metrics({uid: {"observed": 1}})
        observed_generation = (
            "SELECT f.run_id, f.changed_at_seq, r.kind, r.status "
            "FROM live_field_values AS f JOIN runs AS r USING (run_id) "
            "WHERE f.obj_uid=? AND f.output_name='observed'"
        )
        initial = reader.execute(observed_generation, (uid,)).fetchone()
        assert initial[2:] == ("ingest", "COMPLETE")
        target.results.ingest_metrics({uid: {"observed": 2}}, force=True)
        replacement = reader.execute(observed_generation, (uid,)).fetchone()
        assert replacement[2:] == ("ingest", "COMPLETE")
        assert replacement[0] != initial[0]
        assert replacement[1] > initial[1]

        target.utils.merge_other_session(source)
        assert reader.execute(
            "SELECT f.output_name, f.changed_at_seq = r.seq "
            "FROM live_field_values AS f JOIN runs AS r USING (run_id) "
            "WHERE r.kind='merge' AND r.status='COMPLETE' ORDER BY f.output_name"
        ).fetchall() == [("copied", 1), ("weights", 1)]
        observed = target.results.export_metrics("observed", export_format="dict")
        assert observed[uid]["fields"]["observed"] == 2
        copied = target.filter(model_ids=["source"]).results.export_metrics(
            "copied", export_format="dict"
        )
        (copied_row,) = copied.values()
        assert copied_row["fields"]["copied"] == 7
