"""Persistent activities and the sequence shared by value generations."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from diffract.core.utils.hashing import new_entity_uid

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator, Sequence

    from diffract.core.compute.config import KernelConfigIdentity, OutputComparison

    from .sqlite_index import SQLiteMetadataIndex

type RunKind = Literal["kernel", "ingest", "merge", "migration"]
type RunOutcome = Literal["COMPLETE", "FAILED"]

_RUN_COUNTER = "run_seq"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Run:
    """A durable start record and its monotonically increasing sequence."""

    run_id: str
    seq: int


@dataclass(frozen=True, slots=True)
class Generation:
    """The activity, output port, and change clocks of a live value."""

    run_id: str
    output_name: str
    built_at_seq: int | None
    changed_at_seq: int | None


class OwnerVerification(NamedTuple):
    """Owners to inspect and whether all requested outputs are absent."""

    candidates: list[str]
    all_outputs_absent: bool


class ValueTrace(NamedTuple):
    """Observed lineage of one live value, without loading its payload."""

    revision: str
    format_version: int
    run_id: str | None
    kind: str | None
    status: str | None
    config_hash: str | None
    built_at_seq: int | None
    changed_at_seq: int | None
    config_json: str | None = None

    @property
    def predates_tracing(self) -> bool:
        """Whether the value is eligible for its first observed verification."""
        return (self.format_version == 0 and self.run_id is None) or (
            self.kind == "migration" and self.built_at_seq is None
        )


def install_run_schema(conn: sqlite3.Connection) -> None:
    """Create the journal and initialize its sequence within a migration."""
    conn.execute(
        "CREATE TABLE counters (name TEXT PRIMARY KEY NOT NULL, "
        "value INTEGER NOT NULL CHECK(value >= 0)) WITHOUT ROWID"
    )
    conn.execute("INSERT INTO counters(name,value) VALUES (?,0)", (_RUN_COUNTER,))
    conn.execute(
        "CREATE TABLE runs ("
        "seq INTEGER PRIMARY KEY NOT NULL CHECK(seq > 0), "
        "run_id TEXT NOT NULL UNIQUE, "
        "kind TEXT NOT NULL CHECK(kind IN ('kernel','ingest','merge','migration')), "
        "kernel_name TEXT, kernel_version TEXT, config_json TEXT, config_hash TEXT, "
        "hash_scheme INTEGER, "
        "status TEXT NOT NULL CHECK(status IN ('START','COMPLETE','FAILED')), "
        "started_at_ns INTEGER NOT NULL, finished_at_ns INTEGER, "
        "producer TEXT NOT NULL, "
        "CHECK((status = 'START' AND finished_at_ns IS NULL) OR "
        "(status <> 'START' AND finished_at_ns IS NOT NULL)), "
        "CHECK((kind = 'kernel' AND kernel_name IS NOT NULL AND "
        "kernel_version IS NOT NULL AND config_json IS NOT NULL AND "
        "config_hash IS NOT NULL AND hash_scheme IS NOT NULL) OR "
        "(kind <> 'kernel' AND kernel_name IS NULL AND kernel_version IS NULL AND "
        "config_json IS NULL AND config_hash IS NULL AND hash_scheme IS NULL)))"
    )


def install_generation_edges(conn: sqlite3.Connection, *, producer: str) -> None:
    """Add provenance columns and enforce them for new value publications."""
    conn.execute(
        "ALTER TABLE field_values ADD COLUMN run_id TEXT REFERENCES runs(run_id)"
    )
    conn.execute("ALTER TABLE field_values ADD COLUMN output_name TEXT")
    conn.execute("ALTER TABLE field_values ADD COLUMN built_at_seq INTEGER")
    conn.execute("ALTER TABLE field_values ADD COLUMN changed_at_seq INTEGER")
    conn.execute("CREATE INDEX ix_values_by_run ON field_values(run_id, obj_uid)")
    has_v2_values = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM field_values WHERE format_version > 0)"
    ).fetchone()[0]
    if has_v2_values:
        # The migration itself is an observed event. The input rows predate
        # configuration and change tracking, so their trace columns stay unset.
        run_id = new_entity_uid()
        now = time.time_ns()
        conn.execute(
            "INSERT INTO runs(seq, run_id, kind, status, started_at_ns, "
            "finished_at_ns, producer) VALUES (1, ?, 'migration', 'COMPLETE', ?, ?, ?)",
            (run_id, now, now, producer),
        )
        conn.execute("UPDATE counters SET value=1 WHERE name=?", (_RUN_COUNTER,))
        conn.execute(
            "UPDATE field_values SET run_id=?, output_name=CASE "
            "WHEN obj_kind='aggregate' THEN coalesce(("
            "SELECT field_name FROM aggregates "
            "WHERE aggregates.uid=field_values.obj_uid"
            "), field) ELSE field END WHERE format_version > 0",
            (run_id,),
        )
    conn.execute(
        "CREATE TRIGGER require_value_generation_on_insert "
        "BEFORE INSERT ON field_values "
        "WHEN NEW.run_id IS NULL OR NEW.output_name IS NULL "
        "BEGIN SELECT RAISE(ABORT, 'value generation requires a run and output name'); "
        "END"
    )
    conn.execute(
        "CREATE TRIGGER require_value_generation_on_update "
        "BEFORE UPDATE ON field_values "
        "WHEN (NEW.run_id IS NULL OR NEW.output_name IS NULL) AND ("
        "OLD.run_id IS NOT NULL OR OLD.format_version <> 0 OR NEW.format_version <> 0 "
        "OR NEW.obj_uid IS NOT OLD.obj_uid OR NEW.obj_kind IS NOT OLD.obj_kind "
        "OR NEW.field IS NOT OLD.field OR NEW.backend IS NOT OLD.backend "
        "OR NEW.storage_key IS NOT OLD.storage_key "
        "OR NEW.size_bytes IS NOT OLD.size_bytes OR NEW.revision IS NOT OLD.revision "
        "OR NEW.run_id IS NOT OLD.run_id OR NEW.output_name IS NOT OLD.output_name "
        "OR NEW.built_at_seq IS NOT OLD.built_at_seq "
        "OR NEW.changed_at_seq IS NOT OLD.changed_at_seq) "
        "BEGIN SELECT RAISE(ABORT, 'value generation requires a run and output name'); "
        "END"
    )


class RunJournal:
    """Commit activity starts and outcomes apart from their value transactions."""

    def __init__(self, index: SQLiteMetadataIndex) -> None:
        self._index = index
        index.define_table("run_arithmetic", {"run_id": str}, indexes=["run_id"])
        self._active: ContextVar[Run | None] = ContextVar("run", default=None)
        self._recording: ContextVar[bool] = ContextVar("recording_run", default=False)
        self._comparison: ContextVar[OutputComparison | None] = ContextVar(
            "output_comparison", default=None
        )

    def record_arithmetic(
        self, kind: str, uid: str, facts: dict[str, Any], *, observed: bool
    ) -> None:
        """Attach one entity's solver facts to the active run's value transaction."""
        run = self.active
        if run is not None:
            self._index.upsert(
                "run_arithmetic",
                f"{run.run_id}/{kind}/{uid}",
                run_id=run.run_id,
                entity_kind=kind,
                owner_uid=uid,
                origin="observed" if observed else "self-reported",
                **facts,
            )

    def describe(
        self, run_id: str | None, *, parameter_uid: str | None = None
    ) -> dict[str, Any]:
        """Read a run and its available observations without accessing weight data."""
        columns = (
            "status",
            "kind",
            "producer",
            "kernel_name",
            "kernel_version",
            "config_json",
            "config_hash",
        )
        if run_id is None:
            row = dict.fromkeys(columns)
            row["status"] = "unknown"
            entries = []
        else:
            row = self._index.find_row(
                "runs", columns=columns, where={"run_id": run_id}
            )
            if row is None:
                raise KeyError(run_id)
            where = {"run_id": run_id}
            if parameter_uid is not None:
                where["uid"] = f"{run_id}/parameter/{parameter_uid}"
            entries = self._index.query_rows(
                "run_arithmetic", columns=("json_data",), where=where, order_by=("uid",)
            )
            if (
                parameter_uid is not None
                and not entries
                and self._index.query(
                    "run_arithmetic", where={"run_id": run_id}, limit=1
                )
            ):
                raise ValueError(
                    f"No observation for parameter {parameter_uid!r} in run {run_id!r}."
                )
        requested = row.pop("config_json")
        return row | {
            "run_id": run_id,
            "requested": json.loads(requested) if requested else None,
            "arithmetic": [json.loads(entry["json_data"]) for entry in entries],
            "arithmetic_status": "recorded" if entries else "unknown",
        }

    def live_owners(self, run_id: str | None) -> set[tuple[str, str]]:
        """Return the current owners of values linked to this run."""
        if run_id is None:
            return set()
        rows = self._index.query_rows(
            "live_field_values",
            columns=("obj_kind", "obj_uid"),
            where={"run_id": run_id},
            distinct=True,
        )
        return {(row["obj_kind"], row["obj_uid"]) for row in rows}

    @property
    def active(self) -> Run | None:
        """Return the run that owns publications in this execution context."""
        return self._active.get()

    @property
    def recording(self) -> bool:
        """Whether an operation has already reserved its journal records."""
        return self._recording.get() or self.active is not None

    @property
    def output_comparison(self) -> OutputComparison | None:
        """The equality policy for computed writes in this context."""
        return self._comparison.get()

    @contextmanager
    def activity(
        self,
        kind: RunKind,
        config: KernelConfigIdentity | None = None,
        *,
        compare: OutputComparison | None = None,
    ) -> Iterator[Run]:
        """Bracket a public write operation and persist its terminal status."""
        with (
            self.activities(kind, 1, config) as runs,
            self.activate(runs[0], compare=compare),
        ):
            yield runs[0]

    @contextmanager
    def activate(
        self,
        run: Run,
        *,
        compare: OutputComparison | None = None,
    ) -> Iterator[None]:
        """Route value writes in this context to an already started run."""
        token = self._active.set(run)
        comparison = self._comparison.set(compare)
        try:
            yield
        finally:
            self._comparison.reset(comparison)
            self._active.reset(token)

    @contextmanager
    def activities(
        self, kind: RunKind, count: int, config: KernelConfigIdentity | None = None
    ) -> Iterator[tuple[Run, ...]]:
        """Bracket a batch of activities with two index transactions."""
        runs = self.start_batch(kind, count, config)
        token = self._recording.set(True)
        try:
            yield runs
        except BaseException:
            try:
                self.finish_batch(runs, "FAILED")
            except Exception:
                logger.exception("Could not mark %d runs FAILED", len(runs))
            raise
        else:
            self.finish_batch(runs, "COMPLETE")
        finally:
            self._recording.reset(token)

    def start(self, kind: RunKind, config: KernelConfigIdentity | None = None) -> Run:
        """Allocate a sequence and commit START before any value writes."""
        return self.start_batch(kind, 1, config)[0]

    def start_batch(
        self, kind: RunKind, count: int, config: KernelConfigIdentity | None = None
    ) -> tuple[Run, ...]:
        """Reserve contiguous sequences for an aggregate batch in one commit."""
        from diffract import __version__

        if count <= 0:
            raise ValueError("A run batch must contain at least one activity")
        if (kind == "kernel") != (config is not None):
            raise ValueError("Kernel runs require a configuration snapshot")
        index = self._index
        if index.in_write_context:
            raise RuntimeError("Commit the run start before opening a value writer")

        where = {"name": _RUN_COUNTER}
        with index:
            counter = index.find_row("counters", columns=("value",), where=where)
            if counter is None:
                raise RuntimeError("Missing run sequence counter")
            first = counter["value"] + 1
            index.update_rows(
                "counters", where=where, values={"value": first + count - 1}
            )
            runs = tuple(
                Run(new_entity_uid(), seq) for seq in range(first, first + count)
            )
            record = {
                "kind": kind,
                "status": "START",
                "started_at_ns": time.time_ns(),
                "producer": f"diffract-core {__version__}",
            }
            if config is not None:
                record.update(
                    kernel_name=config.kernel_name,
                    kernel_version=config.kernel_version,
                    config_hash=config.config_hash,
                    config_json=config.config_json,
                    hash_scheme=config.hash_scheme,
                )
            index.insert_rows(
                "runs",
                [record | {"seq": run.seq, "run_id": run.run_id} for run in runs],
            )
        return runs

    def finish(self, run: Run, outcome: RunOutcome) -> None:
        """Record an outcome after the caller's value transaction has ended."""
        self.finish_batch((run,), outcome)

    def finish_batch(self, runs: Sequence[Run], outcome: RunOutcome) -> None:
        """Commit terminal statuses for a batch of runs."""
        index = self._index
        if index.in_write_context:
            raise RuntimeError("Finish the run after closing its value transaction")
        if not runs:
            return
        with index:
            updated = index.update_rows_returning(
                "runs",
                where={"status": "START"},
                where_in={"seq": [run.seq for run in runs]},
                where_null=(),
                values={"status": outcome, "finished_at_ns": time.time_ns()},
                columns=("seq", "run_id"),
            )
            completed = {(row["seq"], row["run_id"]) for row in updated}
            for run in runs:
                if (run.seq, run.run_id) in completed:
                    continue
                row = index.find_row(
                    "runs",
                    columns=("status",),
                    where={"run_id": run.run_id, "seq": run.seq},
                )
                if row is None or row["status"] != outcome:
                    raise ValueError(f"Cannot mark run {run.run_id} {outcome}")
