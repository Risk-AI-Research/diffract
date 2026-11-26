"""Exercise the read-only result status API."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from diffract import Session
from diffract.core.compute import KernelApplyLevel
from diffract.core.storage.catalog_manager import CatalogStorageManager
from diffract.session import KernelNotFoundError

pytestmark = pytest.mark.unit


def test_ingested_arrays_and_multi_output_kernel_views_are_snapshots() -> None:
    observations = np.array([2.0, 3.0, 7.0])
    workspace = np.empty_like(observations)
    with Session(profile="ram") as session:
        session.models.add({"w": np.eye(2)}, model_id="m")
        (parameter,) = session.models.parameters.list()
        session.results.ingest_metrics(
            {parameter["uid"]: {"observations": observations}}
        )

        @session.compute.kernel(produce_fields=("leading", "trailing"))
        def observation_windows(
            observations: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            workspace[:] = observations
            return workspace[:-1], workspace[1:]

        @session.compute.kernel()
        def window_product(trailing: np.ndarray) -> float:
            return float(trailing.prod())

        observations.fill(11)
        session.compute.apply("leading")
        workspace.fill(-1)
        (row,) = session.results.export_metrics("leading", export_format="list")
        row["leading"].fill(17)
        session.compute.apply("window_product")
        (row,) = session.results.export_metrics(
            "observations",
            "leading",
            "trailing",
            "window_product",
            export_format="list",
        )
        np.testing.assert_array_equal(row["observations"], [2, 3, 7])
        np.testing.assert_array_equal(row["leading"], [2, 3])
        np.testing.assert_array_equal(row["trailing"], [3, 7])
        assert row["window_product"] == 21


@pytest.mark.parametrize(
    ("state", "status", "reason"),
    [
        ("missing", "never computed", None),
        ("unchanged", "fresh", None),
        ("reconfigured", "stale", "config key scale changed"),
        ("new_input", "stale", "input changed at run {seq}"),
    ],
)
def test_status_reports_why_without_computing_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    status: str,
    reason: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".diffract" / "sqlite" / "metadata_index.db"
    with Session(profile="local") as session:

        @session.compute.kernel()
        def status_observed(weights: np.ndarray, *, scale: float = 1.0) -> float:
            return float(weights.sum()) * scale

        session.models.add({"w": np.eye(2)}, model_id="selected")
        session.models.add({"w": np.eye(3)}, model_id="other")
        scoped = session.filter(model_ids=["selected"])
        (parameter,) = scoped.models.parameters.list()
        uid = parameter["uid"]
        if state != "missing":
            scoped.compute.apply("status_observed")
        if state == "reconfigured":
            session.compute.configure_kernel("status_observed", scale=2.0)
        elif state == "new_input":
            scoped.results.ingest_metrics({uid: {"weights": 3 * np.eye(2)}}, force=True)

        with closing(sqlite3.connect(path)) as reader:
            before = tuple(reader.iterdump())
            if reason is not None and state == "new_input":
                (seq,) = reader.execute(
                    "SELECT changed_at_seq FROM live_field_values "
                    "WHERE obj_uid=? AND field='weights'",
                    (uid,),
                ).fetchone()
                reason = reason.format(seq=seq)

            def reject_value_read(*args: object, **kwargs: object) -> None:
                raise AssertionError("Status reads metadata only")

            monkeypatch.setattr(
                CatalogStorageManager, "read_pointer", reject_value_read
            )
            (row,) = scoped.results.status("status_observed")
            assert (row.field, row.uid, row.status, row.reason) == (
                "status_observed",
                uid,
                status,
                reason,
            )
            assert tuple(reader.iterdump()) == before


def test_aggregate_status_uses_the_same_coordinates_as_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with Session(profile="local") as session:

        @session.compute.kernel(apply_level=KernelApplyLevel.IN_MODEL)
        def status_aggregate_sum(
            weights: list[np.ndarray], *, scale: float = 1.0
        ) -> float:
            return sum(float(value.sum()) for value in weights) * scale

        session.models.add({"w": np.eye(2)}, model_id="first")
        session.models.add({"w": np.eye(3)}, model_id="second")
        scoped = session.filter(model_ids=["first"])
        (empty,) = scoped.results.status("status_aggregate_sum")
        assert empty.status == "never computed"
        scoped.compute.apply("status_aggregate_sum")
        (fresh,) = scoped.results.status("status_aggregate_sum")
        assert fresh.uid == empty.uid
        assert fresh.status == "fresh"
        session.compute.configure_kernel("status_aggregate_sum", scale=2.0)
        (stale,) = scoped.results.status("status_aggregate_sum")
        assert stale.uid == fresh.uid
        assert stale.reason == "config key scale changed"
        with pytest.raises(KernelNotFoundError, match="Did you mean"):
            scoped.results.status("status_agregate_sum")
