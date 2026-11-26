"""Exercise persistent freshness through the public compute API."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from diffract import Session
from diffract.core.compute import KernelApplyLevel, OutputComparison
from diffract.core.storage.catalog_manager import CatalogStorageManager

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path / ".diffract" / "sqlite" / "metadata_index.db"


def test_apply_recomputes_rank_for_the_effective_configuration(store: Path) -> None:
    with Session(profile="local") as session:
        session.models.add({"w": np.diag([1.0, 0.2])}, model_id="model")
        session.compute.configure_kernel("hard_rank", rtol=1e-9)
        session.compute.apply("hard_rank")
    with Session(profile="local") as session:
        session.compute.configure_kernel("hard_rank", rtol=0.5)
        session.compute.apply("hard_rank")
        (row,) = session.results.export_metrics(
            "hard_rank", export_format="dict"
        ).values()
        assert row["fields"]["hard_rank"] == 1
        session.compute.apply("hard_rank")
    with closing(sqlite3.connect(store)) as index:
        assert index.execute(
            "SELECT config_json FROM runs WHERE kernel_name='hard_rank' ORDER BY seq"
        ).fetchall() == [('{"rtol":1e-09}',), ('{"rtol":0.5}',)]


def test_forced_apply_recomputes_output_groups_once_in_scope(store: Path) -> None:
    shift = 0.0
    parameter_calls: list[float] = []
    aggregate_calls: list[float] = []

    with Session(profile="local") as session:
        session.models.add({"w": np.eye(2)}, model_id="first")
        session.models.add({"w": np.eye(3)}, model_id="second")

        @session.compute.kernel(
            require_fields=("weights",), produce_fields=("force_value", "force_sibling")
        )
        def force_pair(weights: np.ndarray) -> tuple[float, float]:
            value = float(weights.sum()) + shift
            parameter_calls.append(value)
            return value, -value

        @session.compute.kernel(
            require_fields=("force_value",),
            produce_fields=("force_aggregate",),
            apply_level=KernelApplyLevel.IN_MODEL,
        )
        def force_total(force_value: list[float]) -> float:
            value = sum(force_value) + shift
            aggregate_calls.append(value)
            return value

        session.compute.apply("force_aggregate")
        parameter_calls.clear()
        aggregate_calls.clear()
        shift = 10.0
        scoped = session.filter(model_ids=["first"])
        scoped.compute.apply("force_aggregate")
        assert parameter_calls == aggregate_calls == []

        summary = scoped.compute.apply(
            "force_aggregate", "force_value", "force_sibling", recompute=True
        )
        assert summary.forced_fields == (
            "force_aggregate",
            "force_sibling",
            "force_value",
        )
        assert parameter_calls == [12.0]
        assert aggregate_calls == [22.0]

        rows = session.results.export_metrics(
            "force_value", "force_sibling", export_format="dict"
        )
        assert {
            row["metadata"]["model_id"]: row["fields"] for row in rows.values()
        } == {
            "first": {"force_value": 12.0, "force_sibling": -12.0},
            "second": {"force_value": 3.0, "force_sibling": -3.0},
        }


def test_parameter_dependencies_use_the_same_scope_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CatalogStorageManager.unverified_parameter_owners
    with Session(profile="ram") as session:
        session.models.add({"first": np.eye(2)}, model_id="initial")
        (initial,) = session.models.parameters.list()
        inserted = False

        def publish_between_scope_and_verification(
            storage: CatalogStorageManager,
            owners: Sequence[str] | None,
            inputs: Sequence[str],
            outputs: Sequence[str],
            config_hash: str,
        ) -> tuple[list[str], bool] | None:
            nonlocal inserted
            if not inserted:
                inserted = True
                session.models.add({"second": np.eye(3)}, model_id="added")
            return original(storage, owners, inputs, outputs, config_hash)

        monkeypatch.setattr(
            CatalogStorageManager,
            "unverified_parameter_owners",
            publish_between_scope_and_verification,
        )
        session.compute.apply("frob_norm")
        values = session.results.export_metrics("frob_norm", export_format="dict")
        assert {uid for uid, row in values.items() if "frob_norm" in row["fields"]} == {
            initial["uid"]
        }

        session.compute.apply("frob_norm")
        values = session.results.export_metrics("frob_norm", export_format="dict")
        assert len(values) == 2
        assert all("frob_norm" in row["fields"] for row in values.values())


def test_equal_upstream_output_does_not_recompute_an_aggregate(store: Path) -> None:
    calls = {"point": 0, "group": 0}
    with Session(profile="local") as session:

        @session.compute.kernel()
        def point(weights: np.ndarray, *, offset: float = 0.0) -> float:
            calls["point"] += 1
            return float(weights[0, 0]) + offset

        @session.compute.kernel(apply_level=KernelApplyLevel.IN_MODEL)
        def group(point: list[float]) -> float:
            calls["group"] += 1
            return sum(point)

        session.models.add({"w": 4.0 * np.eye(2)}, model_id="model")
        (model_parameter,) = session.models.parameters.list()
        uid = model_parameter["uid"]
        session.compute.apply("group")
        assert calls == {"point": 1, "group": 1}

        session.compute.configure_kernel("point", offset=1e-10)
        session.compute.apply("group")
        assert calls == {"point": 2, "group": 1}
        assert (
            session.results.export_metrics("point", export_format="dict")[uid][
                "fields"
            ]["point"]
            == 4.0
        )

        session.compute.configure_kernel("point", offset=1.0)
        session.compute.apply("group")
        assert calls == {"point": 3, "group": 2}
        (aggregate,) = session.results.export_aggregates("group", export_format="list")
        assert aggregate["value"] == 5.0

        session.results.ingest_metrics({uid: {"weights": 6.0 * np.eye(2)}}, force=True)
        session.compute.apply("group")
        assert calls == {"point": 4, "group": 3}
        (aggregate,) = session.results.export_aggregates("group", export_format="list")
        assert aggregate["value"] == 7.0

    with closing(sqlite3.connect(store)) as index:
        assert index.execute(
            "SELECT COUNT(*) FROM runs WHERE kernel_name='group' AND status='COMPLETE'"
        ).fetchone() == (3,)


@pytest.mark.parametrize(("budget", "payload_is_preserved"), [(31, False), (32, True)])
def test_declared_array_equivalence_respects_its_byte_budget(
    store: Path, budget: int, payload_is_preserved: bool
) -> None:
    comparisons = 0

    def same_direction(previous: object, current: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return bool(np.array_equal(np.asarray(previous), -np.asarray(current)))

    with Session(profile="local") as session:

        @session.compute.kernel(
            output_comparison=OutputComparison(same_direction, max_array_bytes=budget)
        )
        def basis(weights: np.ndarray, *, sign: int = 1) -> np.ndarray:
            return sign * weights[:, 0]

        session.models.add({"w": np.eye(2, dtype=np.float64)}, model_id="model")
        session.compute.apply("basis")
        session.compute.configure_kernel("basis", sign=-1)
        session.compute.apply("basis")
        (row,) = session.results.export_metrics("basis", export_format="dict").values()
        expected = 1.0 if payload_is_preserved else -1.0
        np.testing.assert_array_equal(row["fields"]["basis"], [expected, 0.0])
    assert comparisons == int(payload_is_preserved)
    with closing(sqlite3.connect(store)) as index:
        assert index.execute(
            "SELECT built_at_seq = changed_at_seq FROM live_field_values "
            "WHERE field='basis'"
        ).fetchone() == (int(not payload_is_preserved),)
