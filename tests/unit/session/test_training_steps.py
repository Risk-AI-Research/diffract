"""Checkpoint identity, filtering, labels, and compute/export ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.session import ModelAlreadyExistsError, Session
from diffract.session.resolver import resolve_training_coordinates

from .test_session import _write_ram_config

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("state_dict", [False, True], ids=["module", "state-dict"])
def test_torch_checkpoints_survive_in_place_training_updates(state_dict: bool) -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad(), Session(profile="ram") as session:
        module.weight.copy_(torch.eye(8))
        source = module.state_dict() if state_dict else module
        session.models.add(source, model_id="net", step=0)
        session.compute.apply("frob_norm")
        module.weight.mul_(2)
        session.models.add(source, model_id="net", step=1)
        module.weight.zero_()
        (status,) = session.filter(steps=[0]).results.status("frob_norm")
        assert status.status == "fresh"
        session.compute.apply("frob_norm")
        rows = session.results.export_metrics(
            "frob_norm", "weights", export_format="list"
        )
        assert [row["step"] for row in rows] == [0, 1]
        for row, scale in zip(rows, (1, 2), strict=True):
            np.testing.assert_array_equal(row["weights"], np.eye(8) * scale)
            assert row["frob_norm"] == pytest.approx(np.sqrt(8) * scale)


def test_checkpoints_compose_and_keep_per_instance_final(tmp_path: Path) -> None:
    cfg = tmp_path / "steps.ini"
    _write_ram_config(cfg)
    session = Session(config_path=cfg)
    for model, step in [("a", 10), ("b", 2), ("a", 2), ("a", 0), ("b", 0)]:
        session.models.add(
            {"w": np.eye(2)}, model_id=model, step=step, display_label=f"{model}:{step}"
        )
    with pytest.raises(ModelAlreadyExistsError, match="step 2"):
        session.models.add({"other": np.eye(3)}, model_id="a", step=2)
    reopened = Session(config_path=cfg)
    selected = reopened.filter(model_ids=["a"], steps=range(0, 12, 2))
    with reopened:
        coords = resolve_training_coordinates(
            selected._param_filter_context, [10, 0, 2]
        )
    assert [step for _, step in coords] == [0, 2, 10]
    assert [row["step"] for row in selected.models.parameters.list()] == [0, 2, 10]
    final = reopened.filter(steps="final").models.parameters.list(verbose=True)
    assert [(row["model_id"], row["step"], row["step_label"]) for row in final] == [
        ("a", 10, "a:10"),
        ("b", 2, "b:2"),
    ]
    assert selected.filter(steps=[0, 2]).filter(steps="final").models.list() == []
    assert reopened.filter(steps=[]).models.list() == []
    assert reopened.filter(steps=[0, "final"]).filter(
        model_ids=["b"]
    ).models.list() == ["b"]


def test_partial_step_erase_and_merge_keep_coordinate_labels(tmp_path: Path) -> None:
    cfg = tmp_path / "source.ini"
    dst_cfg = tmp_path / "dest.ini"
    _write_ram_config(cfg)
    _write_ram_config(dst_cfg)
    source = Session(config_path=cfg)
    target = Session(config_path=dst_cfg)
    with source, target:
        source.models.add({"w": np.eye(2)}, model_id="net", display_label="start")
        source.models.add(
            {"w": np.ones((2, 2)), "v": np.zeros((2, 2))},
            model_id="net",
            step=5,
            display_label="epoch five",
        )
        target.models.add(
            {"w": np.eye(2)}, model_id="net", step=5, display_label="custom"
        )
        target.utils.merge_other_session(source, verify=True)
        assert {
            r["step_label"] for r in target.models.parameters.list(verbose=True)
        } == {"start", "custom"}
        target.utils.merge_other_session(source, verify=False)
        target.filter(steps=[5], param_names=["w"]).models.erase("net")
        final = target.filter(steps="final").models.parameters.list(verbose=True)
        assert [(r["step"], r["name"], r["step_label"]) for r in final] == [
            (5, "v", "epoch five")
        ]
        target.filter(steps=[5]).models.erase("net")
        assert (
            target.filter(steps="final").models.parameters.list(verbose=True)[0][
                "step_label"
            ]
            == "start"
        )
        target.models.add({"w": np.eye(2)}, model_id="net", step=5)
        assert (
            target.filter(steps=[5]).models.parameters.list(verbose=True)[0][
                "step_label"
            ]
            is None
        )
        target.models.erase("net")
        assert target.models.list() == []


@pytest.mark.parametrize("bad", [True, -1, 2**63, 1.2, "10", "final"])
def test_invalid_step_aborts_before_admission(tmp_path: Path, bad: object) -> None:
    cfg = tmp_path / "invalid.ini"
    _write_ram_config(cfg)
    session = Session(config_path=cfg)
    with pytest.raises((TypeError, ValueError), match="step"):
        session.models.add({"w": np.eye(2)}, model_id="net", step=bad)
    assert session.models.list() == []


def test_compute_runs_in_numeric_checkpoint_order(tmp_path: Path) -> None:
    from diffract.containers import create_main_container

    cfg = tmp_path / "grouping.ini"
    _write_ram_config(cfg)
    container = create_main_container(cfg)
    session = Session(container=container)
    seen = []

    def record(operands):
        values = [int(weight[0, 0]) for weight in operands]
        seen.append(values)
        return float(sum(values))

    registry = container.compute_singleton.kernel_registry()
    registry.register_kernel(
        name="step_probe",
        require_fields=("weights",),
        produce_fields=("step_probe",),
        implementation=record,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(record)[1],
    )
    with session:
        for step in [10, 0, 2]:
            session.models.add(
                {"w": np.full((2, 2), float(step))}, model_id="one", step=step
            )
        session.compute.apply("step_probe")
    assert seen == [[0], [2], [10]]


def test_export_rows_expose_steps_in_numeric_order(tmp_path: Path) -> None:
    cfg = tmp_path / "export.ini"
    _write_ram_config(cfg)
    session = Session(config_path=cfg)
    with session:
        for step in [10, 0, 2]:
            session.models.add({"w": np.eye(2) * (step + 1)}, model_id="one", step=step)
        session.compute.apply("frob_norm")
        rows = session.results.export_metrics("frob_norm", export_format="list")
    assert [row.get("step") for row in rows] == [0, 2, 10]


def test_coordinate_list_and_scoped_erase(tmp_path: Path) -> None:
    """Listing and erasure use the same model and checkpoint boundaries."""
    from diffract.session import ModelNotFoundError

    cfg = tmp_path / "surface.ini"
    _write_ram_config(cfg)
    with Session(config_path=cfg) as session:
        for name, step in [("b", 10), ("a", 1000), ("a", 0), ("b", 2)]:
            session.models.add({"w": np.eye(2)}, model_id=name, step=step)
        assert session.models.list() == ["a", "b"]
        assert session.models.list(layout="coordinates") == [
            ("a", 0),
            ("a", 1000),
            ("b", 2),
            ("b", 10),
        ]
        assert session.models.list(layout="grouped") == {"a": [0, 1000], "b": [2, 10]}
        selection = session.filter(model_ids=["a"], steps=[0])
        assert selection.models.list(layout="grouped") == {"a": [0]}
        with pytest.raises(ModelNotFoundError):
            selection.models.erase(("a", 0), ("a", 1000))
        with pytest.raises(ModelNotFoundError):
            selection.models.erase("b")
        assert selection.models.list(layout="coordinates") == [("a", 0)]
        removed = session.models.erase(("a", 1000), ("a", 1000))
        assert removed.affected_uids == 1
        assert removed.models == ("a",)
        assert session.models.list(layout="grouped") == {"a": [0], "b": [2, 10]}
        removed = session.models.erase(("b", 2), "b")
        assert removed.affected_uids == 2
        assert removed.models == ("b",)
        assert session.models.list() == ["a"]


def test_cascade_tombstones_before_members_and_rolls_back_on_failure(
    tmp_path: Path,
) -> None:
    """Exercise database ordering and rollback while another checkpoint survives."""
    import sqlite3
    from contextlib import closing

    from diffract.core.data.metadata.errors import ConstraintViolationError

    cfg = tmp_path / "cascade.ini"
    _write_ram_config(cfg)
    with Session(config_path=cfg) as session:
        for step in [0, 1000]:
            session.models.add(
                {"w": np.eye(2)}, model_id="net", step=step, display_label=str(step)
            )
        session.compute.apply("param_norm")
        with closing(
            sqlite3.connect(tmp_path / "cascade_metadata.db", isolation_level=None)
        ) as db:
            before = db.execute(
                "SELECT uid FROM parameters WHERE step=1000"
            ).fetchone()[0]
            old_agg = db.execute(
                "SELECT aggregate_uid FROM aggregate_members WHERE member_uid=?",
                (before,),
            ).fetchone()[0]
            db.executescript("""
                CREATE TRIGGER value_order BEFORE DELETE ON aggregate_members
                WHEN EXISTS(SELECT 1 FROM live_field_values
                            WHERE obj_uid IN (OLD.aggregate_uid, OLD.member_uid))
                BEGIN SELECT RAISE(ABORT, 'values still visible'); END;
                CREATE TRIGGER stop_deletion BEFORE DELETE ON parameters
                BEGIN SELECT RAISE(ABORT, 'injected deletion failure'); END;
            """)
            with pytest.raises(
                ConstraintViolationError, match="injected deletion failure"
            ):
                session.models.erase(("net", 1000))
            assert (
                db.execute(
                    "SELECT count(*) FROM live_field_values WHERE obj_uid=?", (before,)
                ).fetchone()[0]
                > 0
            )
            assert db.execute(
                "SELECT count(*) FROM aggregate_members WHERE aggregate_uid=?",
                (old_agg,),
            ).fetchone() == (1,)
            aggregate_owner = "SELECT uid FROM aggregates WHERE uid=?"
            assert db.execute(aggregate_owner, (old_agg,)).fetchall() == [(old_agg,)]
            db.execute("DROP TRIGGER stop_deletion")
            assert session.models.erase(("net", 1000)).affected_uids == 1
            assert db.execute(
                "SELECT count(*) FROM live_field_values WHERE obj_uid IN (?,?)",
                (before, old_agg),
            ).fetchone() == (0,)
            assert (
                db.execute(
                    "SELECT count(*) FROM field_values WHERE obj_uid IN (?,?) "
                    "AND tombstoned_at IS NOT NULL",
                    (before, old_agg),
                ).fetchone()[0]
                > 1
            )
            assert db.execute(aggregate_owner, (old_agg,)).fetchall() == []
            assert db.execute("SELECT step FROM step_labels").fetchall() == [(0,)]
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert session.models.list(layout="coordinates") == [("net", 0)]
        assert (
            len(session.results.export_aggregates("param_norm", export_format="dict"))
            == 1
        )


def test_erase_coordinate_set_preserves_other_model_step_pairs(ram_container) -> None:
    """Removing a diagonal of a model-by-step grid leaves its other diagonal."""
    with Session(container=ram_container) as session:
        for model, offset in [("a", 1), ("b", 3)]:
            for step in [0, 10]:
                session.models.add(
                    {"w": np.eye(2) * (offset + step)}, model_id=model, step=step
                )
        assert session.models.erase(("a", 0), ("b", 10)).affected_uids == 2
        assert session.models.list(layout="coordinates") == [("a", 10), ("b", 0)]
        for model, step, weight in [("a", 10, 11), ("b", 0, 3)]:
            stored = session.filter(
                model_ids=[model], steps=[step]
            ).results.export_metrics("weights", export_format="dict")
            [row] = stored.values()
            np.testing.assert_array_equal(row["fields"]["weights"], np.eye(2) * weight)


def test_scatter_groups_exported_parameter_coordinates_across_steps() -> None:
    pytest.importorskip("plotly")
    from diffract import FieldSelector
    from diffract.viz.data import FieldRef

    steps = [2, 9, 100]
    with Session(profile="ram") as session:
        for step in reversed(steps):
            session.models.add(
                {"left": np.diag([step, 0.0]), "right": np.diag([-2.0 * step, 0.0])},
                model_id="net",
                step=step,
            )
        session.compute.apply("frob_norm")
        for key in ("name", "parameter_name", "param_name"):
            figure = session.viz.scatter(x="step", y="frob_norm", group_by=key)
            assert {
                trace.name: (list(trace.x), list(trace.y)) for trace in figure.data
            } == {
                "left": (steps, [2, 9, 100]),
                "right": (steps, [4, 18, 200]),
            }
        selected = session.viz.scatter(
            x="step",
            y="frob_norm",
            group_by=FieldRef(FieldSelector("param_name", steps=(9,))),
            value_filter={"parameter_name": ("==", "right")},
        )
        (trace,) = selected.data
        assert (trace.name, list(trace.x), list(trace.y)) == ("right", [9], [18])
