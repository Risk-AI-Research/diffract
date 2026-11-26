"""Exercise ordered, name-joined checkpoint comparisons through Session."""

import numpy as np
import pytest

from diffract import FieldSelector, Session
from diffract.core.compute.execution import KernelApplyLevel
from diffract.session.errors import ScopeValidationError
from diffract.session.summaries import PairMismatch

from .test_session import _write_ram_config

pytestmark = pytest.mark.unit


def _add_checkpoints(session):
    weights = {
        ("a", 1): np.array([[2.0, 0.0], [0.0, 1.0]]),
        ("a", 4): np.array([[1.0, 2.0], [0.0, 3.0]]),
        ("b", 2): np.array([[3.0, 1.0], [2.0, 0.0]]),
        ("c", 3): np.array([[1.0, 1.0], [1.0, 2.0]]),
    }
    for (name, step), matrix in weights.items():
        session.models.add(
            {f"only_{name}_{step}": matrix, "shared": matrix},
            model_id=name,
            step=step,
        )
    session.compute.configure_kernel("weights_svd", allow_cuda=False)


def _matches(rows):
    return {
        (
            row["model_source"],
            row["step_source"],
            row["model_target"],
            row["step_target"],
        ): row["value"]
        for row in rows
    }


def test_explicit_pairs_keep_order_match_names_and_resolve_final():
    pairs = [(("a", 1), ("a", 4)), ("a", ("b", 2)), ("b", "a"), ("a", "c")]
    with Session(profile="ram") as session:
        _add_checkpoints(session)
        summary = session.compute.apply("l_overlap", "avg_l_agreement", pairs=pairs)
        rows = _matches(
            session.results.export_aggregates("l_overlap", export_format="list")
        )
        symmetric = session.results.export_aggregates(
            "avg_l_agreement", export_format="list"
        )
        assert summary.pair_mismatches[0] == PairMismatch(
            ("a", 1), ("a", 4), ("only_a_1",), ("only_a_4",)
        )
        assert len(summary.pair_mismatches) == 4
        assert set(rows) == {
            ("a", 1, "a", 4),
            ("a", 4, "b", 2),
            ("b", 2, "a", 4),
            ("a", 4, "c", 3),
        }
        assert len({row["aggregate_uid"] for row in symmetric}) == 3
        np.testing.assert_allclose(rows["a", 4, "b", 2], rows["b", 2, "a", 4].T)
        directed = FieldSelector("l_overlap", roles=("a", "a"), steps=(1, 4))
        [only] = session.results.export_aggregates(directed, export_format="list")
        assert only["context_params"] == ("shared",)

    with Session(profile="ram") as legacy:
        _add_checkpoints(legacy)
        legacy.filter(model_ids=["a"], steps=[1, 4]).compute.apply("l_overlap")
        legacy.filter(model_ids=["a", "b"], steps=[2, 4]).compute.apply("l_overlap")
        legacy.filter(model_ids=["a", "c"], steps=[3, 4]).compute.apply("l_overlap")
        expected = _matches(
            legacy.results.export_aggregates("l_overlap", export_format="list")
        )
    for coordinates, value in rows.items():
        np.testing.assert_allclose(value, expected[coordinates])


def test_explicit_pairs_are_prevalidated_in_active_scope():
    with Session(profile="ram") as session:
        _add_checkpoints(session)
        scope = session.filter(model_ids=["a", "b"], steps=[1, 2])
        with pytest.raises(ScopeValidationError, match="outside the current scope"):
            scope.compute.apply("l_overlap", pairs=[("a", "b"), (("a", 4), "b")])
        assert (
            session.results.export_aggregates("l_overlap", export_format="list") == []
        )
        scope.compute.apply("l_overlap", pairs=[("a", "b")])
        assert set(
            _matches(
                session.results.export_aggregates("l_overlap", export_format="list")
            )
        ) == {("a", 1, "b", 2)}


def test_coverage_resolves_steps_direction_and_cached_measurements():
    field = "relative_weight_change"
    with Session(profile="ram") as session:
        early = session.models.add({"w": np.eye(2), "early": np.eye(2)}, "net", step=3)
        late = session.models.add(
            {"w": 4 * np.eye(2), "late": np.eye(2)}, "net", step=8
        )
        pairs = [(("net", 3), "net"), ("net", ("net", 3))]
        scoped = session.filter(param_names=["w"])
        first = scoped.compute.apply(field, pairs=pairs, return_coverage=True)
        again = scoped.compute.apply(field, pairs=pairs, return_coverage=True)
        assert again.pair_coverage == first.pair_coverage
        forward, reverse = again.pair_coverage
        assert (forward.source, forward.target) == (("net", 3), ("net", 8))
        assert (reverse.source, reverse.target) == (forward.target, forward.source)
        assert (forward.source_extraction, forward.target_extraction) == (early, late)
        assert (reverse.source_extraction, reverse.target_extraction) == (late, early)
        for report in (forward, reverse):
            assert (report.source_count, report.target_count) == (1, 1)
            assert (report.matched, report.source_only, report.target_only) == (
                ("w",),
                (),
                (),
            )
            assert report.measured == {field: ("w",)}
            assert report.measurement_counts == {field: 1}
        rows = session.results.export(
            field, shape="long", sources="aggregates", export_format="list"
        )
        assert {
            (r["step_source"], r["step_target"]): r["value_scalar"] for r in rows
        } == pytest.approx(
            {
                (3, 8): 3.0,
                (8, 3): 0.75,
            },
            rel=1e-14,
        )
        assert len({r["aggregate_uid"] for r in rows}) == 2


@pytest.mark.parametrize("return_coverage", [False, True])
def test_strict_shape_check_validates_all_pairs_before_computing(return_coverage):
    with Session(profile="ram") as session:
        for step, width in enumerate((2, 2, 3)):
            session.models.add({"w": np.eye(2, width)}, "net", step=step)
        with pytest.raises(
            ScopeValidationError,
            match=r"'w'.*source.*'net', 1.*\(2, 2\).*target.*'net', 2.*\(2, 3\)",
        ):
            session.compute.apply(
                "relative_weight_change",
                required_names=("w",),
                return_coverage=return_coverage,
                pairs=[(("net", 0), ("net", 1)), (("net", 1), ("net", 2))],
            )
        assert (
            session.results.export_aggregates(
                "relative_weight_change", export_format="list"
            )
            == []
        )


def test_checkpoint_admission_reports_survive_reopening_and_renaming(tmp_path):
    cfg = tmp_path / "checkpoints.ini"
    _write_ram_config(cfg)
    cfg.write_text(
        cfg.read_text().replace(
            "backend = ram",
            f"backend = sqlite\n\n[storage.sqlite]\npath = {tmp_path / 'weights.db'}",
        )
    )
    pairs = [(("renamed", 1), ("renamed", 2))]
    with Session(config_path=cfg) as session:
        early = session.models.add(
            {"w": np.eye(2), "invalid": np.eye(2, dtype=int)}, "net", step=1
        )
        late = session.models.add({"w": np.eye(2), "extra": np.eye(3)}, "net", step=2)
        session.models.rename("net", "renamed")

    with Session(config_path=cfg) as session:
        (report,) = session.compute.apply(
            "relative_weight_change", pairs=pairs, return_coverage=True
        ).pair_coverage
        assert (report.source_extraction, report.target_extraction) == (early, late)
        assert (report.source_count, report.target_count) == (1, 2)
        assert report.measured == {"relative_weight_change": ("w",)}
        session.models.erase(("renamed", 2))
        with Session(profile="ram") as imported:
            imported.models.add({"w": np.eye(2) * 2}, "renamed", step=2)
            session.utils.merge_other_session(imported)
        (replaced,) = session.compute.apply(
            "relative_weight_change", pairs=pairs, return_coverage=True
        ).pair_coverage
        assert replaced.target_extraction is None
        assert replaced.source_extraction == early
        assert (replaced.matched, replaced.target_only) == (("w",), ())


def test_measurement_counts_follow_fields_returned_by_the_kernel():
    with Session(profile="ram") as session:

        @session.compute.kernel(apply_level=KernelApplyLevel.CROSS_MODEL)
        def pair_norm_ratio(weights):
            source_norm = np.linalg.norm(weights[0])
            if source_norm == 0:
                return {}
            return {"pair_norm_ratio": np.linalg.norm(weights[1]) / source_norm}

        session.models.add({"zero": np.zeros((2, 2)), "unit": np.eye(2)}, "a")
        session.models.add({"zero": np.eye(2), "unit": 3 * np.eye(2)}, "b")
        (report,) = session.compute.apply(
            "pair_norm_ratio", pairs=[("a", "b")], return_coverage=True
        ).pair_coverage
        assert report.matched == ("unit", "zero")
        assert report.measured == {"pair_norm_ratio": ("unit",)}
        assert report.measurement_counts == {"pair_norm_ratio": 1}
        rows = session.results.export_aggregates(
            "pair_norm_ratio", export_format="list"
        )
        assert [(r["context_params"], r["value"]) for r in rows] == [
            (("unit",), pytest.approx(3.0))
        ]
