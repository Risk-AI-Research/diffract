"""Check ordered pair computations against singular-vector overlap identities."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from diffract.core.compute.execution.enums import KernelApplyLevel
from diffract.core.data.nn.aggregates.identity import aggregate_identity
from diffract.session import ScopeValidationError, Session

pytestmark = pytest.mark.unit

SYMMETRIC = {"l_agreement", "r_agreement", "avg_l_agreement", "avg_r_agreement"}
DIRECTED = {
    "l_overlap",
    "r_overlap",
    "max_l_agreement",
    "max_r_agreement",
    "avg_max_l_agreement",
    "avg_max_r_agreement",
    "relative_weight_change",
    "relative_spectrum_change",
}


def test_builtin_cross_model_kernel_symmetry(ram_container) -> None:
    registry = ram_container.compute_singleton.kernel_registry()
    cross = {
        k: registry.get_kernel_symmetry(k)
        for k in registry.list_kernels()
        if registry.get_kernel_apply_level(k) == KernelApplyLevel.CROSS_MODEL
    }
    assert cross == {**dict.fromkeys(SYMMETRIC, True), **dict.fromkeys(DIRECTED, False)}


def test_public_pair_grid_values_roles_and_dedup(ram_container) -> None:
    rng = np.random.default_rng(43)
    left, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    right, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    sigmas = np.diag([1.0, 2.0, 3.0, 5.0, 8.0])
    weights = {"z": sigmas, "a": left @ sigmas @ right.T}
    sess = Session(container=ram_container)

    with sess:
        for model, matrix in weights.items():
            sess.models.add({"w": matrix}, model_id=model)
        view = sess.filter(model_ids=["z", "a"])
        fields = sorted(DIRECTED | SYMMETRIC)
        view.compute.apply(*fields)
        repo = ram_container.nn.aggregate_repository()
        metadata = repo.metadata_index
        records = view.results.export_aggregates(*fields, export_format="list")
        assert len(records) == 2 * len(fields)
        by_key = {
            (r["field"], r["model_source"], r["model_target"]): r for r in records
        }
        assert len(by_key) == len(records)
        assert {r["step_source"] for r in records} == {0}
        assert {r["step_target"] for r in records} == {0}

        members = {r["model_id"]: r["uid"] for r in sess.models.parameters.list()}
        for source, target in permutations(weights, 2):
            change = np.linalg.norm(weights[target] - weights[source])
            assert by_key["relative_weight_change", source, target][
                "value"
            ] == pytest.approx(change / np.linalg.norm(weights[source]))
            assert by_key["relative_spectrum_change", source, target][
                "value"
            ] == pytest.approx(0, abs=1e-6)
            for side, matrix in [("l", left), ("r", right)]:
                overlap = np.abs(matrix if source == "z" else matrix.T)
                agreement = overlap.diagonal()
                best = overlap.max(axis=1)
                assert not np.allclose(best, overlap.max(axis=0))
                for field, expected in [
                    (f"{side}_overlap", overlap),
                    (f"{side}_agreement", agreement),
                    (f"max_{side}_agreement", best),
                    (f"avg_{side}_agreement", agreement.mean()),
                    (f"avg_max_{side}_agreement", best.mean()),
                ]:
                    np.testing.assert_allclose(
                        by_key[field, source, target]["value"], expected, atol=1e-12
                    )
                    uids = (members[source], members[target])
                    directed = field in DIRECTED
                    key = aggregate_identity(field, uids, directed=directed).uid
                    changed = metadata._conn.total_changes
                    assert (
                        repo.get_or_create(
                            field, member_uids=uids, directed=directed
                        ).meta.uid
                        == key
                    )
                    assert metadata._conn.total_changes == changed
                    if directed:
                        assert [
                            (row["base.role"], row["base.member_uid"])
                            for row in repo.member_rows(key)
                        ] == [("source", uids[0]), ("target", uids[1])]
        assert len(repo.list_uids()) == 2 * len(DIRECTED) + len(SYMMETRIC)
        changed = metadata._conn.total_changes
        sess.filter(model_ids=["a", "z"]).compute.apply(*fields)
        assert metadata._conn.total_changes == changed
        sess.compute.apply("frob_norm")
        wide = sess.results.export(
            "frob_norm", "avg_max_l_agreement", sources="all", export_format="dict"
        )
        assert set(wide) == set(members.values())
        for row in wide.values():
            assert sorted(
                float(v)
                for k, v in row["fields"].items()
                if k.startswith("avg_max_l_agreement")
            ) == pytest.approx(
                sorted(
                    float(by_key["avg_max_l_agreement", a, b]["value"])
                    for a, b in permutations(weights, 2)
                )
            )
        sess.models.rename("z", "renamed")
        renamed = sess.results.export_aggregates("l_overlap", export_format="list")
        assert {(r["model_source"], r["model_target"]) for r in renamed} == {
            ("a", "renamed"),
            ("renamed", "a"),
        }


def test_pair_validation_uses_checkpoint_endpoints(ram_container) -> None:
    session = Session(container=ram_container)
    with session:
        session.models.add({"w": np.eye(3)}, model_id="a")
        session.models.add({"w": np.eye(3)}, model_id="a", step=3)
        session.models.add({"v": np.eye(3)}, model_id="b")
        with pytest.raises(ScopeValidationError, match="scope contains 3"):
            session.filter(model_ids=["a", "b"]).compute.apply("l_overlap")
        assert ram_container.nn.aggregate_repository().list_uids() == []
        view = session.filter(model_ids=["a"], steps=[0, 3])
        view.compute.apply("l_overlap")
        records = view.results.export_aggregates("l_overlap", export_format="list")
        assert {
            (r["model_source"], r["step_source"], r["model_target"], r["step_target"])
            for r in records
        } == {("a", 0, "a", 3), ("a", 3, "a", 0)}
