"""Aggregate identity across rename, export, partial views, and merge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from diffract.containers import create_main_container
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.data.metadata.errors import (
    CheckConstraintError,
    ForeignKeyViolationError,
    NotNullViolationError,
    UniqueConstraintError,
)
from diffract.session import ModelNotFoundError, Session, SessionError

from .test_session import _write_ram_config

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_reapply_after_rename_and_reopen_writes_no_aggregate_rows(
    tmp_path: Path,
) -> None:
    config = tmp_path / "persistent.ini"
    _write_ram_config(config)
    config.write_text(
        config.read_text().replace(
            "backend = ram",
            f"backend = sqlite\n\n[storage.sqlite]\npath = {tmp_path / 'values.db'}",
        )
    )
    calls = []

    def total(values):
        calls.append(len(values))
        return sum(float(v.sum()) for v in values)

    for initialize in (True, False):
        container = create_main_container(config)
        session = Session(container=container)
        registry = container.compute_singleton.kernel_registry()
        registry.register_kernel(
            name="weight_total",
            require_fields=("weights",),
            produce_fields=("weight_total",),
            implementation=total,
            apply_level=KernelApplyLevel.IN_MODEL,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=registry._split_signature(total)[1],
        )
        with session:
            repo = container.nn.aggregate_repository()
            index = repo.metadata_index
            if initialize:
                session.models.add(
                    {"a": np.eye(2), "b": np.ones((2, 2))}, model_id="before"
                )
                session.compute.apply("weight_total")
                [original_uid] = repo.list_uids()
                original_members = set(container.nn.parameter_repository().list_uids())
                session.models.rename("before", "after")
            changed = index._conn.total_changes
            session.compute.apply("weight_total")
            assert index._conn.total_changes == changed
            assert calls == [2]
            assert repo.list_uids() == [original_uid]
            proxy = repo.get_proxy(original_uid)
            assert proxy.meta.context_models == ("after",)
            assert proxy.get_field("value") == 6
            assert {
                (r["base.role"], r["base.member_uid"])
                for r in repo.member_rows(original_uid)
            } == {("member", uid) for uid in original_members}


def test_export_and_erase_follow_exact_sparse_membership(ram_container) -> None:
    session = Session(container=ram_container)
    with session:
        for model in ["m1", "m2"]:
            session.models.add({"a": np.eye(2), "b": np.eye(2)}, model_id=model)
        members = {
            (r["model_id"], r["name"]): r["uid"]
            for r in session.models.parameters.list()
        }
        session.results.ingest_metrics(
            {uid: {"baseline": 1} for uid in members.values()}
        )
        wanted = {members["m1", "a"], members["m2", "b"]}
        repo = ram_container.nn.aggregate_repository()
        agg = repo.get_or_create("pair_score", member_uids=wanted)
        agg.set_field("value", 2)
        exported = session.results.export(
            "baseline", "pair_score", sources="all", export_format="dict"
        )
        label = "pair_score [m1 (step 0) / a, m2 (step 0) / b]"
        assert set(exported) == set(members.values())
        assert {
            uid for uid, record in exported.items() if record["fields"].get(label) == 2
        } == wanted
        session.filter(model_ids=["m1"], param_names=["b"]).models.erase("m1")
        assert repo.get_proxy(agg.meta.uid).get_field("value") == 2
        session.filter(model_ids=["m1"], param_names=["a"]).models.erase("m1")
        assert repo.list_uids() == []
        assert session.models.list() == ["m2"]


def test_bridge_enforces_constraints_and_rejects_dangling_members(
    ram_container,
) -> None:
    session = Session(container=ram_container)
    with session:
        session.models.add({"w": np.eye(2)}, model_id="m")
        [uid] = ram_container.nn.parameter_repository().list_uids()
        repo = ram_container.nn.aggregate_repository()
        index = repo.metadata_index
        agg = repo.get_or_create("score", member_uids=[uid])
        row = {"aggregate_uid": agg.meta.uid, "member_uid": uid, "role": "member"}
        for overrides, error in [
            ({}, UniqueConstraintError),
            ({"role": None}, NotNullViolationError),
            ({"role": "other"}, CheckConstraintError),
            ({"member_type": "other"}, CheckConstraintError),
            ({"aggregate_uid": "absent"}, ForeignKeyViolationError),
            ({"member_uid": "absent"}, ForeignKeyViolationError),
        ]:
            with pytest.raises(error):
                index.insert_rows("aggregate_members", [row | overrides])
        with pytest.raises(ValueError, match="absent"):
            repo.get_or_create("invalid", member_uids=[uid, "absent"])
        assert repo.list_uids() == [agg.meta.uid]
        assert index._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_bad_import_and_erase_preserve_prior_values(ram_container) -> None:
    session = Session(container=ram_container)
    with session:
        session.models.add({"w": np.eye(2)}, model_id="existing")
        bound = {"field_name": "score", "context_models": ("existing",), "value": 4.0}
        missing = bound | {"context_params": ("missing",)}
        with pytest.raises(SessionError, match="missing"):
            session.results.ingest_aggregates([bound, missing])
        repo = ram_container.nn.aggregate_repository()
        assert repo.list_uids() == []
        with pytest.raises(SessionError, match="existing"):
            session.filter(param_names=["missing"]).results.ingest_aggregates([bound])
        session.results.ingest_aggregates([bound])
        with pytest.raises(ModelNotFoundError, match="absent"):
            session.models.erase("existing", "absent")
        [agg] = repo
        assert agg.get_field("value") == 4.0
        [param] = ram_container.nn.parameter_repository()
        np.testing.assert_array_equal(param.get_field("weights"), np.eye(2))


def test_merge_deduplicates_translated_member_uids(tmp_path: Path) -> None:
    sessions = []
    for name in ("source", "dest"):
        cfg = tmp_path / f"{name}.ini"
        _write_ram_config(cfg)
        container = create_main_container(cfg)
        session = Session(container=container)
        session.models.add({"w": np.eye(2)}, model_id="shared")
        sessions.append((session, container.nn.aggregate_repository()))
    (source, src_repo), (target, dst_repo) = sessions
    bound = {"field_name": "score", "context_models": ("shared",)}
    with source, target:
        source.results.ingest_aggregates([bound | {"value": 7}])
        target.results.ingest_aggregates([bound | {"value": 3}])
        [src_agg] = src_repo
        [dest_agg] = dst_repo
        assert src_agg.meta.member_uids != dest_agg.meta.member_uids
        for verify, expected in ((True, 3), (False, 7), (True, 7)):
            target.utils.merge_other_session(source, verify=verify)
            [agg] = dst_repo
            assert agg.meta.uid == dest_agg.meta.uid
            assert agg.meta.member_uids == dest_agg.meta.member_uids
            assert agg.get_field("value") == expected


@pytest.mark.parametrize("invalid", [{"context_models": None}, {"context_params": 5}])
def test_malformed_aggregate_is_rejected_before_writes(ram_container, invalid) -> None:
    session = Session(container=ram_container)
    session.models.add({"w": np.eye(2)}, model_id="m")
    row = {"field_name": "score", "context_models": ["m"], "value": 1.0}
    with pytest.raises(SessionError, match="sequence of names"):
        session.results.ingest_aggregates([row, row | invalid])
    with session:
        assert ram_container.nn.aggregate_repository().list_uids() == []
