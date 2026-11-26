"""Check long exports preserve exact joins, provenance and mixed value types."""

import json

import numpy as np
import pytest

from diffract import FieldSelector, Session

pytestmark = pytest.mark.unit


def test_long_export_joins_directed_members_and_provenance():
    with Session(profile="ram") as session:
        for step, weights in ((2, np.diag([1.0, 2.0])), (10, np.diag([2.0, 1.0]))):
            session.models.add({"w": weights}, model_id="model", step=step)
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        session.compute.apply("frob_norm", "l_overlap", "avg_l_agreement")
        forward = FieldSelector("l_overlap", roles=("model", "model"), steps=(2, 10))
        rows = session.results.export(
            "frob_norm", forward, shape="long", export_format="list"
        )
        wide = session.results.export_aggregates(forward, export_format="list")
        symmetric = session.results.export(
            "avg_l_agreement", shape="long", export_format="list"
        )

    parameters = {row["step"]: row for row in rows if row["entity_kind"] == "parameter"}
    aggregates = [row for row in rows if row["entity_kind"] == "aggregate"]
    assert set(parameters) == {2, 10}
    assert all(row["run_id"] for row in rows)
    assert [(r["member_role"], r["step"], r["parameter_uid"]) for r in aggregates] == [
        ("source", 2, parameters[2]["parameter_uid"]),
        ("target", 10, parameters[10]["parameter_uid"]),
    ]
    assert {r["aggregate_uid"] for r in aggregates} == {wide[0]["aggregate_uid"]}
    assert len({r["run_id"] for r in aggregates}) == 1
    assert aggregates[0]["value_shape"] == [2, 2]
    np.testing.assert_array_equal(
        np.reshape(aggregates[0]["value_array"], (2, 2)), wide[0]["value"]
    )
    assert len(symmetric) == 4
    assert len({row["aggregate_uid"] for row in symmetric}) == 1
    for row in symmetric:
        assert row["parameter_uid"] == parameters[row["step"]]["parameter_uid"]
        assert row["step"] == row[f"step_{row['member_role']}"]


def test_long_export_accepts_mixed_ingested_values_and_has_stable_polars_schema():
    pl = pytest.importorskip("polars")
    with Session(profile="ram") as session:
        session.models.add({"w": np.eye(2)}, model_id="model")
        uid = session.models.parameters.list()[0]["uid"]
        session.results.ingest_metrics(
            {uid: {"count": 2**60 + 1, "vec": np.arange(3.0), "name": "label"}}
        )
        fields = ("count", "vec", "name")
        records = json.loads(
            session.results.export(*fields, shape="long", export_format="json")
        )
        frame = session.results.export(*fields, shape="long", export_format="polars")
        empty = session.results.export(
            "frob_norm", shape="long", export_format="polars"
        )
    assert frame.to_dicts() == records
    assert frame.schema == empty.schema
    assert frame.filter(pl.col("field") == "count")["value_integer"].item() == 2**60 + 1
    assert frame.filter(pl.col("field") == "vec")["value_array"].to_list() == [
        [0.0, 1.0, 2.0]
    ]


def test_long_export_keeps_an_aggregate_without_members():
    with Session(profile="ram") as session:
        session.results.ingest_aggregates(
            [{"field_name": "global_score", "context_models": [], "value": 1}]
        )
        (row,) = session.results.export(
            "global_score", shape="long", export_format="list"
        )
    assert row["parameter_uid"] is None
    assert row["aggregate_uid"]
    assert row["run_id"]
    assert row["value_integer"] == 1


def test_long_json_encodes_nonfinite_numbers_as_null():
    with Session(profile="ram") as session:
        session.models.add({"w": np.eye(2)}, model_id="model")
        uid = session.models.parameters.list()[0]["uid"]
        session.results.ingest_metrics(
            {uid: {"scalar": float("nan"), "array": np.array([1.0, float("inf")])}}
        )
        payload = session.results.export(
            "scalar", "array", shape="long", export_format="json"
        )
    rows = {row["field"]: row for row in json.loads(payload)}
    assert rows["scalar"]["value_scalar"] is None
    assert rows["array"]["value_array"] == [1.0, None]
