"""Exercise display and selection through public export and plotting APIs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from diffract import FieldSelector, Session
from diffract.viz.data import DataProvider, FieldRef
from diffract.viz.plots import BoxPlot, ClusterBarChart, ScatterPlot

pytestmark = pytest.mark.unit


def test_lone_contextual_ref_and_export_select_exact_parameters(ram_container):
    with Session(container=ram_container) as session:
        session.models.add({k: np.eye(2) for k in ("0", "1", "2")}, model_id="m1")
        for names, value in [(("0", "2"), 3.5), (("1",), 15.0)]:
            session.results.ingest_aggregates(
                [
                    {
                        "field_name": "log_norm",
                        "context_models": ("m1",),
                        "context_params": names,
                        "value": value,
                    }
                ]
            )
        address = "log_norm@models[m1]@params[0,2]"
        (row,) = session.results.export_aggregates(address, export_format="list")
        assert row["value"] == 3.5
        assert {m["name"] for m in row["members"]} == {"0", "2"}
        provider = DataProvider(session)
        ref = FieldRef(address)
        assert provider.get_field_data(ref)[0] == [3.5, 3.5]
        assert {e["fields"]["name"] for e in provider.fetch([ref]).values()} == {
            "0",
            "2",
        }


def test_directed_pair_selection_plot_labels_and_filter(ram_container, monkeypatch):
    with Session(container=ram_container) as session:
        for name, step in (("a", 2), ("b", 10)):
            session.models.add({"w": np.eye(2)}, model_id=name, step=step)
        params = session.models.parameters.list()
        ids = {r["model_id"]: r["uid"] for r in params}
        repo = ram_container.nn.aggregate_repository()
        for left, right, score in [("a", "b", 3.0), ("b", "a", 9.0)]:
            agg = repo.get_or_create(
                "score", member_uids=(ids[left], ids[right]), directed=True
            )
            agg.set_field("value", score)
        bulk = repo.storage_manager.backend

        def unexpected_write(*_args, **_kwargs):
            pytest.fail("A display operation wrote a bulk value")

        for method in (
            "put_blob",
            "set_field",
            "move_blob",
            "delete_blob",
            "erase_obj",
            "erase_field",
            "clear",
        ):
            monkeypatch.setattr(bulk, method, unexpected_write)
        old = "score@models[b,a]@params[w]"
        both = session.results.export_aggregates(old, export_format="list")
        assert [(r["model_source"], r["model_target"], r["value"]) for r in both] == [
            ("a", "b", 3.0),
            ("b", "a", 9.0),
        ]
        provider = DataProvider(session)
        with pytest.raises(ValueError, match=r"FieldSelector.*roles=\('a', 'b'\)"):
            provider.get_field_data(FieldRef(old))
        assert set(provider.fetch(["step"], value_filter={old: (">", 5)})) == set(
            ids.values()
        )
        assert set(provider.fetch(["step"], value_filter={old: ("<", 5)})) == set(
            ids.values()
        )
        for rejected in (
            FieldSelector("score", roles=("a", "b"), steps=(10, 2)),
            FieldSelector("score", roles=("a", "a")),
        ):
            assert (
                session.results.export_aggregates(rejected, export_format="list") == []
            )
        assert provider.fetch(["step"], value_filter={old: (">", 10)}) == {}
        forward = FieldSelector("score", roles=("a", "b"), steps=(2, 10))
        (row,) = session.results.export_aggregates(forward, export_format="list")
        assert row["value"] == 3.0
        assert provider.get_field_data(forward)[0] == [3.0, 3.0]
        with pytest.raises(ValueError, match="Ambiguous directed field"):
            ClusterBarChart(field=old).render(session)
        chart = ClusterBarChart(field=forward, binning="linear").render(session)
        assert "source=a (step 2)" in chart.layout.xaxis.title.text
        plot = ScatterPlot(x=FieldRef("step"), y=FieldRef(forward))
        first = plot.render(session)
        assert list(first.data[0].y) == [3.0, 3.0]
        assert "source=a (step 2)" in first.layout.title.text
        session.models.set_display_label("a", 2, "warmup")
        assert "warmup" in plot.render(session).layout.title.text
        (updated,) = {
            key
            for e in provider.fetch([forward]).values()
            for key in e["field_contexts"]
        }
        assert (
            updated == "score [source=a (step 2, 'warmup') / w, target=b (step 10) / w]"
        )
        session.models.rename("a", "after")
        renamed = FieldSelector("score", roles=("after", "b"), steps=(2, 10))
        plot.y = FieldRef(renamed)
        assert (
            "source=after (step 2, 'warmup')" in plot.render(session).layout.title.text
        )
        wide = json.loads(session.results.export(renamed, export_format="json"))
        for entry in wide.values():
            (key,) = entry["fields"]
            assert "source=after (step 2, 'warmup')" in key
            assert entry["fields"][key] == 3.0


@pytest.mark.parametrize("marker_table", ["aggregates", "aggregate_members"])
def test_unknown_direction_from_hand_built_schema_remains_one_row(
    ram_container, marker_table
):
    with Session(container=ram_container) as session:
        for name in ["b", "a"]:
            session.models.add({"w": np.eye(2)}, model_id=name)
        repo = ram_container.nn.aggregate_repository()
        index = repo.metadata_index
        uid = "a" * 64
        with index:
            index.insert_rows(
                "aggregates",
                [
                    {
                        "uid": uid,
                        "field_name": "avg_l_agreement",
                        "json_data": json.dumps(
                            {"direction_unknown": marker_table == "aggregates"}
                        ),
                    }
                ],
            )
            index.insert_rows(
                "aggregate_members",
                [
                    {
                        "aggregate_uid": uid,
                        "member_uid": p["uid"],
                        "role": "member",
                        "json_data": json.dumps(
                            {"direction_unknown": marker_table == "aggregate_members"}
                        ),
                    }
                    for p in session.models.parameters.list()
                ],
            )
        repo.storage_manager.set_field(uid, "value", 0.75, table="aggregates")
        address = "avg_l_agreement@models[a,b]@params[w]"
        (row,) = session.results.export_aggregates(address, export_format="list")
        assert row["value"] == 0.75
        assert row["direction_unknown"]
        assert "model_source" not in row
        assert "model_target" not in row
        assert DataProvider(session).get_field_data(FieldRef(address))[0] == [
            0.75,
            0.75,
        ]
        forward = FieldSelector("avg_l_agreement", roles=("a", "b"))
        assert session.results.export_aggregates(forward, export_format="list") == []


def test_step_categories_and_metric_selector(ram_container):
    with Session(container=ram_container) as session:
        for step in (10, 0, 2):
            session.models.add({"w": np.eye(2) * (step + 1)}, model_id="m", step=step)
        session.models.add({"w": np.eye(2)}, model_id="a", step=10)
        session.compute.apply("frob_norm")
        figure = BoxPlot(x=FieldRef("step"), y=FieldRef("frob_norm")).render(session)
        assert [t.name for t in figure.data] == ["0", "2", "10"]
        by_step = {trace.name: sorted(trace.y) for trace in figure.data}
        assert by_step["0"] == pytest.approx([np.sqrt(2)])
        assert by_step["2"] == pytest.approx([np.sqrt(18)])
        assert by_step["10"] == pytest.approx([np.sqrt(2), np.sqrt(242)])
        only_two = FieldSelector("frob_norm", models=("m",), params=("w",), steps=(2,))
        values, _, _ = DataProvider(session).get_field_data(FieldRef(only_two))
        assert values == pytest.approx([np.sqrt(18)])


@pytest.mark.parametrize("output", ["list", "pandas", "polars"])
def test_flat_metrics_preserve_identity_and_escaped_metric_names(ram_container, output):
    if output != "list":
        pytest.importorskip(output)
    with Session(container=ram_container) as session:
        session.models.add(
            {"w": np.eye(2)}, model_id="m", step=2, display_label="warmup"
        )
        (param,) = session.models.parameters.list()
        session.results.ingest_metrics(
            {
                param["uid"]: {
                    "step": 99,
                    "field_step": 100,
                    "step_label": 101,
                    "instance_uid": 102,
                }
            }
        )
        data = session.results.export_metrics(
            "step", "field_step", "step_label", "instance_uid", export_format=output
        )
        if output == "pandas":
            data = data.to_dict("records")
        elif output == "polars":
            data = data.to_dicts()
        (row,) = data
        assert (row["step"], row["step_label"], row["instance_uid"]) == (
            2,
            "warmup",
            ram_container.nn.parameter_repository()
            .instances.get_by_name("m")
            .instance_uid,
        )
        assert (
            row["field_step"],
            row["field_field_step"],
            row["field_step_label"],
            row["field_instance_uid"],
        ) == (99, 100, 101, 102)


def test_symmetric_pair_has_one_value_for_an_unqualified_plot_slot(ram_container):
    with Session(container=ram_container) as session:
        for model in ("a", "b"):
            session.models.add({"w": np.diag([2.0, 1.0])}, model_id=model)
        session.compute.apply("avg_l_agreement")
        values, _, _ = DataProvider(session).get_field_data(
            FieldRef("avg_l_agreement@models[a,b]")
        )
        assert values == pytest.approx([1.0, 1.0])
