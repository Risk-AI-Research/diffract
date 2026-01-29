"""Unit tests for shared viz plot helpers."""

from __future__ import annotations

import numpy as np
import pytest

from diffract.core.constants import parse_contextual_field_name
from diffract.viz.plots import common

pytestmark = pytest.mark.unit


def test_parse_contextual_field_name_base_only() -> None:
    base, models, params = parse_contextual_field_name("stable_rank")
    assert base == "stable_rank"
    assert models is None
    assert params is None


def test_parse_contextual_field_name_with_models() -> None:
    base, models, params = parse_contextual_field_name("metric@models[m1,m2]")
    assert base == "metric"
    assert models == ("m1", "m2")
    assert params is None


def test_parse_contextual_field_name_with_params() -> None:
    base, models, params = parse_contextual_field_name("metric@params[p1,p2,p3]")
    assert base == "metric"
    assert models is None
    assert params == ("p1", "p2", "p3")


def test_parse_contextual_field_name_full() -> None:
    base, models, params = parse_contextual_field_name(
        "agreement@models[m1,m2]@params[w1,w2]"
    )
    assert base == "agreement"
    assert models == ("m1", "m2")
    assert params == ("w1", "w2")


def test_as_float() -> None:
    assert common.as_float(None) is None
    assert common.as_float(1) == 1.0
    assert common.as_float(1.5) == 1.5
    assert common.as_float(np.int64(2)) == 2.0
    assert common.as_float(np.float64(2.25)) == 2.25

    scalar = np.asarray(3.0)
    assert common.as_float(scalar) == 3.0

    assert common.as_float("x") is None


def test_as_int() -> None:
    assert common.as_int(None) is None
    assert common.as_int(1) == 1
    assert common.as_int(np.int64(2)) == 2
    assert common.as_int(3.0) == 3
    assert common.as_int(3.1) is None

    scalar = np.asarray(4)
    assert common.as_int(scalar) == 4


def test_sort_key_numbers_before_strings() -> None:
    items = [10, "2", 1.5, "a", np.int64(3)]
    got = sorted(items, key=common.sort_key)
    assert got[:3] == [1.5, np.int64(3), 10] or got[:3] == [1.5, 3, 10]
    assert got[-2:] == ["2", "a"] or got[-2:] == ["a", "2"]


def test_density_scaled_jitter_shapes() -> None:
    y = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    jitter = np.ones_like(y)
    out = common.density_scaled_jitter(y=y, jitter=jitter)

    assert out.shape == jitter.shape
    assert np.all(out <= jitter)
    assert np.all(out >= 0.0)


def test_group_entries_by_and_extract_meta_value() -> None:
    results = {
        "p1": {"metadata": {"name": "w1", "model_id": "m1", "layer_id": 0}, "fields": {}},
        "p2": {"metadata": {"name": "w2", "model_id": "m1", "layer_id": 1}, "fields": {}},
    }

    grouped = common.group_entries_by(results, group_by="model_id")
    assert set(grouped.keys()) == {"m1"}
    assert len(grouped["m1"]) == 2

    grouped_all = common.group_entries_by(results, group_by="")
    assert set(grouped_all.keys()) == {"all"}

    e0 = results["p1"]
    assert common.extract_meta_value(e0, "parameter_name") == "w1"
    assert common.extract_meta_value(e0, "layer_id") == 0


def test_build_jitter_scatter_basic() -> None:
    xs = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    colors = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)

    trace = common.build_jitter_scatter(
        xs=xs,
        ys=ys,
        jitter_width=0.0,
        jitter_offset=0.0,
        density_scale=False,
        colors=colors,
        coloraxis="coloraxis",
        name="t",
    )

    assert trace["mode"] == "markers"
    assert trace["name"] == "t"
    assert trace["marker"]["coloraxis"] == "coloraxis"
    assert np.allclose(trace["x"], xs)


def test_collect_unique_meta_values_preserves_order() -> None:
    results = {
        "p1": {"metadata": {"model_id": "m1"}, "fields": {}},
        "p2": {"metadata": {"model_id": "m2"}, "fields": {}},
        "p3": {"metadata": {"model_id": "m1"}, "fields": {}},
    }
    assert common.collect_unique_meta_values(results, "model_id") == ["m1", "m2"]


def test_get_field_value_base_name() -> None:
    fields = {"stable_rank": 1.5, "frob_norm": 2.0}
    assert common.get_field_value(fields, "stable_rank") == 1.5
    assert common.get_field_value(fields, "frob_norm") == 2.0
    assert common.get_field_value(fields, "missing") is None


def test_get_field_value_contextual_field() -> None:
    fields = {"stable_rank@models[m1,m2]": 3.5}
    assert common.get_field_value(fields, "stable_rank") == 3.5
    assert common.get_field_value(fields, "frob_norm") is None


def test_get_field_value_prefers_base_over_contextual() -> None:
    fields = {"stable_rank": 1.0, "stable_rank@models[m1]": 2.0}
    assert common.get_field_value(fields, "stable_rank") == 1.0


def test_get_field_value_selects_matching_context() -> None:
    fields = {
        "agreement@models[m1,m2]@params[p1,p2,p3]": 1.0,  # broad context
        "agreement@models[m1]@params[p1]": 2.0,  # specific to m1/p1
    }
    # Without context info, should pick deterministically (smaller context first)
    assert common.get_field_value(fields, "agreement") == 2.0

    # With matching context, should pick the one that includes this param
    assert (
        common.get_field_value(fields, "agreement", model_id="m1", parameter_name="p1")
        == 2.0
    )
    assert (
        common.get_field_value(fields, "agreement", model_id="m1", parameter_name="p2")
        == 1.0
    )


def test_get_field_value_prefers_specific_context() -> None:
    fields = {
        "metric@models[m1,m2,m3]": 1.0,  # larger context
        "metric@models[m1,m2]": 2.0,  # smaller context
    }
    # Both match m1, but smaller context is preferred
    assert common.get_field_value(fields, "metric", model_id="m1") == 2.0


def test_list_contextual_field_values() -> None:
    fields = {
        "stable_rank": 1.0,
        "stable_rank@models[m1]": 2.0,
        "stable_rank@models[m2]": 3.0,
        "frob_norm": 4.0,
    }
    result = common.list_contextual_field_values(fields, "stable_rank")
    result_dict = dict(result)
    assert len(result) == 3
    assert result_dict["stable_rank"] == 1.0
    assert result_dict["stable_rank@models[m1]"] == 2.0
    assert result_dict["stable_rank@models[m2]"] == 3.0


def test_list_contextual_field_values_no_matches() -> None:
    fields = {"frob_norm": 1.0}
    result = common.list_contextual_field_values(fields, "stable_rank")
    assert result == []

