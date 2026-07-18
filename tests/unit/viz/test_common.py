"""Unit tests for shared viz plot helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

from diffract.viz.data import get_field_value
from diffract.viz.data.types import EntryContext
from diffract.viz.plots.base import jitter as jitter_module

pytestmark = pytest.mark.unit


def test_density_scaled_jitter_shapes() -> None:
    y = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    jitter = np.ones_like(y)
    out = jitter_module.density_scaled_jitter(y=y, jitter=jitter)

    assert out.shape == jitter.shape
    assert np.all(out <= jitter)
    assert np.all(out >= 0.0)


def test_get_field_value_base_name() -> None:
    fields = {"stable_rank": 1.5, "frob_norm": 2.0}
    ctx = EntryContext(fields=fields, model_id=None, parameter_name=None)
    assert get_field_value(ctx, "stable_rank") == 1.5
    assert get_field_value(ctx, "frob_norm") == 2.0
    with pytest.raises(ValueError):
        get_field_value(ctx, "missing")


def test_get_field_value_contextual_field() -> None:
    fields = {"stable_rank@models[m1,m2]": 3.5}
    ctx = EntryContext(fields=fields, model_id=None, parameter_name=None)
    assert get_field_value(ctx, "stable_rank") == 3.5
    with pytest.raises(ValueError):
        get_field_value(ctx, "frob_norm")


def test_get_field_value_prefers_base_over_contextual() -> None:
    fields = {"stable_rank": 1.0, "stable_rank@models[m1]": 2.0}
    ctx = EntryContext(fields=fields, model_id=None, parameter_name=None)
    assert get_field_value(ctx, "stable_rank") == 1.0


def test_get_field_value_selects_matching_context() -> None:
    fields = {
        "agreement@models[m1,m2]@params[p1,p2,p3]": 1.0,  # broad context
        "agreement@models[m1]@params[p1]": 2.0,  # specific to m1/p1
    }
    # Without context info, should pick deterministically (smaller context first)
    ctx_none = EntryContext(fields=fields, model_id=None, parameter_name=None)
    assert get_field_value(ctx_none, "agreement") == 2.0

    # With matching context, should pick the one that includes this param
    ctx_m1_p1 = EntryContext(fields=fields, model_id="m1", parameter_name="p1")
    assert get_field_value(ctx_m1_p1, "agreement") == 2.0

    ctx_m1_p2 = EntryContext(fields=fields, model_id="m1", parameter_name="p2")
    assert get_field_value(ctx_m1_p2, "agreement") == 1.0


def test_get_field_value_prefers_specific_context() -> None:
    fields = {
        "metric@models[m1,m2,m3]": 1.0,  # larger context
        "metric@models[m1,m2]": 2.0,  # smaller context
    }
    # Both match m1, but smaller context is preferred
    ctx = EntryContext(fields=fields, model_id="m1", parameter_name=None)
    assert get_field_value(ctx, "metric") == 2.0


def test_get_field_value_contextual_request_matches_context_exactly() -> None:
    fields = {
        "metric@models[m1]@params[p1]": 1.0,
        "metric@models[m2]@params[p2]": 2.0,
    }
    ctx = EntryContext(fields=fields, model_id=None, parameter_name=None)
    # A contextual request is served only by keys carrying that exact context.
    assert get_field_value(ctx, "metric@models[m2]") == 2.0
    with pytest.raises(ValueError):
        get_field_value(ctx, "metric@models[m3]")


def test_get_field_value_ignores_non_contextual_suffix_keys() -> None:
    # A non-canonical suffix is an opaque name, never a variant of "metric".
    fields = {"metric@garbage": 42.0}
    ctx = EntryContext(fields=fields, model_id=None, parameter_name=None)
    assert get_field_value(ctx, "metric@garbage") == 42.0
    with pytest.raises(ValueError):
        get_field_value(ctx, "metric")


def test_extraction_reaches_contextual_fields_through_the_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the resolver unavailable, only exact keys stay reachable.
    from diffract.viz.data import extraction

    def _poisoned(address: object) -> object:
        raise AssertionError("resolver must not be bypassed")

    monkeypatch.setattr(extraction, "resolve", _poisoned)

    fields = {"stable_rank": 1.5, "agreement@models[m1,m2]": 3.0}
    ctx = EntryContext(fields=fields, model_id="m1", parameter_name=None)

    # Exact hits never parse anything.
    assert get_field_value(ctx, "stable_rank") == 1.5
    # Contextual fallback is impossible without the resolver.
    with pytest.raises(AssertionError, match="resolver"):
        get_field_value(ctx, "agreement")


def test_data_type_from_string() -> None:
    from diffract.viz.data import DataType

    assert DataType.from_string("numeric") is DataType.NUMERIC
    assert DataType.from_string("Categorical") is DataType.CATEGORICAL
    with pytest.raises(ValueError, match="blob"):
        DataType.from_string("blob")


def test_data_shape_from_string() -> None:
    from diffract.viz.data import DataShape

    assert DataShape.from_string("scalar") is DataShape.SCALAR
    assert DataShape.from_string("Vector") is DataShape.VECTOR
    with pytest.raises(ValueError, match="blob"):
        DataShape.from_string("blob")


def test_order_mode_from_string() -> None:
    from diffract.viz.data.ordering import OrderMode

    assert OrderMode.from_string("as_is") is OrderMode.AS_IS
    assert OrderMode.from_string("NUMERIC") is OrderMode.NUMERIC
    with pytest.raises(ValueError, match="blob"):
        OrderMode.from_string("blob")


def test_ordering_custom_mode_routes_by_priority() -> None:
    from diffract.viz.data.ordering import custom

    indices = custom(["b", "a", "c"]).argsort(["a", "b", "c"])

    assert indices.tolist() == [1, 0, 2]


def test_ordering_rejects_unhandled_mode() -> None:
    from diffract.viz.data.ordering import Ordering, OrderMode

    ordering = Ordering(mode=OrderMode.AS_IS)
    ordering.mode = object()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unhandled order mode"):
        ordering.argsort(["a", "b", "c"])
