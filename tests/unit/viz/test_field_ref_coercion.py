"""Config-time string coercion: literals stay literal, field names become refs."""

from __future__ import annotations

import pytest

from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.sparkline import SparklinePlot
from diffract.viz.renderer import _coerce_field_refs, _is_style_literal
from diffract.viz.styling.sources import StyleLiteralKind

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        ("#ff0000", StyleLiteralKind.COLOR, True),
        ("steelblue", StyleLiteralKind.COLOR, True),
        ("rgba(1,2,3,0.5)", StyleLiteralKind.COLOR, True),
        ("model_id", StyleLiteralKind.COLOR, False),
        ("layer_id", StyleLiteralKind.COLOR, False),
        ("circle", StyleLiteralKind.SYMBOL, True),
        ("star-open", StyleLiteralKind.SYMBOL, True),
        ("model_id", StyleLiteralKind.SYMBOL, False),
        ("dash", StyleLiteralKind.DASH, True),
        ("5px 10px", StyleLiteralKind.DASH, True),
        ("model_id", StyleLiteralKind.DASH, False),
    ],
)
def test_style_literal_detection(
    value: str, kind: StyleLiteralKind, expected: bool
) -> None:
    assert _is_style_literal(value, kind) is expected


def test_field_names_coerce_to_field_refs() -> None:
    plot = BoxPlot(
        y="stable_rank",
        x="model_id",
        marker_color="model_id",
        jitter_color="layer_id",
        marker_symbol="model_id",
    )

    _coerce_field_refs(plot)

    assert isinstance(plot.x, FieldRef)
    assert isinstance(plot.y, FieldRef)
    assert isinstance(plot.marker_color, FieldRef)
    assert plot.marker_color.field == "model_id"
    assert isinstance(plot.jitter_color, FieldRef)
    assert isinstance(plot.marker_symbol, FieldRef)


def test_valid_literals_stay_literal() -> None:
    plot = SparklinePlot(
        y="frob_norm",
        x="layer_id",
        line_color="black",
        line_dash="dash",
        marker_color="#1f77b4",
        marker_symbol="circle",
    )

    _coerce_field_refs(plot)

    assert plot.line_color == "black"
    assert plot.line_dash == "dash"
    assert plot.marker_color == "#1f77b4"
    assert plot.marker_symbol == "circle"


def test_literal_wins_over_identically_named_field() -> None:
    """Deterministic collision rule: a valid literal is never reinterpreted
    as a field, regardless of session content; an explicit FieldRef is the
    escape hatch."""
    plot = BoxPlot(y="stable_rank", x="model_id", marker_color="red")
    _coerce_field_refs(plot)
    assert plot.marker_color == "red"

    explicit = BoxPlot(
        y="stable_rank", x="model_id", marker_color=FieldRef(field="red")
    )
    _coerce_field_refs(explicit)
    assert isinstance(explicit.marker_color, FieldRef)


def test_typo_field_name_becomes_field_ref() -> None:
    """A string that is neither a literal nor an existing field turns into
    a FieldRef, so it fails at data lookup with the field name in the
    message instead of a cryptic plotly literal error."""
    plot = BoxPlot(y="stable_rank", x="model_id", jitter_color="layerr_id")
    _coerce_field_refs(plot)
    assert isinstance(plot.jitter_color, FieldRef)
    assert plot.jitter_color.field == "layerr_id"
