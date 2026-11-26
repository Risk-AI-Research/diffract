"""Config-time string coercion: literals stay literal, field names become refs."""

from __future__ import annotations

import numpy as np
import pytest

from diffract import Session
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.sparkline import SparklinePlot
from diffract.viz.plots.violin import ViolinPlot
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


def test_wrapper_and_config_style_coercion_on_real_checkpoint_data() -> None:
    options = {
        "x": "model_id",
        "y": "weights_svals",
        "jitter_enabled": True,
        "jitter_color": "frob_norm",
        "jitter_seed": 4,
        "jitter_coloraxis_id": 1,
        "jitter_coloraxis_override": True,
        "jitter_colorbar_title": "Frobenius norm",
    }
    with Session(profile="ram") as session:
        for name, scale in (("a", 2), ("b", 3)):
            session.models.add({"w": np.diag([1.0, scale, 5.0])}, model_id=name)
        session.compute.apply("weights_svals", "frob_norm")
        wrapped = session.viz.violin(**options)
        configured = ViolinPlot(**options)
        _coerce_field_refs(configured)
        direct = configured.render(session)

    for figure in (wrapped, direct):
        assert figure.layout.coloraxis.colorbar.title.text == "Frobenius norm"
        jitter = [trace for trace in figure.data if trace.type == "scatter"]
        assert len(jitter) == 2
        assert all(len(trace.y) == 3 for trace in jitter)
        assert all(len(trace.marker.color) == len(trace.y) for trace in jitter)
