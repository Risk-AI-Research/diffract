"""Tests for colors module."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_color_mapper_defaults():
    """ColorMapper should have sensible defaults."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    assert mapper.theme is None
    assert isinstance(mapper.overrides, dict)


def test_get_colorscale_discrete():
    """get_colorscale should return discrete cmap for known discrete keys."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    cs = mapper.get_colorscale("model_id")
    assert isinstance(cs, list)
    assert len(cs) > 0


def test_get_colorscale_continuous():
    """Unknown keys should default to discrete colormap."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    cs = mapper.get_colorscale("layer_id")
    assert isinstance(cs, list)
    assert len(cs) > 0


def test_get_color_discrete():
    """get_color should cycle through discrete cmap."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    all_vals = ["m1", "m2", "m3"]

    c1 = mapper.get_color("model_id", "m1", all_vals)
    c2 = mapper.get_color("model_id", "m2", all_vals)
    c3 = mapper.get_color("model_id", "m3", all_vals)

    assert c1 != c2
    assert c2 != c3


def test_get_color_unknown_key_uses_discrete():
    """Unknown keys should default to discrete colormap."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    c = mapper.get_color("unknown_key", "val1", ["val1", "val2"])
    assert isinstance(c, str)


def test_color_mapper_with_theme():
    """ColorMapper should use theme's discrete_colormap."""
    from diffract.viz.colors import ColorMapper
    from diffract.viz.themes import Theme

    theme = Theme(discrete_colormap=["red", "blue"])
    mapper = ColorMapper(theme=theme)

    c1 = mapper.get_color("model_id", "m1", ["m1", "m2"])
    c2 = mapper.get_color("model_id", "m2", ["m1", "m2"])

    assert c1 == "red"
    assert c2 == "blue"


def test_color_mapper_with_overrides():
    """ColorMapper should respect overrides."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper(overrides={"model_id": ["green", "yellow"]})

    c1 = mapper.get_color("model_id", "m1", ["m1", "m2"])
    c2 = mapper.get_color("model_id", "m2", ["m1", "m2"])

    assert c1 == "green"
    assert c2 == "yellow"


def test_get_colors_for_values():
    """get_colors_for_values should return mapping for all values."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    values = ["a", "b", "a", "c"]
    colors = mapper.get_colors_for_values("model_id", values)

    assert len(colors) == 3  # unique values
    assert "a" in colors
    assert "b" in colors
    assert "c" in colors


def test_get_color_continuous() -> None:
    """get_color should interpolate for continuous keys."""
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper(overrides={"layer_id": "turbo"})
    all_vals = [0, 1, 2, 3, 4]

    c_min = mapper.get_color("layer_id", 0, all_vals)
    c_max = mapper.get_color("layer_id", 4, all_vals)

    # Should be different colors from the turbo colorscale
    assert isinstance(c_min, str)
    assert isinstance(c_max, str)
    assert c_min != c_max


def test_get_colorscale_override_string() -> None:
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper(overrides={"layer_id": "viridis"})
    assert mapper.get_colorscale("layer_id") == "viridis"


def test_interpolate_colorscale_named_colors() -> None:
    from diffract.viz.colors import ColorMapper

    mapper = ColorMapper()
    c = mapper._interpolate_colorscale([(0.0, "red"), (1.0, "blue")], 0.5)
    assert isinstance(c, str)
    assert c.startswith("rgb(")


def test_get_symbol():
    """get_symbol should cycle through symbols."""
    from diffract.viz.colors import get_symbol

    s0 = get_symbol(0)
    s1 = get_symbol(1)
    s7 = get_symbol(7)

    assert s0 == "circle"
    assert s1 == "square"
    assert s7 == "circle"  # wraps around
