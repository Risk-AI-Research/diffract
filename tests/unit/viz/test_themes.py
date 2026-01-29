"""Tests for themes module."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_theme_defaults():
    """Theme should have sensible defaults."""
    from diffract.viz.themes import Theme

    theme = Theme()
    assert theme.width == 1200
    assert theme.height == 600
    assert theme.font_family == "Times New Roman"
    assert theme.show_borders is True
    assert len(theme.discrete_colormap) > 0


def test_predefined_themes():
    """Predefined themes should be valid Theme instances."""
    from diffract.viz.themes import (
        DARK_THEME,
        DEFAULT_THEME,
        MINIMAL_THEME,
        Theme,
    )

    assert isinstance(DARK_THEME, Theme)
    assert isinstance(MINIMAL_THEME, Theme)
    assert isinstance(DEFAULT_THEME, Theme)


def test_theme_from_dict():
    """theme_from_dict should create Theme from dict."""
    from diffract.viz.themes import Theme, theme_from_dict

    d = {"width": 800, "height": 400, "font_family": "Arial"}
    theme = theme_from_dict(d)

    assert isinstance(theme, Theme)
    assert theme.width == 800
    assert theme.height == 400
    assert theme.font_family == "Arial"


def test_theme_from_dict_ignores_unknown_keys():
    """theme_from_dict should ignore unknown keys."""
    from diffract.viz.themes import theme_from_dict

    d = {"width": 800, "unknown_key": "value"}
    theme = theme_from_dict(d)
    assert theme.width == 800
    assert not hasattr(theme, "unknown_key")


def test_apply_theme() -> None:
    """apply_theme should modify figure layout."""
    import plotly.graph_objects as go

    from diffract.viz.themes import Theme, apply_theme

    fig = go.Figure()
    theme = Theme(width=999, height=555, background_color="red")

    result = apply_theme(fig, theme)

    assert result is fig  # returns same object
    assert fig.layout.width == 999
    assert fig.layout.height == 555
    assert fig.layout.plot_bgcolor == "red"


def test_apply_theme_with_none() -> None:
    """apply_theme with None theme should use DEFAULT_THEME."""
    import plotly.graph_objects as go

    from diffract.viz.themes import DEFAULT_THEME, apply_theme

    fig = go.Figure()
    apply_theme(fig, None)

    assert fig.layout.width == DEFAULT_THEME.width
    assert fig.layout.height == DEFAULT_THEME.height
