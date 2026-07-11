"""Tests for styling module (formerly themes)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_theme_defaults():
    """Theme should have sensible defaults."""
    from diffract.viz.styling import Theme

    theme = Theme()
    # In v0.2.0 the flat Theme fields moved into nested style objects.
    assert theme.layout.width is None
    assert theme.layout.height is None
    assert theme.typography.font_family == "Times New Roman"
    # `show_borders` is now expressed as the axes line visibility.
    assert theme.axes.show_line is True
    # `discrete_colormap` is now the color palette's list of colors.
    assert len(theme.palettes.color.colors) > 0


def test_predefined_themes():
    """Predefined themes should be valid Theme instances."""
    from diffract.viz.styling import (
        DARK_THEME,
        DEFAULT_THEME,
        MINIMAL_THEME,
        Theme,
    )

    assert isinstance(DARK_THEME, Theme)
    assert isinstance(MINIMAL_THEME, Theme)
    assert isinstance(DEFAULT_THEME, Theme)


def test_apply_theme() -> None:
    """apply_theme should modify figure layout."""
    import plotly.graph_objects as go

    from diffract.viz.styling import BackgroundStyle, LayoutStyle, Theme, apply_theme

    fig = go.Figure()
    # Dimensions and background are now configured via nested style objects.
    theme = Theme(
        layout=LayoutStyle(width=999, height=555),
        background=BackgroundStyle(plot_bgcolor="red"),
    )

    result = apply_theme(fig, theme)

    assert result is fig  # returns same object
    assert fig.layout.width == 999
    assert fig.layout.height == 555
    assert fig.layout.plot_bgcolor == "red"
