"""Tests for renderer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_render_basic(mock_session) -> None:
    """render should produce a figure."""
    from diffract.viz.plots.scalar import BoxPlot
    from diffract.viz.renderer import render

    plot = BoxPlot(field="stable_rank")
    fig = render(plot, session=mock_session)

    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)


def test_render_with_theme(mock_session) -> None:
    """render should apply theme when provided."""
    from diffract.viz.plots.scalar import BoxPlot
    from diffract.viz.renderer import render
    from diffract.viz.themes import Theme

    theme = Theme(width=777)
    plot = BoxPlot(field="stable_rank")
    fig = render(plot, session=mock_session, theme=theme)

    assert fig.layout.width == 777


def test_load_theme():
    """load_theme should create Theme from YAML."""
    from diffract.viz.renderer import load_theme

    yaml_content = """
width: 888
height: 444
font_family: "Helvetica"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        theme = load_theme(f.name)

    assert theme.width == 888
    assert theme.height == 444
    assert theme.font_family == "Helvetica"


def test_render_from_config_basic(mock_session) -> None:
    """render_from_config should load and render from YAML."""
    from diffract.viz.renderer import render_from_config

    # Use an existing config file
    config_path = _REPO_ROOT / "examples" / "configs" / "boxplot_stable_rank.yaml"

    if not config_path.exists():
        pytest.skip("Config file not found")

    fig = render_from_config(session=mock_session, config_path=config_path)

    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)


def test_render_from_config_with_theme_path(mock_session) -> None:
    """render_from_config should apply theme from theme_path."""
    from diffract.viz.renderer import render_from_config

    config_path = _REPO_ROOT / "examples" / "configs" / "boxplot_stable_rank.yaml"
    theme_path = _REPO_ROOT / "examples" / "configs" / "theme_example.yaml"

    if not config_path.exists() or not theme_path.exists():
        pytest.skip("Config or theme file not found")

    fig = render_from_config(
        session=mock_session, config_path=config_path, theme_path=theme_path
    )

    assert fig.layout.width == 800  # from theme_example.yaml
