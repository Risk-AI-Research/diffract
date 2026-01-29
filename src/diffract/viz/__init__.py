"""Visualization utilities for publication-ready plots.

This module provides:

- **Theming**: Unified styling for consistent, publication-ready figures
- **Color Mapping**: Flexible color assignment by metadata keys
- **Plot Classes**: BoxPlot, ViolinPlot, ScatterPlot, etc.
- **Composition**: GridPlot, UpdateFigure for complex layouts

Quick Start:

    from diffract import Session
    from diffract.viz import render
    from diffract.viz.plots import BoxPlot
    from diffract.viz.themes import DEFAULT_THEME

    session = Session(...)
    plot = BoxPlot(field="stable_rank", theme=DEFAULT_THEME)
    fig = plot.render(session)
    fig.show()

Or using YAML configs:

    from diffract.viz import render_from_config

    fig = render_from_config(
        session=session,
        config_path="path/to/boxplot.yaml",
        theme_path="path/to/theme.yaml",
    )
"""

from diffract.core.utils import imports as import_utils

# Viz is not available without Plotly.
import_utils.require("plotly")

from .colors import ColorMapper, get_symbol
from .renderer import Plot, load_theme, render, render_from_config
from .themes import (
    DARK_THEME,
    DEFAULT_THEME,
    MINIMAL_THEME,
    Theme,
    apply_theme,
    theme_from_dict,
)

__all__ = [
    "DARK_THEME",
    "DEFAULT_THEME",
    "MINIMAL_THEME",
    "ColorMapper",
    "Plot",
    "Theme",
    "apply_theme",
    "get_symbol",
    "load_theme",
    "plots",
    "render",
    "render_from_config",
    "theme_from_dict",
]

from . import plots
