"""Refactored visualization stack for diffract."""

from .renderer import Plot, load_theme, render, render_from_config
from .styling import DARK_THEME, DEFAULT_THEME, MINIMAL_THEME, Theme, apply_theme

__all__ = [
    "DARK_THEME",
    "DEFAULT_THEME",
    "MINIMAL_THEME",
    "Plot",
    "Theme",
    "apply_theme",
    "load_theme",
    "plots",
    "render",
    "render_from_config",
]

from . import plots
