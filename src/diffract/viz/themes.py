"""Unified theming system for publication-ready plots.

Provides a Theme dataclass with sensible defaults and predefined themes.
Use apply_theme() to apply a theme to any Plotly figure.

Example:
    from diffract.viz.themes import DEFAULT_THEME, apply_theme

    fig = some_plot.render(session)
    fig = apply_theme(fig, DEFAULT_THEME)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from diffract.core.utils import imports as import_utils

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]


@dataclass(slots=True)
class Theme:
    """Theme configuration for Plotly figures.

    Example:
        >>> theme = Theme(width=1000, height=500, font_family="Arial")
        >>> fig = session.draw(plot=my_plot, theme=theme)
        >>> # Or use predefined themes:
        >>> from diffract.viz.themes import DARK_THEME, MINIMAL_THEME
        >>> fig = apply_theme(fig, DARK_THEME)
    """

    # Figure dimensions
    width: int | None = None
    height: int | None = None

    # Fonts
    font_family: str = "Times New Roman"
    title_font_size: int = 16
    label_font_size: int = 14
    tick_font_size: int = 12

    # Colors
    background_color: str = "white"
    paper_bgcolor: str = "white"
    grid_color: str = "lightgrey"
    border_color: str = "black"

    # Legend
    legend_bgcolor: str = "rgba(255,255,255,0.9)"
    legend_border_color: str = "gray"
    legend_border_width: int = 1
    legend_font_size: int = 12

    # Axes
    show_borders: bool = True
    show_x_grid: bool = True
    show_y_grid: bool = True
    mirror_axes: bool = True

    # Margins
    margin: dict[str, int] = field(
        default_factory=lambda: {"l": 80, "r": 40, "t": 60, "b": 80}
    )

    # Colormaps for color_by
    discrete_colormap: list[str] = field(
        default_factory=lambda: [
            "navy",
            "crimson",
            "green",
            "chocolate",
            "orange",
            "violet",
            "purple",
            "blue",
            "grey",
        ]
    )

    # Symbols for marker differentiation
    marker_symbols: list[str] = field(
        default_factory=lambda: [
            "circle",
            "square",
            "triangle-up",
            "diamond",
            "cross",
            "x",
            "star",
        ]
    )

    # Line dash patterns
    line_dashes: list[str] = field(
        default_factory=lambda: [
            "solid",
            "dot",
            "dash",
            "dashdot",
            "longdash",
            "longdashdot",
        ]
    )

    # Colorbar positioning (for colorbars used with jitter coloring)
    colorbar_orientation: str = "h"  # "h" for horizontal, "v" for vertical
    colorbar_y: float = -0.15  # Position below plot
    colorbar_yanchor: str = "top"
    colorbar_x: float = 0.5  # Center horizontally
    colorbar_xanchor: str = "center"
    colorbar_thickness: int = 15
    colorbar_len: float = 0.5  # Length as fraction of plot

    # X-axis tick angle for long labels
    x_tickangle: int | None = None  # None = auto, or set e.g. -45, 90


# Predefined themes
DEFAULT_THEME = Theme(
    width=800,
    height=400,
    font_family="Times New Roman",
    title_font_size=16,
    label_font_size=14,
    tick_font_size=12,
    background_color="white",
    paper_bgcolor="white",
    grid_color="lightgrey",
    border_color="black",
    legend_bgcolor="rgba(255,255,255,0.9)",
    legend_border_color="gray",
    legend_border_width=1,
    show_borders=True,
    show_x_grid=True,
    show_y_grid=True,
    mirror_axes=True,
)

DARK_THEME = Theme(
    width=800,
    height=400,
    font_family="Arial",
    title_font_size=16,
    label_font_size=14,
    tick_font_size=12,
    background_color="#1e1e1e",
    paper_bgcolor="#2d2d2d",
    grid_color="#444444",
    border_color="#666666",
    legend_bgcolor="rgba(45,45,45,0.9)",
    legend_border_color="#666666",
    legend_border_width=1,
    show_borders=True,
    show_x_grid=True,
    show_y_grid=True,
    mirror_axes=False,
    discrete_colormap=[
        "#5dade2",
        "#f1948a",
        "#58d68d",
        "#f7dc6f",
        "#bb8fce",
        "#85c1e9",
        "#f8c471",
        "#abebc6",
        "#d7bde2",
    ],
)

MINIMAL_THEME = Theme(
    width=800,
    height=400,
    font_family="Arial",
    title_font_size=14,
    label_font_size=12,
    tick_font_size=10,
    background_color="white",
    paper_bgcolor="white",
    grid_color="white",
    border_color="lightgrey",
    legend_bgcolor="rgba(255,255,255,0)",
    legend_border_color="rgba(0,0,0,0)",
    legend_border_width=0,
    show_borders=False,
    show_x_grid=False,
    show_y_grid=False,
    mirror_axes=False,
    margin={"l": 50, "r": 20, "t": 40, "b": 50},
)


def apply_theme(fig: go.Figure, theme: Theme | None = None) -> go.Figure:
    """Apply a theme to a Plotly figure.

    Modifies figure layout including dimensions, fonts, colors, axes styling,
    legend, and margins. Safe to call multiple times.

    Args:
        fig: Plotly Figure to style.
        theme: Theme to apply. If None, uses DEFAULT_THEME.

    Returns:
        The same figure (mutated in place) for chaining.
    """
    import_utils.require("plotly.graph_objects")

    if theme is None:
        return fig

    # Apply axis styling to all x/y axes
    for attr in dir(fig.layout):
        if attr.startswith(("xaxis", "yaxis")):
            is_xaxis = attr.startswith("xaxis")
            axis_update: dict[str, Any] = {
                "tickfont": {"size": theme.tick_font_size, "family": theme.font_family},
                "title": {
                    "font": {
                        "size": theme.label_font_size,
                        "family": theme.font_family,
                    }
                },
                "gridcolor": theme.grid_color,
                "showline": theme.show_borders,
                "showgrid": theme.show_x_grid if is_xaxis else theme.show_y_grid,
                "linecolor": theme.border_color,
                "mirror": theme.mirror_axes,
            }
            # Apply tick angle only for x-axes if specified
            if is_xaxis and theme.x_tickangle is not None:
                axis_update["tickangle"] = theme.x_tickangle
            fig.update_layout({attr: axis_update})

    # Apply global layout settings
    if theme.width is not None:
        fig.update_layout(width=theme.width)
        
    if theme.height is not None:
        fig.update_layout(height=theme.height)
    
    fig.update_layout(
        title=dict(font=dict(size=theme.title_font_size, family=theme.font_family)),
        font=dict(size=theme.label_font_size, family=theme.font_family),
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.background_color,
        margin=theme.margin,
        legend=dict(
            font=dict(size=theme.legend_font_size, family=theme.font_family),
            bgcolor=theme.legend_bgcolor,
            bordercolor=theme.legend_border_color,
            borderwidth=theme.legend_border_width,
        ),
    )

    # Style annotations with consistent font
    fig.update_annotations(
        font=dict(family=theme.font_family, size=theme.label_font_size)
    )

    # Apply colorbar styling to any coloraxis
    colorbar_cfg = dict(
        orientation=theme.colorbar_orientation,
        y=theme.colorbar_y,
        yanchor=theme.colorbar_yanchor,
        x=theme.colorbar_x,
        xanchor=theme.colorbar_xanchor,
        thickness=theme.colorbar_thickness,
        len=theme.colorbar_len,
        tickfont=dict(size=theme.tick_font_size, family=theme.font_family),
    )
    for attr in dir(fig.layout):
        if attr.startswith("coloraxis"):
            coloraxis = getattr(fig.layout, attr)
            colorbar = getattr(coloraxis, "colorbar", None)
            if colorbar is None:
                coloraxis.update(dict(colorbar=colorbar_cfg))
            else:
                colorbar.update(colorbar_cfg)
                
    return fig


def theme_from_dict(d: dict[str, Any]) -> Theme:
    """Create a Theme from a dictionary (e.g., loaded from YAML)."""
    return Theme(**{k: v for k, v in d.items() if hasattr(Theme, k)})
