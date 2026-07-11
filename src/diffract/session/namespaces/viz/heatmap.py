"""Heatmap plot wrapper for Session.viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diffract.session.namespaces.viz._utils import _to_field_ref
from diffract.viz.data import FieldRef

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.styling import Theme


def heatmap(
    self: Any,
    *,
    z: str | FieldRef,
    x: str | FieldRef,
    y: str | FieldRef,
    title: str | None = None,
    value_filter: dict[str, tuple[str, Any]] | None = None,
    fill_value: float = float("nan"),
    show_text: bool = False,
    text_format: str = ".2f",
    text_font_size: int = 10,
    reverse_y: bool = True,
    x_title: str | None = None,
    x_showticklabels: bool = True,
    x_tickangle: int | None = None,
    x_tickfont_size: int | None = None,
    x_tickfont_family: str | None = None,
    x_tickfont_color: str | None = None,
    x_showgrid: bool = True,
    x_gridcolor: str | None = None,
    x_showline: bool = True,
    x_linecolor: str | None = None,
    x_categoryorder: str | None = None,
    x_categoryarray: list[str] | None = None,
    y_title: str | None = None,
    y_showticklabels: bool = True,
    y_tickangle: int | None = None,
    y_tickfont_size: int | None = None,
    y_tickfont_family: str | None = None,
    y_tickfont_color: str | None = None,
    y_showgrid: bool = True,
    y_gridcolor: str | None = None,
    y_showline: bool = True,
    y_linecolor: str | None = None,
    y_categoryorder: str | None = None,
    y_categoryarray: list[str] | None = None,
    heatmap_coloraxis_id: int | None = None,
    heatmap_colorscale: str = "Viridis",
    heatmap_showscale: bool = True,
    heatmap_cmin: float | None = None,
    heatmap_cmax: float | None = None,
    heatmap_colorbar_title: str | None = None,
    heatmap_coloraxis_override: bool = False,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Create a heatmap by pivoting a scalar field.

    Args:
        self: The bound viz namespace providing the draw() method.
        z: Field for cell values (field name or FieldRef).
        x: Field for columns (field name or FieldRef).
        y: Field for rows (field name or FieldRef).
        title: Figure title.
        value_filter: Filter entries by field values.
        fill_value: Value for missing (x, y) cells.
        show_text: Show cell values as text.
        text_format: Format string for text (e.g. ".2f").
        text_font_size: Font size for cell text.
        reverse_y: Reverse y-axis so first row is at top.
        x_title: X-axis title.
        x_showticklabels: Show x-axis tick labels.
        x_tickangle: Angle of x-axis tick labels.
        x_tickfont_size: Font size of x-axis tick labels.
        x_tickfont_family: Font family of x-axis tick labels.
        x_tickfont_color: Font color of x-axis tick labels.
        x_showgrid: Show x-axis grid lines.
        x_gridcolor: Color of x-axis grid lines.
        x_showline: Show the x-axis line.
        x_linecolor: Color of the x-axis line.
        x_categoryorder: Ordering strategy for x-axis categories.
        x_categoryarray: Explicit ordering of x-axis categories.
        y_title: Y-axis title.
        y_showticklabels: Show y-axis tick labels.
        y_tickangle: Angle of y-axis tick labels.
        y_tickfont_size: Font size of y-axis tick labels.
        y_tickfont_family: Font family of y-axis tick labels.
        y_tickfont_color: Font color of y-axis tick labels.
        y_showgrid: Show y-axis grid lines.
        y_gridcolor: Color of y-axis grid lines.
        y_showline: Show the y-axis line.
        y_linecolor: Color of the y-axis line.
        y_categoryorder: Ordering strategy for y-axis categories.
        y_categoryarray: Explicit ordering of y-axis categories.
        heatmap_coloraxis_id: Shared color axis id for the heatmap.
        heatmap_colorscale: Colorscale for the heatmap.
        heatmap_showscale: Show the heatmap colorbar.
        heatmap_cmin: Minimum value for heatmap color mapping.
        heatmap_cmax: Maximum value for heatmap color mapping.
        heatmap_colorbar_title: Title of the heatmap colorbar.
        heatmap_coloraxis_override: Override the shared heatmap color axis.
        theme: Theme to apply to the figure.
        theme_path: Path to a theme file to apply to the figure.

    Returns:
        Plotly Figure.

    Example:
        >>> session.viz.heatmap(z="stable_rank", x="head_id", y="layer_id")
    """
    from diffract.viz.plots.heatmap import HeatmapPlot

    plot = HeatmapPlot(
        z=_to_field_ref(z),
        x=_to_field_ref(x),
        y=_to_field_ref(y),
        title=title,
        value_filter=value_filter,
        fill_value=fill_value,
        show_text=show_text,
        text_format=text_format,
        text_font_size=text_font_size,
        reverse_y=reverse_y,
        x_title=x_title,
        x_showticklabels=x_showticklabels,
        x_tickangle=x_tickangle,
        x_tickfont_size=x_tickfont_size,
        x_tickfont_family=x_tickfont_family,
        x_tickfont_color=x_tickfont_color,
        x_showgrid=x_showgrid,
        x_gridcolor=x_gridcolor,
        x_showline=x_showline,
        x_linecolor=x_linecolor,
        x_categoryorder=x_categoryorder,
        x_categoryarray=x_categoryarray,
        y_title=y_title,
        y_showticklabels=y_showticklabels,
        y_tickangle=y_tickangle,
        y_tickfont_size=y_tickfont_size,
        y_tickfont_family=y_tickfont_family,
        y_tickfont_color=y_tickfont_color,
        y_showgrid=y_showgrid,
        y_gridcolor=y_gridcolor,
        y_showline=y_showline,
        y_linecolor=y_linecolor,
        y_categoryorder=y_categoryorder,
        y_categoryarray=y_categoryarray,
        heatmap_coloraxis_id=heatmap_coloraxis_id,
        heatmap_colorscale=heatmap_colorscale,
        heatmap_showscale=heatmap_showscale,
        heatmap_cmin=heatmap_cmin,
        heatmap_cmax=heatmap_cmax,
        heatmap_colorbar_title=heatmap_colorbar_title,
        heatmap_coloraxis_override=heatmap_coloraxis_override,
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)
