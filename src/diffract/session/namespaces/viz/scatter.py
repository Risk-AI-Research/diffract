"""Scatter plot wrapper for Session.viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diffract.session.namespaces.viz._utils import _to_field_ref
from diffract.viz.data import FieldRef

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.styling import Theme


def scatter(
    self: Any,
    *,
    x: str | FieldRef,
    y: str | FieldRef,
    group_by: str | FieldRef | None = None,
    title: str | None = None,
    value_filter: dict[str, tuple[str, Any]] | None = None,
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
    x_range: tuple[float, float] | None = None,
    x_dtick: float | str | None = None,
    x_tick0: float | None = None,
    x_tickformat: str | None = None,
    x_zeroline: bool = False,
    x_zerolinecolor: str | None = None,
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
    y_range: tuple[float, float] | None = None,
    y_dtick: float | str | None = None,
    y_tick0: float | None = None,
    y_tickformat: str | None = None,
    y_zeroline: bool = False,
    y_zerolinecolor: str | None = None,
    marker_coloraxis_id: int | None = None,
    marker_colorscale: str = "Viridis",
    marker_showscale: bool = True,
    marker_cmin: float | None = None,
    marker_cmax: float | None = None,
    marker_colorbar_title: str | None = None,
    marker_coloraxis_override: bool = False,
    marker_size: str | float | None = 6,
    marker_size_range: tuple[float, float] | None = None,
    marker_opacity: str | float | None = 0.7,
    marker_opacity_range: tuple[float, float] | None = None,
    marker_color: str | FieldRef | None = None,
    marker_symbol: str | FieldRef | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Create a scatter plot.

    Each entry becomes a point. group_by splits entries into separate traces.

    Args:
        self: The bound viz namespace providing the draw() method.
        x: Field for horizontal values (field name or FieldRef).
        y: Field for vertical values (field name or FieldRef).
        title: Figure title.
        value_filter: Filter entries by field values.
        group_by: Field to group points into traces (field name or FieldRef or None).
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
        x_range: X-axis value range.
        x_dtick: X-axis tick step.
        x_tick0: X-axis starting tick.
        x_tickformat: X-axis tick format string.
        x_zeroline: Show the x-axis zero line.
        x_zerolinecolor: Color of the x-axis zero line.
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
        y_range: Y-axis value range.
        y_dtick: Y-axis tick step.
        y_tick0: Y-axis starting tick.
        y_tickformat: Y-axis tick format string.
        y_zeroline: Show the y-axis zero line.
        y_zerolinecolor: Color of the y-axis zero line.
        marker_coloraxis_id: Shared color axis id for markers.
        marker_colorscale: Colorscale for markers.
        marker_showscale: Show the marker colorbar.
        marker_cmin: Minimum value for marker color mapping.
        marker_cmax: Maximum value for marker color mapping.
        marker_colorbar_title: Title of the marker colorbar.
        marker_coloraxis_override: Override the shared marker color axis.
        marker_size: Field or fixed size for markers.
        marker_size_range: Range to scale marker sizes into.
        marker_opacity: Field or fixed opacity for markers.
        marker_opacity_range: Range to scale marker opacity into.
        marker_color: Field or color for markers.
        marker_symbol: Field or symbol for markers.
        theme: Theme to apply to the figure.
        theme_path: Path to a theme file to apply to the figure.

    Returns:
        Plotly Figure.

    Example:
        >>> session.viz.scatter(
        ...     x="frob_norm",
        ...     y="stable_rank",
        ...     marker_color="model_id",
        ...     marker_size="layer_id",
        ... )
    """
    from diffract.viz.plots.scatter import ScatterPlot

    plot = ScatterPlot(
        x=_to_field_ref(x),
        y=_to_field_ref(y),
        group_by=_to_field_ref(group_by) if isinstance(group_by, str) else group_by,
        title=title,
        value_filter=value_filter,
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
        x_range=x_range,
        x_dtick=x_dtick,
        x_tick0=x_tick0,
        x_tickformat=x_tickformat,
        x_zeroline=x_zeroline,
        x_zerolinecolor=x_zerolinecolor,
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
        y_range=y_range,
        y_dtick=y_dtick,
        y_tick0=y_tick0,
        y_tickformat=y_tickformat,
        y_zeroline=y_zeroline,
        y_zerolinecolor=y_zerolinecolor,
        marker_coloraxis_id=marker_coloraxis_id,
        marker_colorscale=marker_colorscale,
        marker_showscale=marker_showscale,
        marker_cmin=marker_cmin,
        marker_cmax=marker_cmax,
        marker_colorbar_title=marker_colorbar_title,
        marker_coloraxis_override=marker_coloraxis_override,
        marker_size=_to_field_ref(marker_size)
        if isinstance(marker_size, str)
        else marker_size,
        marker_size_range=marker_size_range,
        marker_opacity=_to_field_ref(marker_opacity)
        if isinstance(marker_opacity, str)
        else marker_opacity,
        marker_opacity_range=marker_opacity_range,
        marker_color=marker_color,
        marker_symbol=marker_symbol,
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)
