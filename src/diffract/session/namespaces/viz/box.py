"""Box plot wrapper for Session.viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from diffract.session.namespaces.viz._utils import _to_field_ref
from diffract.viz.data import FieldRef

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.styling import Theme


def box(
    self: Any,
    *,
    y: str | FieldRef,
    x: str | FieldRef,
    title: str | None = None,
    value_filter: dict[str, tuple[str, Any]] | None = None,
    box_width: float = 0.5,
    boxpoints: Literal["all", "outliers", False] = "outliers",
    jitter_enabled: bool = False,
    jitter_width: float = 0.12,
    jitter_offset: float = -0.35,
    jitter_seed: int = 42,
    jitter_density_scale: bool = True,
    jitter_color: str | FieldRef | None = None,
    jitter_colorscale: str = "Viridis",
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
    marker_size: str | FieldRef | float | None = 6,
    marker_size_range: tuple[float, float] | None = None,
    marker_opacity: str | FieldRef | float | None = 0.7,
    marker_opacity_range: tuple[float, float] | None = None,
    marker_color: str | FieldRef | None = None,
    marker_symbol: str | FieldRef | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Create a box plot.

    Args:
        self: The bound viz namespace providing the draw() method.
        y: Field for vertical values (field name or FieldRef).
        x: Field for categories (field name or FieldRef).
        title: Figure title.
        value_filter: Filter entries by field values.
        box_width: Width of boxes.
        boxpoints: Show points: "all", "outliers", or False.
        jitter_enabled: Enable the jitter overlay of individual points.
        jitter_width: Horizontal spread of the jitter overlay.
        jitter_offset: Horizontal offset of the jitter overlay from the box.
        jitter_seed: Random seed for the jitter overlay.
        jitter_density_scale: Scale jitter spread by local point density.
        jitter_color: Field or color for jitter points.
        jitter_colorscale: Colorscale for jitter points.
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
        >>> session.viz.box(y="stable_rank", x="model_id", marker_color="layer_id")
    """
    from diffract.viz.plots.boxplot import BoxPlot

    plot = BoxPlot(
        y=_to_field_ref(y),
        x=_to_field_ref(x),
        title=title,
        value_filter=value_filter,
        box_width=box_width,
        boxpoints=boxpoints,
        jitter_enabled=jitter_enabled,
        jitter_width=jitter_width,
        jitter_offset=jitter_offset,
        jitter_seed=jitter_seed,
        jitter_density_scale=jitter_density_scale,
        jitter_color=_to_field_ref(jitter_color)
        if isinstance(jitter_color, str)
        else jitter_color,
        jitter_colorscale=jitter_colorscale,
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
        marker_color=_to_field_ref(marker_color)
        if isinstance(marker_color, str)
        else marker_color,
        marker_symbol=_to_field_ref(marker_symbol)
        if isinstance(marker_symbol, str)
        else marker_symbol,
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)
