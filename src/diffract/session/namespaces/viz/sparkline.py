"""Line/sparkline plot wrapper for Session.viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from diffract.session.namespaces.viz._utils import _to_field_ref, _to_style_source
from diffract.viz.data import FieldRef
from diffract.viz.styling.sources import StyleLiteralKind

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.styling import Theme


def sparkline(
    self: Any,
    *,
    y: str | FieldRef,
    x: str | FieldRef,
    group_by: str | FieldRef | None = None,
    title: str | None = None,
    value_filter: dict[str, tuple[str, Any]] | None = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines",
    show_bands: bool = True,
    band_opacity: float = 0.3,
    band_line_width: float = 0.5,
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
    x_range: tuple[float, float] | None = None,
    x_dtick: float | str | None = None,
    x_tick0: float | None = None,
    x_tickformat: str | None = None,
    x_zeroline: bool = False,
    x_zerolinecolor: str | None = None,
    x_axis_mode: Literal["numeric", "categorical"] | None = None,
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
    line_coloraxis_id: int | None = None,
    line_colorscale: str = "Viridis",
    line_showscale: bool = True,
    line_cmin: float | None = None,
    line_cmax: float | None = None,
    line_colorbar_title: str | None = None,
    line_coloraxis_override: bool = False,
    line_width: str | float | None = 2,
    line_width_range: tuple[float, float] | None = None,
    line_color: str | FieldRef | None = None,
    line_dash: str | FieldRef | None = None,
    line_shape: str | None = None,
    line_smoothing: float | None = None,
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
    """Create a line plot of a field vs a metadata key.

    Automatically computes mean and std for duplicate x values.
    Each group (group_by) becomes a separate line trace.

    Args:
        y: Field for vertical values (field name or FieldRef).
        x: Field for horizontal axis (field name or FieldRef).
        title: Figure title.
        value_filter: Filter entries by field values.
        group_by: Field to group series by (field name or FieldRef or None).
        mode: "lines", "markers", or "lines+markers".
        show_bands: Show mean ± std bands.
        band_opacity: Opacity of std band fill.
        band_line_width: Line width for band edges.
        self: The bound viz namespace providing the draw() method.
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
        x_range: X-axis value range.
        x_dtick: X-axis tick step.
        x_tick0: X-axis starting tick.
        x_tickformat: X-axis tick format string.
        x_zeroline: Show the x-axis zero line.
        x_zerolinecolor: Color of the x-axis zero line.
        x_axis_mode: Force the x-axis data type ("numeric" or "categorical");
            inferred from the data when None.
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
        line_coloraxis_id: Shared color axis id for lines.
        line_colorscale: Colorscale for lines.
        line_showscale: Show the line colorbar.
        line_cmin: Minimum value for line color mapping.
        line_cmax: Maximum value for line color mapping.
        line_colorbar_title: Title of the line colorbar.
        line_coloraxis_override: Override the shared line color axis.
        line_width: Field or fixed width for lines.
        line_width_range: Range to scale line widths into.
        line_color: Field or color for lines.
        line_dash: Field or dash style for lines.
        line_shape: Line interpolation shape.
        line_smoothing: Line smoothing factor.
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
        >>> session.viz.line(y="stable_rank", x="in_model_idx", group_by="model_id")
    """
    from diffract.viz.plots.sparkline import SparklinePlot

    plot = SparklinePlot(
        y=_to_field_ref(y),
        x=_to_field_ref(x),
        group_by=_to_field_ref(group_by) if isinstance(group_by, str) else group_by,
        title=title,
        value_filter=value_filter,
        mode=mode,
        show_bands=show_bands,
        band_opacity=band_opacity,
        band_line_width=band_line_width,
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
        x_categoryorder=x_categoryorder,
        x_categoryarray=x_categoryarray,
        x_axis_mode=x_axis_mode,
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
        line_coloraxis_id=line_coloraxis_id,
        line_colorscale=line_colorscale,
        line_showscale=line_showscale,
        line_cmin=line_cmin,
        line_cmax=line_cmax,
        line_colorbar_title=line_colorbar_title,
        line_coloraxis_override=line_coloraxis_override,
        line_width=_to_field_ref(line_width)
        if isinstance(line_width, str)
        else line_width,
        line_width_range=line_width_range,
        line_color=_to_style_source(line_color, StyleLiteralKind.COLOR),
        line_dash=_to_style_source(line_dash, StyleLiteralKind.DASH),
        line_shape=line_shape,
        line_smoothing=line_smoothing,
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
        marker_color=_to_style_source(marker_color, StyleLiteralKind.COLOR),
        marker_symbol=_to_style_source(marker_symbol, StyleLiteralKind.SYMBOL),
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)
