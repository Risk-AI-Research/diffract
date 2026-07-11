from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from diffract.viz.data import Entry, FieldRef
from diffract.viz.styling import (
    CategoricalPropertyResolver,
    ColorResolver,
    ColorSource,
    DashSource,
    DefaultColorPalette,
    DefaultDashPalette,
    NumericPropertyResolver,
    ResolvedColor,
    Theme,
)

from .coloraxis import SupportsColoraxis


@dataclass(kw_only=True)
class SupportsLine(SupportsColoraxis("line")):
    """Configurator mixin for line properties.

    Trace data keys used (resolved in _build_traces_data):
        - line_width: resolved width values
        - line_color: resolved color (single color or array)
        - line_color_values: numeric values for coloraxis (if applicable)
        - line_dash: resolved dash pattern
    """

    line_width: FieldRef | float | None = 2
    line_width_range: tuple[float, float] | None = None
    line_color: ColorSource = None
    line_dash: DashSource = None
    line_shape: str | None = None
    line_smoothing: float | None = None

    def configure(self, fig: go.Figure) -> None:
        """Apply resolved line styling to matching traces in the figure.

        Args:
            fig: The Plotly figure whose line traces are updated in place.
        """
        super().configure(fig)

        if self._traces_data is None:
            return

        coloraxis = self.resolve_coloraxis(fig)

        for trace_id, trace_data in self._traces_data.items():
            kwargs = _build_line_kwargs(self, trace_data, coloraxis)

            if kwargs:
                fig.update_traces(
                    line=kwargs,
                    selector=lambda t, tid=trace_id: t.meta
                    and t.meta.get("trace_id") == tid,
                )

    def _resolve_line_width(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> float | list[float] | None:
        match self.line_width:
            case None:
                return None
            case int() | float():
                return float(self.line_width)
            case FieldRef():
                resolver = NumericPropertyResolver(self.line_width_range)
                return resolver.resolve(self.line_width, entries)

    def _add_line_width_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_line_width: float | list[float] | None,
    ) -> None:
        """Add resolved line width to trace data."""
        if resolved_line_width is None:
            return
        trace_data["line_width"] = resolved_line_width

    def _resolve_line_dash(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> str | list[str] | None:
        match self.line_dash:
            case None:
                return None
            case str():
                return self.line_dash
            case FieldRef():
                dash_palette = theme.palettes.dashes if theme else DefaultDashPalette()
                resolver = CategoricalPropertyResolver(list(dash_palette.dashes))
                return resolver.resolve(self.line_dash, entries)

    def _add_line_dash_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_line_dash: str | list[str] | None,
    ) -> None:
        """Add resolved line dash to trace data."""
        if resolved_line_dash is None:
            return
        trace_data["line_dash"] = resolved_line_dash

    def _resolve_line_color(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> ResolvedColor:
        color_palette = theme.palettes.color if theme else DefaultColorPalette()
        resolver = ColorResolver(color_palette)
        return resolver.resolve(self.line_color, entries)

    def _add_line_color_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_color: ResolvedColor,
    ) -> None:
        """Add resolved line color to trace data."""
        if resolved_color.values is not None:
            trace_data["line_color_values"] = resolved_color.values
        elif resolved_color.color is not None:
            trace_data["line_color"] = resolved_color.color


def _build_line_kwargs(
    line: SupportsLine,
    trace_data: dict[str, Any],
    coloraxis: str | None = None,
) -> dict[str, Any]:
    """Build line kwargs from pre-resolved trace data.

    Expected trace_data keys:
        - line_width: resolved width (number or array)
        - line_dash: resolved dash pattern (string or array)
        - line_color: resolved color (string, rgb, or array)
        - line_color_values: numeric values for coloraxis mapping
    """
    kwargs: dict[str, Any] = {}

    # Width
    width = trace_data.get("line_width")
    if width is not None:
        kwargs["width"] = width

    # Dash
    dash = trace_data.get("line_dash")
    if dash is not None:
        kwargs["dash"] = dash

    # Shape (static, from config)
    if line.line_shape is not None:
        kwargs["shape"] = line.line_shape

    # Smoothing (static, from config)
    if line.line_smoothing is not None:
        kwargs["smoothing"] = line.line_smoothing

    # Color
    color_kwargs = _build_line_color_kwargs(trace_data, coloraxis)
    kwargs.update(color_kwargs)

    return kwargs


def _build_line_color_kwargs(
    trace_data: dict[str, Any], coloraxis: str | None = None
) -> dict[str, Any]:
    """Build line color kwargs from pre-resolved trace data.

    Expected trace_data keys:
        - line_color: resolved color (string or array of colors)
        - line_color_values: numeric values for coloraxis mapping
    """
    kwargs: dict[str, Any] = {}

    color = trace_data.get("line_color")
    color_values = trace_data.get("line_color_values")

    if color_values is not None:
        kwargs["color"] = color_values
        if coloraxis is not None:
            kwargs["coloraxis"] = coloraxis

    elif color is not None:
        kwargs["color"] = color

    return kwargs
