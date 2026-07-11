from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from diffract.viz.data import Entry, FieldRef
from diffract.viz.styling import (
    CategoricalPropertyResolver,
    ColorResolver,
    ColorSource,
    DefaultColorPalette,
    DefaultSymbolPalette,
    NumericPropertyResolver,
    ResolvedColor,
    SymbolSource,
    Theme,
)

from .coloraxis import SupportsColoraxis


@dataclass(kw_only=True)
class SupportsMarker(SupportsColoraxis("marker")):
    """Configurator mixin for marker properties."""

    marker_size: FieldRef | float | None = 6
    marker_size_range: tuple[float, float] | None = None
    marker_opacity: FieldRef | float | None = 0.7
    marker_opacity_range: tuple[float, float] | None = None
    marker_color: ColorSource = None
    marker_symbol: SymbolSource = None

    def configure(self, fig: go.Figure) -> None:
        """Apply resolved marker styling to matching traces in the figure.

        Args:
            fig: The Plotly figure whose marker traces are updated in place.
        """
        super().configure(fig)

        if self._traces_data is None:
            return

        coloraxis = self.resolve_coloraxis(fig)

        for trace_id, trace_data in self._traces_data.items():
            base_kwargs, per_point_kwargs = _build_marker_kwargs(
                trace_data,
                coloraxis,
            )

            if base_kwargs:
                fig.update_traces(
                    marker=base_kwargs,
                    selector=lambda t, tid=trace_id: (
                        t.meta and t.meta.get("trace_id") == tid
                    ),
                )

            if per_point_kwargs:
                fig.update_traces(
                    marker=per_point_kwargs,
                    selector=lambda t, tid=trace_id: (
                        t.meta
                        and t.meta.get("trace_id") == tid
                        and _supports_per_point_marker(t)
                    ),
                )

    def _resolve_marker_size(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> float | list[float] | None:
        match self.marker_size:
            case None:
                return None
            case int() | float():
                return float(self.marker_size)
            case FieldRef():
                resolver = NumericPropertyResolver(self.marker_size_range)
                return resolver.resolve(self.marker_size, entries)

    def _add_marker_size_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_size: float | list[float] | None,
    ) -> None:
        """Add resolved marker size to trace data."""
        if resolved_size is None:
            return
        trace_data["marker_size"] = resolved_size

    def _resolve_marker_opacity(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> float | list[float] | None:
        match self.marker_opacity:
            case None:
                return None
            case int() | float():
                return float(self.marker_opacity)
            case FieldRef():
                resolver = NumericPropertyResolver(self.marker_opacity_range)
                return resolver.resolve(self.marker_opacity, entries)

    def _add_marker_opacity_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_opacity: float | list[float] | None,
    ) -> None:
        """Add resolved marker opacity to trace data."""
        if resolved_opacity is None:
            return
        trace_data["marker_opacity"] = resolved_opacity

    def _resolve_marker_symbol(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> str | list[str] | None:
        match self.marker_symbol:
            case None:
                return None
            case str():
                return self.marker_symbol
            case FieldRef():
                symbol_palette = (
                    theme.palettes.symbols if theme else DefaultSymbolPalette()
                )
                resolver = CategoricalPropertyResolver(symbol_palette.symbols)
                return resolver.resolve(self.marker_symbol, entries)

    def _add_marker_symbol_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_symbol: str | list[str] | None,
    ) -> None:
        """Add resolved marker symbol to trace data."""
        if resolved_symbol is None:
            return
        trace_data["marker_symbol"] = resolved_symbol

    def _resolve_marker_color(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> ResolvedColor:
        color_palette = theme.palettes.color if theme else DefaultColorPalette()
        resolver = ColorResolver(color_palette)
        return resolver.resolve(self.marker_color, entries)

    def _add_marker_color_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_color: ResolvedColor,
    ) -> None:
        """Add resolved marker color to trace data."""
        if resolved_color.values is not None:
            trace_data["marker_color_values"] = resolved_color.values
        elif resolved_color.color is not None:
            trace_data["marker_color"] = resolved_color.color


def _build_marker_kwargs(
    trace_data: dict[str, Any],
    coloraxis: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build marker kwargs, split into universal and per-point.

    Returns:
        Tuple of ``(base_kwargs, per_point_kwargs)``.

        *base_kwargs* contains properties supported by all trace types
        (scalar size, scalar opacity, symbol, scalar/single color, line).

        *per_point_kwargs* contains list-valued properties and the
        ``coloraxis`` reference, which are only supported by
        scatter-type traces (not ``go.Box``, ``go.Violin``, etc.).
    """
    base_kwargs: dict[str, Any] = {}
    per_point_kwargs: dict[str, Any] = {}

    # Size — scalar goes to base, list goes to per-point
    size = trace_data.get("marker_size")
    if size is not None:
        if isinstance(size, list):
            per_point_kwargs["size"] = size
        else:
            base_kwargs["size"] = size

    # Opacity — scalar goes to base, list goes to per-point
    opacity = trace_data.get("marker_opacity")
    if opacity is not None:
        if isinstance(opacity, list):
            per_point_kwargs["opacity"] = opacity
        else:
            base_kwargs["opacity"] = opacity

    # Symbol
    symbol = trace_data.get("marker_symbol")
    if symbol is not None:
        base_kwargs["symbol"] = symbol

    # Color (split into base color value and coloraxis reference)
    color_base, coloraxis_kwargs = _build_marker_color_kwargs(trace_data, coloraxis)
    base_kwargs.update(color_base)
    per_point_kwargs.update(coloraxis_kwargs)

    return base_kwargs, per_point_kwargs


def _build_marker_color_kwargs(
    trace_data: dict[str, Any],
    coloraxis: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build marker color kwargs, split into base and per-point.

    Returns:
        Tuple of ``(base_kwargs, per_point_kwargs)``.

        Numeric ``color_values`` (continuous colorscale mapping) and
        list-valued categorical colors are routed to *per_point_kwargs*
        together with the ``coloraxis`` reference, since these are only
        supported by scatter-type traces.

        A single color string is placed in *base_kwargs* (safe for all
        trace types).
    """
    base_kwargs: dict[str, Any] = {}
    per_point_kwargs: dict[str, Any] = {}

    color = trace_data.get("marker_color")
    color_values = trace_data.get("marker_color_values")

    if color_values is not None:
        per_point_kwargs["color"] = color_values
        if coloraxis is not None:
            per_point_kwargs["coloraxis"] = coloraxis
    elif color is not None:
        if isinstance(color, list):
            per_point_kwargs["color"] = color
        else:
            base_kwargs["color"] = color

    return base_kwargs, per_point_kwargs


# --- Trace-type support ---

_PER_POINT_MARKER_TRACE_TYPES = frozenset(
    {
        "scatter",
        "scattergl",
        "scatter3d",
        "scattergeo",
        "scattermapbox",
        "scatterpolar",
        "scatterpolargl",
        "scatterternary",
        "splom",
        "barpolar",
    }
)


def _supports_per_point_marker(trace: Any) -> bool:
    """Check if a Plotly trace type supports per-point marker properties.

    Scatter-type traces accept list-valued ``marker.size``,
    ``marker.opacity``, ``marker.color``, and the ``marker.coloraxis``
    reference.  Aggregation traces like ``go.Box`` and ``go.Violin``
    only accept scalar values for these properties.
    """
    return getattr(trace, "type", None) in _PER_POINT_MARKER_TRACE_TYPES
