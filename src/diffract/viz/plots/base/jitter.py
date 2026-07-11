"""Jitter overlay mixin for categorical plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from diffract.viz.data import Entry
from diffract.viz.styling import (
    ColorResolver,
    ColorSource,
    DefaultColorPalette,
    ResolvedColor,
)

from .coloraxis import SupportsColoraxis
from .overlay import Overlay

if TYPE_CHECKING:
    from diffract.viz.styling import Theme

_JITTER_TRACE_ID_SUFFIX = "_jitter"


def density_scaled_jitter(
    y: np.ndarray,
    jitter: np.ndarray,
    *,
    n_bins: int = 20,
) -> np.ndarray:
    """Scale jitter by local density to reduce overlap in dense regions.

    Points in denser regions get larger jitter spread, while isolated points
    stay closer to the center.

    Args:
        y: Y-values for the points.
        jitter: Base jitter values (uniform random).
        n_bins: Number of bins for density estimation.

    Returns:
        Scaled jitter values.
    """
    if y.size == 0:
        return jitter

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if y_min == y_max:
        return jitter

    bins = np.linspace(y_min, y_max, n_bins + 1)
    counts, _ = np.histogram(y, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    bin_indices = np.clip(
        np.digitize(y, bins) - 1,
        0,
        n_bins - 1,
    )
    scale = counts[bin_indices] / max_count

    return jitter * scale


@dataclass(slots=False, kw_only=True)
class SupportsJitter(Overlay, SupportsColoraxis("jitter")):
    """Mixin that adds jitter scatter overlay to categorical plots.

    Jitter displays individual data points with horizontal spread,
    useful for showing distribution density alongside box/violin plots.

    Creates ``go.Scatter`` traces with their own ``trace_id``
    (``{parent_trace_id}_jitter``).  Visual properties are applied
    in :meth:`configure`:

    - **Color**: from ``jitter_color`` if set, otherwise falls back
      to the parent trace's ``marker_color`` / ``marker_color_values``.
    - **Size, opacity, symbol**: inherited from the parent trace's
      ``marker_size``, ``marker_opacity``, ``marker_symbol``.

    Requires ``_traces_data`` to contain for each trace:
        - ``y_values``: list of y values for points
        - ``category_idx`` *or* ``jitter_x_center``: numeric x center

    Optional per-trace key:
        - ``jitter_xaxis``: Plotly x-axis reference for overlay points
          (for example ``"x2"`` for a hidden linear axis overlaid on a
          categorical ``x`` axis).
    """

    jitter_enabled: bool = False
    jitter_width: float = 0.12
    jitter_offset: float = -0.35
    jitter_seed: int = 42
    jitter_density_scale: bool = True
    jitter_color: ColorSource = None

    # --- Overlay: add geometry ------------------------------------------------

    def add_overlay_traces(self, fig: go.Figure) -> None:
        """Add jitter scatter traces if enabled."""
        if not self.jitter_enabled:
            return

        if self._traces_data is None:
            return

        self._configure_jitter_overlay_xaxes(fig)

        for trace_id, trace_data in self._traces_data.items():
            self._add_jitter_trace(fig, trace_id, trace_data)

    def _add_jitter_trace(
        self,
        fig: go.Figure,
        trace_id: str,
        trace_data: dict[str, Any],
    ) -> None:
        """Add a single jitter scatter trace for a category."""
        y_values = trace_data.get("y_values", [])
        if len(y_values) == 0:
            return

        x_center = _coerce_jitter_center(trace_data)
        if x_center is None:
            return

        y_arr = np.asarray(y_values, dtype=np.float64)
        n_points = len(y_arr)

        # Generate jittered x positions
        rng = np.random.default_rng(self.jitter_seed)
        jitter = rng.uniform(-self.jitter_width, self.jitter_width, size=n_points)

        if self.jitter_density_scale:
            jitter = density_scaled_jitter(y_arr, jitter)

        x_jittered = (
            np.full(n_points, x_center, dtype=np.float64) + self.jitter_offset + jitter
        )

        jitter_trace_id = f"{trace_id}{_JITTER_TRACE_ID_SUFFIX}"
        scatter_kwargs: dict[str, Any] = {
            "x": x_jittered,
            "y": y_arr,
            "mode": "markers",
            "showlegend": False,
            "meta": {"trace_id": jitter_trace_id},
        }

        jitter_xaxis = trace_data.get("jitter_xaxis")
        if isinstance(jitter_xaxis, str) and jitter_xaxis:
            scatter_kwargs["xaxis"] = jitter_xaxis

        fig.add_trace(go.Scatter(**scatter_kwargs))

    def _configure_jitter_overlay_xaxes(self, fig: go.Figure) -> None:
        """Ensure overlay x-axes exist for jitter traces that need them."""
        if self._traces_data is None:
            return

        centers_by_axis: dict[str, list[float]] = {}
        for trace_data in self._traces_data.values():
            jitter_xaxis = trace_data.get("jitter_xaxis")
            if not isinstance(jitter_xaxis, str) or jitter_xaxis == "x":
                continue

            center = _coerce_jitter_center(trace_data)
            if center is None:
                continue

            centers_by_axis.setdefault(jitter_xaxis, []).append(center)

        for axis_ref, centers in centers_by_axis.items():
            self._configure_single_overlay_xaxis(
                fig, axis_ref=axis_ref, centers=centers
            )

    def _configure_single_overlay_xaxis(
        self,
        fig: go.Figure,
        *,
        axis_ref: str,
        centers: list[float],
    ) -> None:
        """Configure one hidden linear axis overlaid on the primary x-axis."""
        if not axis_ref.startswith("x") or not centers:
            return

        suffix = axis_ref[1:]
        axis_layout_key = "xaxis" if not suffix else f"xaxis{suffix}"

        # Keep jitter points near category centers while allowing configured offsets.
        half_span = max(0.5, abs(self.jitter_offset) + self.jitter_width + 0.05)
        axis_range = [min(centers) - half_span, max(centers) + 0.5 * half_span]

        fig.update_layout(
            **{
                axis_layout_key: {
                    "overlaying": "x",
                    "showgrid": False,
                    "zeroline": False,
                    "showline": False,
                    "showticklabels": False,
                    "ticks": "",
                    "range": axis_range,
                }
            }
        )

    # --- Configurator: apply visuals ------------------------------------------

    def configure(self, fig: go.Figure) -> None:
        """Apply marker styling to jitter traces.

        Color is taken from ``jitter_color`` / ``jitter_color_values``
        in trace data if present, otherwise falls back to
        ``marker_color`` / ``marker_color_values``.  Size, opacity
        and symbol are always inherited from the parent marker.
        """
        super().configure(fig)  # SupportsColoraxis("jitter") layout

        if not self.jitter_enabled or self._traces_data is None:
            return

        coloraxis = self.resolve_coloraxis(fig)

        for trace_id, trace_data in self._traces_data.items():
            jitter_trace_id = f"{trace_id}{_JITTER_TRACE_ID_SUFFIX}"
            kwargs = _build_jitter_marker_kwargs(trace_data, coloraxis)

            if kwargs:
                fig.update_traces(
                    marker=kwargs,
                    selector=lambda t, jtid=jitter_trace_id: (
                        t.meta and t.meta.get("trace_id") == jtid
                    ),
                )

    # --- Resolution helpers (called by concrete plots in _build_traces_data) --

    def _resolve_jitter_color(
        self,
        entries: dict[str, Entry] | None,
        theme: Theme | None = None,
    ) -> ResolvedColor | None:
        """Resolve ``jitter_color`` source.

        Returns ``None`` when ``jitter_color`` is not set, signalling
        that :meth:`configure` should fall back to the parent's
        ``marker_color``.
        """
        if self.jitter_color is None:
            return None
        color_palette = theme.palettes.color if theme else DefaultColorPalette()
        resolver = ColorResolver(color_palette)
        return resolver.resolve(self.jitter_color, entries)

    def _add_jitter_color_to_trace(
        self,
        trace_data: dict[str, Any],
        resolved_color: ResolvedColor | None,
    ) -> None:
        """Add resolved jitter color to trace data.

        Vector values (numpy arrays) are flattened to match the
        observation-level expansion performed by ``_group_by_category``.
        """
        if resolved_color is None:
            return
        if resolved_color.values is not None:
            trace_data["jitter_color_values"] = _flatten_color_values(
                resolved_color.values,
            )
        elif resolved_color.color is not None:
            trace_data["jitter_color"] = resolved_color.color


# --- Private helpers ----------------------------------------------------------


def _build_jitter_marker_kwargs(
    trace_data: dict[str, Any],
    coloraxis: str | None,
) -> dict[str, Any]:
    """Build marker kwargs for a jitter trace.

    Reads jitter-specific color keys first, falling back to the parent
    trace's marker color when no jitter color is present.
    """
    kwargs: dict[str, Any] = {}

    # Inherit size, opacity, symbol from parent marker
    size = trace_data.get("marker_size")
    if size is not None:
        kwargs["size"] = size

    opacity = trace_data.get("marker_opacity")
    if opacity is not None:
        kwargs["opacity"] = opacity

    symbol = trace_data.get("marker_symbol")
    if symbol is not None:
        kwargs["symbol"] = symbol

    # Color: jitter-specific keys take priority over marker fallback
    jitter_color = trace_data.get("jitter_color")
    jitter_color_values = trace_data.get("jitter_color_values")

    if jitter_color_values is not None:
        kwargs["color"] = jitter_color_values
        if coloraxis is not None:
            kwargs["coloraxis"] = coloraxis
    elif jitter_color is not None:
        kwargs["color"] = jitter_color
    else:
        # Fallback to parent marker color.
        # Flatten potential arrays (vector fields stored by SupportsMarker).
        marker_color = trace_data.get("marker_color")
        marker_color_values = trace_data.get("marker_color_values")

        if marker_color_values is not None:
            kwargs["color"] = _flatten_color_values(marker_color_values)
            if coloraxis is not None:
                kwargs["coloraxis"] = coloraxis
        elif marker_color is not None:
            kwargs["color"] = marker_color

    return kwargs


def _flatten_color_values(values: list[Any]) -> list[float]:
    """Flatten per-entry color values, expanding numpy arrays.

    Scalar values are converted to ``float``.  Array values are
    ``ravel()``-ed and their elements appended individually, matching
    the observation-level expansion performed by ``_group_by_category``.

    If all values are already scalar, the output is equivalent to a
    plain ``[float(v) for v in values]``.
    """
    flat: list[float] = []
    for v in values:
        if isinstance(v, np.ndarray):
            flat.extend(v.ravel().tolist())
        elif v is None:
            flat.append(float("nan"))
        else:
            flat.append(float(v))
    return flat


def _coerce_jitter_center(trace_data: dict[str, Any]) -> float | None:
    """Extract a numeric x-center for jitter from trace data."""
    x_center = trace_data.get("jitter_x_center")
    if x_center is None:
        x_center = trace_data.get("category_idx")

    if x_center is None:
        return None

    try:
        return float(x_center)
    except (TypeError, ValueError):
        return None
