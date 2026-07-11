from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from diffract.viz.data import (
    Entry,
    FieldRef,
    get_field_values,
)
from diffract.viz.plots.base.axis import AxisType, SupportsAxis
from diffract.viz.plots.base.marker import SupportsMarker
from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import (
    NumericPropertyResolver,
    ResolvedColor,
)


@dataclass(kw_only=True)
class ScatterPlot(
    Plot,
    SupportsAxis("x", AxisType.NUMERIC),
    SupportsAxis("y", AxisType.NUMERIC),
    SupportsMarker,
):
    """Scatter plot for two numeric fields."""

    y: FieldRef
    y_rescale_range: tuple[float, float] | None = None
    y_rescale_traces_separately: bool = False

    x: FieldRef
    x_rescale_range: tuple[float, float] | None = None
    x_rescale_traces_separately: bool = False

    group_by: FieldRef | None = None

    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        """Build per-group trace data dicts."""
        x_resolver = NumericPropertyResolver(self.x_rescale_range)
        y_resolver = NumericPropertyResolver(self.y_rescale_range)

        x_values = x_resolver.resolve(self.x, entries)
        y_values = y_resolver.resolve(self.y, entries)

        groups, group_indices = self._group_entries(entries, x_values, y_values)

        # 1) Resolve marker properties globally for all entries
        resolved_color = self._resolve_marker_color(entries, self._theme)
        resolved_symbol = self._resolve_marker_symbol(entries, self._theme)
        resolved_size = self._resolve_marker_size(entries, self._theme)
        resolved_opacity = self._resolve_marker_opacity(entries, self._theme)

        group_keys = self._get_sorted_group_keys(groups)

        # 2) Build per-group traces
        traces_data: dict[str, dict[str, Any]] = {}

        for idx, group in enumerate(group_keys):
            points = groups[group]
            indices = group_indices[group]

            trace_x = [p[0] for p in points]
            if self.x_rescale_traces_separately:
                trace_x = x_resolver._normalize(trace_x)

            trace_y = [p[1] for p in points]
            if self.y_rescale_traces_separately:
                trace_y = y_resolver._normalize(trace_y)

            trace_id = f"scatter_{group}"
            trace_data: dict[str, Any] = {
                "group": group,
                "group_idx": idx,
                "x_values": trace_x,
                "y_values": trace_y,
            }

            # 3) Pick per-group values and add to trace via mixin methods
            self._add_marker_color_to_trace(
                trace_data,
                self._pick_group_color(indices, resolved_color),
            )
            self._add_marker_symbol_to_trace(
                trace_data,
                self._pick_group_scalar(indices, resolved_symbol),
            )
            self._add_marker_size_to_trace(
                trace_data,
                self._pick_group_values(indices, resolved_size),
            )
            self._add_marker_opacity_to_trace(
                trace_data,
                self._pick_group_values(indices, resolved_opacity),
            )

            traces_data[trace_id] = trace_data

        return traces_data

    def _group_entries(
        self,
        entries: dict[str, Entry],
        x_values: list[Any],
        y_values: list[Any],
    ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[int]]]:
        """Group (x, y) pairs by the ``group_by`` field.

        Returns:
            groups: mapping from group key to list of (x, y) pairs.
            group_indices: mapping from group key to original entry indices
                (only entries with non-None x *and* y).
        """
        if self.group_by is not None:
            group_values = get_field_values(entries, self.group_by.field)
        else:
            group_values = ["all"] * len(x_values)

        groups: dict[str, list[tuple[float, float]]] = {}
        group_indices: dict[str, list[int]] = {}

        for i, (g_val, x_val, y_val) in enumerate(
            zip(group_values, x_values, y_values, strict=False)
        ):
            if x_val is None or y_val is None:
                continue
            key = str(g_val) if g_val is not None else "null"
            groups.setdefault(key, []).append((float(x_val), float(y_val)))
            group_indices.setdefault(key, []).append(i)

        return groups, group_indices

    def _get_sorted_group_keys(self, groups: dict[str, list[Any]]) -> list[str]:
        """Return group keys sorted according to ``group_by`` ordering."""
        group_keys = list(groups.keys())
        if self.group_by is None:
            return group_keys
        order_indices = self.group_by.ordering.argsort(group_keys)
        return [str(group_keys[i]) for i in order_indices]

    @staticmethod
    def _pick_group_color(
        indices: list[int],
        resolved_color: ResolvedColor,
    ) -> ResolvedColor:
        """Build a per-group ResolvedColor from a global ResolvedColor.

        For continuous ``values``, slices to entries belonging to this group.
        For categorical per-entry colors, picks the first entry's color
        (within a group all entries share the same categorical color).
        For a single color string, passes through as-is.
        """
        if resolved_color.values is not None:
            filtered = [resolved_color.values[i] for i in indices]
            return ResolvedColor(values=filtered)

        if resolved_color.color is not None:
            if isinstance(resolved_color.color, list):
                if not resolved_color.color or not indices:
                    return ResolvedColor()
                return ResolvedColor(color=resolved_color.color[indices[0]])
            return resolved_color

        return resolved_color

    @staticmethod
    def _pick_group_scalar(
        indices: list[int],
        resolved: str | list[str] | None,
    ) -> str | None:
        """Pick a single categorical value for a group (first match).

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — return the first entry's value.
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        if not resolved or not indices:
            return None
        return resolved[indices[0]]

    @staticmethod
    def _pick_group_values(
        indices: list[int],
        resolved: float | list[float] | None,
    ) -> float | list[float] | None:
        """Slice a resolved numeric property for a group.

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — filter to entries in this group.
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        return [resolved[i] for i in indices]

    def _build_figure(self) -> go.Figure:
        """Build the Plotly figure with scatter traces."""
        fig = go.Figure()

        if self._traces_data is None:
            return fig

        for trace_id, trace_data in self._traces_data.items():
            group = trace_data["group"]
            x_vals = trace_data["x_values"]
            y_vals = trace_data["y_values"]

            if not x_vals:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    name=group,
                    mode="markers",
                    legendgroup=group,
                    meta={"trace_id": trace_id},
                )
            )

        # Title
        if self.title:
            fig.update_layout(title=self.title)
        else:
            fig.update_layout(title=f"{self.y.field} vs {self.x.field}")

        return fig
