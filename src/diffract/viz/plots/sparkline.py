from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean, stdev
from typing import Any, Literal

import plotly.graph_objects as go

from diffract.viz.data import (
    DataType,
    Entry,
    FieldRef,
    get_field_data,
    get_field_values,
)
from diffract.viz.plots.base.axis import AxisType, SupportsAxis
from diffract.viz.plots.base.line import SupportsLine
from diffract.viz.plots.base.marker import SupportsMarker
from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import (
    CategoricalPropertyResolver,
    NumericPropertyResolver,
    ResolvedColor,
)


@dataclass(kw_only=True)
class SparklinePlot(
    Plot,
    SupportsAxis("x", AxisType.NUMERIC | AxisType.CATEGORICAL),
    SupportsAxis("y", AxisType.NUMERIC),
    SupportsLine,
    SupportsMarker,
):
    """Line/sparkline plot for a scalar field as a function of another field."""

    y: FieldRef
    y_rescale_range: tuple[float, float] | None = None
    y_rescale_traces_separately: bool = False

    x: FieldRef
    x_rescale_range: tuple[float, float] | None = None
    x_rescale_traces_separately: bool = False

    group_by: FieldRef | None = None

    mode: Literal["lines", "markers", "lines+markers"] = "lines"
    show_bands: bool = True
    band_opacity: float = 0.3
    band_line_width: float = 0.5

    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        """Build trace data dict for each group (series)."""
        x_resolver = NumericPropertyResolver(self.x_rescale_range)
        y_resolver = NumericPropertyResolver(self.y_rescale_range)

        x_values = self._resolve_x_values(entries, x_resolver)
        y_values = y_resolver.resolve(self.y, entries)

        groups, group_indices = self._group_entries(entries, x_values, y_values)

        # 1) Resolve globally for all entries
        resolved_line_color = self._resolve_line_color(entries, self._theme)
        resolved_marker_color = self._resolve_marker_color(entries, self._theme)
        resolved_symbol = self._resolve_marker_symbol(entries, self._theme)
        resolved_dash = self._resolve_line_dash(entries, self._theme)
        resolved_size = self._resolve_marker_size(entries, self._theme)
        resolved_opacity = self._resolve_marker_opacity(entries, self._theme)
        resolved_width = self._resolve_line_width(entries, self._theme)

        group_keys = self._get_sorted_group_keys(groups)

        # 2) Build per-group traces
        traces_data: dict[str, dict[str, Any]] = {}

        for idx, group in enumerate(group_keys):
            points = groups[group]
            indices = group_indices[group]

            x_keys, means, stds = self._compute_stats(points)
            sort_idx = self.x.ordering.argsort(x_keys)
            x_keys, means, stds = zip(
                *[(x_keys[i], means[i], stds[i]) for i in sort_idx], strict=False
            )

            if self.x_rescale_traces_separately:
                x_keys = x_resolver._normalize(x_keys)

            if self.y_rescale_traces_separately:
                means_stds_normalized = y_resolver._normalize(means + stds)
                means = means_stds_normalized[: len(means)]
                stds = means_stds_normalized[len(means) :]

            trace_id = f"sparkline_{group}"
            trace_data: dict[str, Any] = {
                "group": group,
                "group_idx": idx,
                "x_keys": x_keys,
                "means": means,
                "stds": stds,
            }

            # 3) Pick per-group values and add to trace via mixin methods

            # --- line properties ---
            self._add_line_color_to_trace(
                trace_data,
                self._pick_group_color(indices, resolved_line_color),
            )
            self._add_line_dash_to_trace(
                trace_data,
                self._pick_group_scalar(indices, resolved_dash),
            )
            self._add_line_width_to_trace(
                trace_data,
                self._pick_group_numeric(indices, resolved_width),
            )

            # --- marker properties ---
            self._add_marker_color_to_trace(
                trace_data,
                self._pick_group_color(indices, resolved_marker_color),
            )
            self._add_marker_symbol_to_trace(
                trace_data,
                self._pick_group_scalar(indices, resolved_symbol),
            )
            self._add_marker_size_to_trace(
                trace_data,
                self._pick_group_numeric(indices, resolved_size),
            )
            self._add_marker_opacity_to_trace(
                trace_data,
                self._pick_group_numeric(indices, resolved_opacity),
            )

            traces_data[trace_id] = trace_data

        return traces_data

    def _resolve_x_values(
        self,
        entries: dict[str, Entry],
        x_resolver: NumericPropertyResolver,
    ) -> list[Any] | None:
        """Resolve x values with the resolver matching the effective axis mode."""
        mode = self.x_axis_mode
        if mode is None:
            data_type = self.x.data_type
            if data_type is None:
                _, data_type, _ = get_field_data(entries, self.x.field)
            mode = "categorical" if data_type == DataType.CATEGORICAL else "numeric"
        self._x_resolved_mode = mode

        if mode == "numeric":
            x_ref = replace(self.x, data_type=DataType.NUMERIC)
            return x_resolver.resolve(x_ref, entries)

        if self.x_rescale_range is not None or self.x_rescale_traces_separately:
            raise ValueError(
                f"x rescaling requires a numeric x axis, "
                f"but '{self.x.field}' resolved as categorical"
            )
        x_ref = replace(self.x, data_type=DataType.CATEGORICAL)
        return CategoricalPropertyResolver().resolve(x_ref, entries)

    def _group_entries(
        self,
        entries: dict[str, Entry],
        x_values: list[Any],
        y_values: list[Any],
    ) -> tuple[dict[str, list[tuple[Any, float]]], dict[str, list[int]]]:
        """Group (x, y) pairs by the group_by field.

        Returns:
            groups: mapping from group key to list of (x, y) pairs.
            group_indices: mapping from group key to original entry indices
                (only entries with non-None y).
        """
        if self.group_by is not None:
            group_values = get_field_values(entries, self.group_by.field)
        else:
            group_values = ["all"] * len(x_values)

        groups: dict[str, list[tuple[Any, float]]] = {}
        group_indices: dict[str, list[int]] = {}
        for i, (g_val, x_val, y_val) in enumerate(
            zip(group_values, x_values, y_values, strict=False)
        ):
            if y_val is None:
                continue
            key = str(g_val) if g_val is not None else "null"
            groups.setdefault(key, []).append((x_val, float(y_val)))
            group_indices.setdefault(key, []).append(i)
        return groups, group_indices

    def _get_sorted_group_keys(self, groups: dict[str, list[Any]]) -> list[str]:
        """Return group keys sorted lexicographically."""
        groups_keys = list(groups.keys())
        if self.group_by is None:
            return groups_keys
        order_indices = self.group_by.ordering.argsort(groups_keys)
        return [str(groups_keys[i]) for i in order_indices]

    @staticmethod
    def _compute_stats(
        points: list[tuple[Any, float]],
    ) -> tuple[list[Any], list[float], list[float]]:
        """Compute mean and std per unique x-value, preserving x order.

        Returns (x_keys, means, stds).
        """
        subgroups: dict[Any, list[float]] = {}
        for x_val, y_val in points:
            subgroups.setdefault(x_val, []).append(y_val)

        x_keys: list[Any] = list(subgroups.keys())
        means: list[float] = []
        stds: list[float] = []
        for x_val in x_keys:
            vals = subgroups[x_val]
            means.append(mean(vals) if len(vals) > 1 else vals[0])
            stds.append(stdev(vals) if len(vals) > 1 else 0.0)
        return x_keys, means, stds

    @staticmethod
    def _pick_group_color(
        indices: list[int],
        resolved_color: ResolvedColor,
    ) -> ResolvedColor:
        """Build a per-group ResolvedColor from a global ResolvedColor.

        For continuous values, slices to entries belonging to this group.
        For categorical per-entry colors, picks the color of the first
        entry in this group (within a group, all entries share the same
        categorical color).
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
        """Pick a single categorical value for a group.

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — return the value of the first
        entry belonging to this group.
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        if not resolved or not indices:
            return None
        return resolved[indices[0]]

    @staticmethod
    def _pick_group_numeric(
        indices: list[int],
        resolved: float | list[float] | None,
    ) -> float | None:
        """Pick a single numeric value for a group.

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — return the value of the first
        entry belonging to this group (line/sparkline properties like size,
        opacity, and width apply uniformly to the whole trace).
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        if not resolved or not indices:
            return None
        return resolved[indices[0]]

    def _build_figure(self) -> go.Figure:
        """Build the Plotly figure with scatter traces and optional bands."""
        fig = go.Figure()

        if self._traces_data is None:
            return fig

        for trace_id, trace_data in self._traces_data.items():
            group = trace_data["group"]
            x_keys = trace_data["x_keys"]
            means = trace_data["means"]
            stds = trace_data["stds"]

            if not x_keys:
                continue

            # Determine effective mode (add markers if symbol is set)
            effective_mode = self.mode
            if trace_data.get("marker_symbol") and "markers" not in effective_mode:
                effective_mode = (
                    "lines+markers" if "lines" in effective_mode else "markers"
                )

            # --- main trace ---
            fig.add_trace(
                go.Scatter(
                    x=x_keys,
                    y=means,
                    name=group,
                    mode=effective_mode,
                    legendgroup=group,
                    meta={"trace_id": trace_id},
                )
            )

            # --- std-band traces ---
            if self.show_bands and any(s > 0 for s in stds):
                self._add_band_traces(
                    fig,
                    trace_id,
                    group,
                    x_keys,
                    means,
                    stds,
                    trace_data,
                )

        # Title
        if self.title:
            fig.update_layout(title=self.title)
        else:
            fig.update_layout(title=f"{self.y.field} by {self.x.field}")

        return fig

    def _add_band_traces(
        self,
        fig: go.Figure,
        trace_id: str,
        group: str,
        x_keys: list[Any],
        means: list[float],
        stds: list[float],
        trace_data: dict[str, Any],
    ) -> None:
        """Add upper and lower std-band traces for a group."""
        band_trace_id_upper = f"{trace_id}_band_upper"
        band_trace_id_lower = f"{trace_id}_band_lower"

        line_color = trace_data.get("line_color")
        fill_color = self._compute_fill_color(line_color)

        upper_y = [m + s for m, s in zip(means, stds, strict=False)]
        lower_y = [m - s for m, s in zip(means, stds, strict=False)]

        fig.add_trace(
            go.Scatter(
                x=x_keys,
                y=upper_y,
                name=group,
                showlegend=False,
                mode="lines",
                line={"width": self.band_line_width, "color": line_color},
                legendgroup=group,
                meta={"trace_id": band_trace_id_upper},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_keys,
                y=lower_y,
                name=group,
                showlegend=False,
                mode="lines",
                fill="tonexty",
                fillcolor=fill_color,
                line={"width": self.band_line_width, "color": line_color},
                legendgroup=group,
                meta={"trace_id": band_trace_id_lower},
            )
        )

    def _compute_fill_color(self, band_color: str | None) -> str:
        """Compute RGBA fill color with transparency for the std band."""
        if band_color is None:
            return f"rgba(128,128,128,{self.band_opacity})"
        try:
            from matplotlib.colors import to_rgba

            r_, g_, b_, _ = to_rgba(band_color, alpha=self.band_opacity)
            return (
                f"rgba({round(r_ * 255)},{round(g_ * 255)},"
                f"{round(b_ * 255)},{self.band_opacity})"
            )
        except (ImportError, ValueError):
            return f"rgba(128,128,128,{self.band_opacity})"
