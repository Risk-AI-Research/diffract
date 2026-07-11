from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go

from diffract.core.data.nn.params.schema import ParameterType
from diffract.viz.data import Entry, EntryContext
from diffract.viz.data.extraction import get_field_value
from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import (
    AxesStyle,
    DefaultColorPalette,
    DefaultDashPalette,
    DefaultSymbolPalette,
)

if TYPE_CHECKING:
    from diffract.session import Session
    from diffract.viz.styling import Theme


@dataclass(kw_only=True)
class ClusterBarChart(Plot):
    """Clustered line chart for array-like fields (e.g. singular values).

    For each parameter, the array field is binned into ``num_bins`` histogram
    counts over the global value range (linear or exponential edges). Parameters
    are grouped by the ``group_by`` metadata keys, aggregated (mean, and std when
    ``draw_statistics``) across the parameters in each group, and drawn as one
    lines+markers trace per group.

    Use ``color_by`` / ``dash_by`` / ``marker_by`` to style traces by a metadata
    key. When ``draw_statistics`` is set, per-group std is drawn below zero.
    """

    field: str

    group_by: list[str] = field(default_factory=lambda: ["model_id"])
    aggregate_by: str | None = None

    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None

    num_bins: int = 20
    binning: Literal["linear", "exponential"] = "exponential"
    left_bound: float | None = None
    right_bound: float | None = None

    draw_statistics: bool = False

    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers"

    color_by: str | None = None
    dash_by: str | None = None
    marker_by: str | None = None

    legend_format: str | None = None
    legend_keys: list[str] | None = None

    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render the chart, applying plot-level parameter/model filtering."""
        if self._has_filter():
            ptypes = (
                [ParameterType.from_string(p) for p in self.parameter_types]
                if self.parameter_types
                else None
            )
            session = session.filter(
                param_ids=self.parameter_uids,
                param_names=self.parameter_names,
                param_types=ptypes,
                model_ids=self.model_ids,
            )
        return super().render(session, theme)

    def _has_filter(self) -> bool:
        return any(
            value is not None
            for value in (
                self.parameter_uids,
                self.parameter_names,
                self.parameter_types,
                self.model_ids,
            )
        )

    def _collect_fields_to_fetch(self) -> list[str]:
        return [self.field]

    def _group_keys(self) -> list[str]:
        group_by = list(self.group_by)
        if self.aggregate_by is not None and self.aggregate_by not in group_by:
            group_by.append(self.aggregate_by)
        return group_by

    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        """Bin arrays per parameter, group them, and aggregate per group."""
        group_by = self._group_keys()

        keyed_arrays, all_values = self._collect_arrays(entries, group_by)
        if not keyed_arrays:
            return {}

        edges = self._make_edges(all_values)

        grouped: dict[tuple[tuple[str, str], ...], list[np.ndarray]] = {}
        for key, arr in keyed_arrays:
            counts = self._binned_counts(arr, edges)
            if self.aggregate_by is not None:
                group_key = tuple(kv for kv in key if kv[0] != self.aggregate_by)
            else:
                group_key = key
            grouped.setdefault(group_key, []).append(counts)

        groups_sorted = sorted(grouped.items(), key=lambda kv: str(kv[0]))

        color_map = self._build_style_map(groups_sorted, self.color_by)
        dash_map = self._build_style_map(groups_sorted, self.dash_by)
        marker_map = self._build_style_map(groups_sorted, self.marker_by)

        color_palette = DefaultColorPalette()
        dash_palette = DefaultDashPalette()
        symbol_palette = DefaultSymbolPalette()

        # Numeric bin centers keep the x axis quantitative, so linear or
        # log axis types position the bins honestly.
        if self.binning == "exponential":
            x = np.sqrt(edges[:-1] * edges[1:])
        else:
            x = (edges[:-1] + edges[1:]) / 2.0

        traces_data: dict[str, dict[str, Any]] = {}
        for idx, (group_key, counts_list) in enumerate(groups_sorted):
            matrix = np.stack(counts_list, axis=0)
            trace_data: dict[str, Any] = {
                "x": x,
                "y_mean": np.nanmean(matrix, axis=0),
                "name": self._legend_name(group_key),
            }

            if self.draw_statistics and matrix.shape[0] > 1:
                trace_data["y_std"] = np.nanstd(matrix, axis=0)

            color = self._pick_style(group_key, self.color_by, color_map, color_palette)
            dash = self._pick_style(group_key, self.dash_by, dash_map, dash_palette)
            symbol = self._pick_style(
                group_key, self.marker_by, marker_map, symbol_palette
            )
            if color is not None:
                trace_data["line_color"] = color
            if dash is not None:
                trace_data["line_dash"] = dash
            if symbol is not None:
                trace_data["marker_symbol"] = symbol

            traces_data[f"cluster_{idx}"] = trace_data

        traces_data["__edges__"] = {"edges": edges}
        return traces_data

    def _collect_arrays(
        self,
        entries: dict[str, Entry],
        group_by: list[str],
    ) -> tuple[list[tuple[tuple[tuple[str, str], ...], np.ndarray]], list[float]]:
        keyed_arrays: list[tuple[tuple[tuple[str, str], ...], np.ndarray]] = []
        all_values: list[float] = []

        for entry in entries.values():
            ctx = EntryContext.from_entry(entry)
            arr = _as_1d_float_array(self._get_field(ctx))
            if arr.size == 0:
                continue

            key = self._group_key_for(ctx, group_by)
            if key is None:
                continue

            keyed_arrays.append((key, arr))
            all_values.extend(arr.tolist())

        return keyed_arrays, all_values

    def _get_field(self, ctx: EntryContext) -> Any:
        try:
            return get_field_value(ctx, self.field)
        except ValueError:
            return None

    @staticmethod
    def _group_key_for(
        ctx: EntryContext,
        group_by: list[str],
    ) -> tuple[tuple[str, str], ...] | None:
        kvs: list[tuple[str, str]] = []
        for key in group_by:
            value = ctx.fields.get(key)
            if value is None:
                return None
            kvs.append((key, str(value)))
        return tuple(kvs)

    def _make_edges(self, all_values: list[float]) -> np.ndarray:
        values = np.asarray(all_values, dtype=np.float64)
        left = (
            float(np.nanmin(values))
            if self.left_bound is None
            else float(self.left_bound)
        )
        right = (
            float(np.nanmax(values))
            if self.right_bound is None
            else float(self.right_bound)
        )
        if self.num_bins <= 0:
            raise ValueError("num_bins must be > 0")
        if right <= left:
            right = left + 1.0

        if self.binning == "linear":
            return np.linspace(left, right, self.num_bins + 1, dtype=np.float64)

        eps = 1e-12
        left = max(left, eps)
        right = max(right, left + eps)
        return np.geomspace(left, right, self.num_bins + 1, dtype=np.float64)

    @staticmethod
    def _binned_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=edges)
        return counts.astype(np.float64)

    @staticmethod
    def _build_style_map(
        groups_sorted: list[tuple[tuple[tuple[str, str], ...], Any]],
        style_key: str | None,
    ) -> list[str]:
        if style_key is None:
            return []
        values: list[str] = []
        for group_key, _ in groups_sorted:
            for k, v in group_key:
                if k == style_key and v not in values:
                    values.append(v)
        return values

    @staticmethod
    def _pick_style(
        group_key: tuple[tuple[str, str], ...],
        style_key: str | None,
        all_values: list[str],
        palette: Any,
    ) -> str | None:
        if style_key is None:
            return None
        for k, v in group_key:
            if k == style_key:
                if isinstance(palette, DefaultColorPalette):
                    return palette.get_color(v, all_values)
                candidates = (
                    palette.dashes
                    if isinstance(palette, DefaultDashPalette)
                    else palette.symbols
                )
                idx = all_values.index(v) if v in all_values else 0
                return candidates[idx % len(candidates)]
        return None

    def _legend_name(self, group_key: tuple[tuple[str, str], ...]) -> str:
        key_dict = dict(group_key)
        if self.legend_format:
            return self.legend_format.format(**key_dict)
        if self.legend_keys:
            return ", ".join(
                f"{k}={key_dict[k]}" for k in self.legend_keys if k in key_dict
            )
        return ", ".join(f"{k}={v}" for k, v in group_key)

    def _build_figure(self) -> go.Figure:
        """Build the Plotly figure with one lines+markers trace per group."""
        fig = go.Figure()

        if not self._traces_data:
            fig.update_layout(title=self.title or self.field)
            return fig

        for trace_id, trace_data in self._traces_data.items():
            if trace_id == "__edges__":
                continue

            line = {}
            if trace_data.get("line_color") is not None:
                line["color"] = trace_data["line_color"]
            if trace_data.get("line_dash") is not None:
                line["dash"] = trace_data["line_dash"]

            marker = {}
            if trace_data.get("marker_symbol") is not None:
                marker["symbol"] = trace_data["marker_symbol"]

            fig.add_trace(
                go.Scatter(
                    x=trace_data["x"],
                    y=trace_data["y_mean"],
                    mode=self.mode,
                    name=trace_data["name"],
                    line=line or None,
                    marker=marker or None,
                    legendgroup=trace_data["name"],
                    meta={"trace_id": trace_id},
                )
            )

            if "y_std" in trace_data:
                fig.add_trace(
                    go.Scatter(
                        x=trace_data["x"],
                        y=-trace_data["y_std"],
                        mode=self.mode,
                        name=f"{trace_data['name']} (std)",
                        showlegend=False,
                        line={"color": trace_data.get("line_color"), "dash": "dash"},
                        legendgroup=trace_data["name"],
                        meta={"trace_id": f"{trace_id}_std"},
                    )
                )

        if self.draw_statistics:
            fig.add_hline(y=0, line_width=1, line={"color": "lightgrey"})

        axes_style = self._theme.axes if self._theme else AxesStyle()
        axes_kwargs = {
            "showline": axes_style.show_line,
            "mirror": axes_style.mirror,
            "linecolor": axes_style.line_color,
            "showgrid": axes_style.show_grid,
            "gridcolor": axes_style.grid_color,
        }
        fig.update_xaxes(title=self.field, **axes_kwargs)
        fig.update_yaxes(**axes_kwargs)
        fig.update_layout(title=self.title or f"Cluster bar chart ({self.field})")
        return fig


def _as_1d_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.asarray([], dtype=np.float64)
        return value.astype(np.float64, copy=False).ravel()
    if isinstance(value, (list, tuple)):
        if not value:
            return np.asarray([], dtype=np.float64)
        try:
            return np.asarray(value, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            out: list[float] = []
            for item in value:
                with suppress(TypeError, ValueError):
                    out.append(float(item))
            return np.asarray(out, dtype=np.float64)
    try:
        return np.asarray([float(value)], dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)
