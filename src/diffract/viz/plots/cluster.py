"""Cluster bar chart (ported conceptually from notebooks_src).

This plot is intended for array-like fields (e.g. singular values). It bins the
values and draws a clustered line chart (lines+markers) for multiple groups.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.plots.common import fetch_data, get_field_value

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


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
            for x in value:
                with suppress(TypeError, ValueError):
                    out.append(float(x))
            return np.asarray(out, dtype=np.float64)
        except Nonetry:
            return np.asarray([float(value)], dtype=np.float64)
        except (TypeError, ValueError):
            return np.asarray([], dtype=np.float64)


def _make_edges(
    *,
    left: float,
    right: float,
    num_bins: int,
    binning: Literal["linear", "exponential"],
) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    if right <= left:
        raise ValueError("right_bound must be > left_bound")

    if binning == "linear":
        return np.linspace(left, right, num_bins + 1, dtype=np.float64)

    # exponential / log-like
    eps = 1e-12
    left_ = max(float(left), eps)
    right_ = max(float(right), left_ + eps)
    return np.geomspace(left_, right_, num_bins + 1, dtype=np.float64)


def _binned_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    return hist.astype(np.float64)


@dataclass(slots=True)
class ClusterBarChart:
    """Cluster bar chart for array-like fields.

    For each parameter, we compute a binned vector (by default: counts per bin).
    Then we group parameters by `group_by` metadata keys and aggregate (mean, and
    optionally std) across parameters in the group.

    Use `color_by` to color lines by a metadata key (defaults to group key).

    Example:
        >>> plot = ClusterBarChart(
        ...     field="esd",
        ...     group_by=["model_id"],
        ...     aggregate_by="model_id",
        ...     num_bins=50,
        ... )
        >>> fig = session.draw(plot=plot)
    """

    field: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None

    # Grouping
    group_by: list[str] = None  # type: ignore[assignment]
    aggregate_by: str | None = None

    # Filtering
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None

    # Binning
    num_bins: int = 20
    binning: Literal["linear", "exponential"] = "exponential"
    left_bound: float | None = None
    right_bound: float | None = None

    # Aggregation and statistics
    draw_statistics: bool = False

    # Trace rendering
    mode: str = "lines+markers"
    cluster_width: float = 0.9

    # Color customization
    color_by: str | None = None

    # Symbol/dash customization (applies per trace)
    marker_by: str | None = None
    dash_by: str | None = None

    # Legend formatting
    legend_format: str | None = None  # e.g. "{model_id}, L{layer_id}" for shorter names
    legend_keys: list[str] | None = (
        None  # Keys to include in legend name (subset of group_by)
    )

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the cluster bar chart using data from the session."""
        go = import_utils.require("plotly.graph_objects")
        from diffract.viz.colors import ColorMapper
        from diffract.viz.themes import apply_theme

        group_by = ["model_id"] if self.group_by is None else list(self.group_by)

        if self.aggregate_by is not None and self.aggregate_by not in group_by:
            group_by.append(self.aggregate_by)

        ptypes = (
            [ParameterType.from_string(p) for p in self.parameter_types]
            if self.parameter_types
            else None
        )

        results = fetch_data(
            session,
            [self.field],
            parameter_uids=self.parameter_uids,
            parameter_names=self.parameter_names,
            parameter_types=ptypes,
            model_ids=self.model_ids,
        )

        # First pass: collect arrays and decide bin edges.
        entries: list[tuple[tuple[tuple[str, str], ...], np.ndarray]] = []
        all_values: list[float] = []

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            arr = _as_1d_float_array(get_field_value(fields, self.field, **ctx))
            if arr.size == 0:
                continue

            # Extract group key values.
            kvs: list[tuple[str, str]] = []
            ok = True
            for k in group_by:
                v = meta.get("name") if k == "parameter_name" else meta.get(k)
                if v is None:
                    ok = False
                    break
                kvs.append((k, str(v)))
            if not ok:
                continue

            key = tuple(kvs)
            entries.append((key, arr))
            all_values.extend(arr.tolist())

        if not entries:
            fig = go.Figure()
            fig.update_layout(title=self.title or self.field)
            return apply_theme(fig, self.theme)

        values_np = np.asarray(all_values, dtype=np.float64)
        left = (
            float(np.nanmin(values_np))
            if self.left_bound is None
            else float(self.left_bound)
        )
        right = (
            float(np.nanmax(values_np))
            if self.right_bound is None
            else float(self.right_bound)
        )
        edges = _make_edges(
            left=left, right=right, num_bins=self.num_bins, binning=self.binning
        )

        # Second pass: build per-parameter y vectors and group them.
        grouped: dict[tuple[tuple[str, str], ...], list[np.ndarray]] = {}
        for key, arr in entries:
            y = _binned_counts(arr, edges)

            if self.aggregate_by is not None:
                group_key = tuple((k, v) for (k, v) in key if k != self.aggregate_by)
            else:
                group_key = key

            grouped.setdefault(group_key, []).append(y)

        groups_sorted = sorted(grouped.items(), key=lambda kv: str(kv[0]))
        n_groups = len(groups_sorted)
        offsets = (
            np.linspace(0.0, self.cluster_width, n_groups + 2, dtype=np.float64)[1:-1]
            if n_groups > 1
            else np.asarray([0.0], dtype=np.float64)
        )

        x = np.arange(edges.size - 1, dtype=np.float64)
        x_labels = np.round(edges, 4)

        color_mapper = ColorMapper(theme=self.theme)

        # Determine what to color by
        color_key = self.color_by
        if color_key is None and group_by:
            # Default: color by first group key
            color_key = group_by[0]

        # Collect all unique values for color_by
        all_color_vals = []
        if color_key:
            for key, _ in groups_sorted:
                for k, v in key:
                    if k == color_key and v not in all_color_vals:
                            all_color_vals.append(v)

        # Collect unique values for marker_by / dash_by
        all_marker_vals: list[str] = []
        if self.marker_by:
            for key, _ in groups_sorted:
                for k, v in key:
                    if k == self.marker_by and v not in all_marker_vals:
                        all_marker_vals.append(v)

        all_dash_vals: list[str] = []
        if self.dash_by:
            for key, _ in groups_sorted:
                for k, v in key:
                    if k == self.dash_by and v not in all_dash_vals:
                        all_dash_vals.append(v)

        fig = go.Figure()
        for idx, (key, ys_list) in enumerate(groups_sorted):
            mat = np.stack(ys_list, axis=0)
            y_mean = np.nanmean(mat, axis=0)

            # Generate legend name
            key_dict = dict(key)
            if self.legend_format:
                # Use format string, e.g. "{model_id}, L{layer_id}"
                name = self.legend_format.format(**key_dict)
            elif self.legend_keys:
                # Use only specified keys
                name = ", ".join(
                    [f"{k}={key_dict[k]}" for k in self.legend_keys if k in key_dict]
                )
            else:
                # Default: show all keys with k=v format
                name = ", ".join([f"{k}={v}" for k, v in key])

            # Get color
            line_color = None
            if color_key:
                for k, v in key:
                    if k == color_key:
                        line_color = color_mapper.get_color(
                            color_key, v, all_color_vals
                        )
                        break

            # Get marker symbol / dash
            marker_symbol = None
            if self.marker_by:
                for k, v in key:
                    if k == self.marker_by:
                        marker_symbol = color_mapper.get_symbol_for_value(
                            self.marker_by, v, all_marker_vals
                        )
                        break

            line_dash = None
            if self.dash_by:
                for k, v in key:
                    if k == self.dash_by:
                        line_dash = color_mapper.get_dash_for_value(
                            self.dash_by, v, all_dash_vals
                        )
                        break

            fig.add_trace(
                go.Scatter(
                    x=x + offsets[idx],
                    y=y_mean,
                    mode=self.mode,
                    name=name,
                    showlegend=True,
                    line=(
                        dict(color=line_color, dash=line_dash)
                        if (line_color or line_dash)
                        else None
                    ),
                    marker=(dict(symbol=marker_symbol) if marker_symbol else None),
                    meta=dict(trace_type="cluster_bar_chart"),
                )
            )

            if self.draw_statistics and mat.shape[0] > 1:
                y_std = np.nanstd(mat, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x + offsets[idx],
                        y=-y_std,
                        mode=self.mode,
                        name=f"{name} (std)",
                        showlegend=False,
                        line=dict(color=line_color, dash="dash")
                        if line_color
                        else None,
                        meta=dict(trace_type="cluster_bar_chart_std"),
                    )
                )

        if self.draw_statistics:
            fig.add_hline(y=0, line_width=1, line=dict(color="lightgrey"))

        fig.update_xaxes(
            title=self.x_title or self.field,
            tickmode="array",
            tickvals=np.arange(x_labels.size, dtype=np.float64),
            ticktext=[f"{v:.2g}" for v in x_labels],
        )
        fig.update_yaxes(title=self.y_title or "count per bin (mean; std below 0)")
        fig.update_layout(title=self.title or f"Cluster bar chart ({self.field})")
        return apply_theme(fig, self.theme)
