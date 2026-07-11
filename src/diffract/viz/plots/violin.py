from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from diffract.viz.data import (
    DataType,
    Entry,
    FieldRef,
)
from diffract.viz.plots.base.axis import AxisType, SupportsAxis
from diffract.viz.plots.base.jitter import SupportsJitter
from diffract.viz.plots.base.marker import SupportsMarker
from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import (
    CategoricalPropertyResolver,
    NumericPropertyResolver,
    ResolvedColor,
)


@dataclass(kw_only=True)
class ViolinPlot(
    Plot,
    SupportsJitter,
    SupportsAxis("x", AxisType.CATEGORICAL),
    SupportsAxis("y", AxisType.NUMERIC),
    SupportsMarker,
):
    """Violin plot for a scalar or vector numeric field.

    Supports grouping by categorical x-axis and color customization via marker
    properties. Violin-specific options include box/meanline visibility, side
    rendering, and bandwidth.

    When ``y`` is a vector field (e.g., an eigenvalue spectrum per entry), the array
    is automatically expanded into individual observations for the violin.
    """

    y: FieldRef
    y_rescale_range: tuple[float, float] | None = None
    y_rescale_traces_separately: bool = False

    x: FieldRef

    points: Literal["all", "outliers", False] = "outliers"
    box_visible: bool = True
    meanline_visible: bool = True
    side: Literal["positive", "negative", "both"] = "positive"
    bandwidth: float | None = None

    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        """Build trace data dict for each group (category on x-axis)."""
        x_resolver = CategoricalPropertyResolver()
        y_resolver = NumericPropertyResolver(self.y_rescale_range)

        # The x axis is categorical by design; without an explicit
        # data_type digit-like string fields (layer_id, head_id) would be
        # detected as numeric and rejected.
        x_ref = replace(self.x, data_type=self.x.data_type or DataType.CATEGORICAL)
        x_values = x_resolver.resolve(x_ref, entries)
        y_values = y_resolver.resolve(self.y, entries)

        sorted_categories = self._get_sorted_categories(x_values)
        category_order = self._get_effective_category_order(sorted_categories)
        groups = self._group_by_category(x_values, y_values, category_order)

        # 1) Resolve globally for all entries
        resolved_size = self._resolve_marker_size(entries, self._theme)
        resolved_opacity = self._resolve_marker_opacity(entries, self._theme)
        resolved_symbol = self._resolve_marker_symbol(entries, self._theme)
        resolved_color = self._resolve_marker_color(entries, self._theme)
        resolved_jitter_color = self._resolve_jitter_color(entries, self._theme)

        # 2) Build per-category traces
        traces_data: dict[str, dict[str, Any]] = {}

        for idx, category in enumerate(category_order):
            trace_id = f"violin_{category}"

            trace_y = groups[category]
            if self.y_rescale_traces_separately:
                trace_y = y_resolver._normalize(trace_y)

            trace_data: dict[str, Any] = {
                "category": category,
                "category_idx": idx,
                "jitter_x_center": float(idx),
                "jitter_xaxis": "x2",
                "y_values": trace_y,
            }

            # 3) Slice resolved values for this category and add to trace
            self._add_marker_size_to_trace(
                trace_data,
                self._pick_category_values(x_values, y_values, category, resolved_size),
            )
            self._add_marker_opacity_to_trace(
                trace_data,
                self._pick_category_values(
                    x_values, y_values, category, resolved_opacity
                ),
            )
            self._add_marker_symbol_to_trace(
                trace_data,
                self._pick_category_scalar(x_values, category, resolved_symbol),
            )
            self._add_marker_color_to_trace(
                trace_data,
                self._pick_category_color(x_values, y_values, category, resolved_color),
            )

            if resolved_jitter_color is not None:
                self._add_jitter_color_to_trace(
                    trace_data,
                    self._pick_category_color(
                        x_values, y_values, category, resolved_jitter_color
                    ),
                )

            traces_data[trace_id] = trace_data

        return traces_data

    def _get_sorted_categories(self, x_values: list[Any]) -> list[str]:
        """Get unique categories sorted according to x ordering."""
        unique_categories = list(dict.fromkeys(x_values))
        order_indices = self.x.ordering.argsort(unique_categories)
        return [str(unique_categories[i]) for i in order_indices]

    def _get_effective_category_order(self, sorted_categories: list[str]) -> list[str]:
        """Resolve category order used for both traces and jitter centers."""
        if self.x_categoryorder != "array" or self.x_categoryarray is None:
            return sorted_categories

        preferred = [str(category) for category in self.x_categoryarray]
        preferred_existing = [
            category for category in preferred if category in sorted_categories
        ]
        remaining = [
            category
            for category in sorted_categories
            if category not in preferred_existing
        ]
        return preferred_existing + remaining

    def _group_by_category(
        self, x_values: list[Any], y_values: list[Any], categories: list[str]
    ) -> dict[str, list[float]]:
        """Group y-values by their x category.

        Scalar y-values are appended as single floats.
        Array y-values (``np.ndarray``) are exploded into individual floats,
        so that each element of the vector becomes a separate observation
        in the violin (useful for per-parameter spectra like ESD).
        """
        groups: dict[str, list[float]] = {cat: [] for cat in categories}

        for x_val, y_val in zip(x_values, y_values, strict=False):
            if y_val is None:
                continue
            cat = str(x_val)
            if isinstance(y_val, np.ndarray):
                groups[cat].extend(y_val.ravel().tolist())
            else:
                groups[cat].append(float(y_val))

        return groups

    def _build_figure(self) -> go.Figure:
        """Build the plotly figure with violin traces."""
        fig = go.Figure()

        if self._traces_data is None:
            return fig

        for trace_id, trace_data in self._traces_data.items():
            category = trace_data["category"]
            y_values = trace_data["y_values"]

            if len(y_values) == 0:
                continue

            fig.add_trace(
                go.Violin(
                    name=category,
                    x=[category] * len(y_values),
                    y=y_values,
                    points=self.points,
                    box_visible=self.box_visible,
                    meanline_visible=self.meanline_visible,
                    width=0.5,
                    side=self.side,
                    bandwidth=self.bandwidth,
                    meta={"trace_id": trace_id},
                )
            )

        # Set title
        if self.title:
            fig.update_layout(title=self.title)
        else:
            fig.update_layout(title=f"{self.y.field} by {self.x.field}")

        categoryorder = self.x_categoryorder or "array"

        if categoryorder == "array":
            categoryarray = self.x_categoryarray or [
                trace_data["category"]
                for trace_data in sorted(
                    self._traces_data.values(),
                    key=lambda data: data["category_idx"],
                )
            ]
        else:
            categoryarray = None

        half_span = max(0.5, abs(self.jitter_offset) + self.jitter_width + 0.05)
        fig.update_layout(
            xaxis=dict(
                categoryorder=categoryorder,
                categoryarray=categoryarray,
                range=[-half_span, len(self._traces_data) - 1 + 0.5 * half_span],
            )
        )

        return fig

    @staticmethod
    def _pick_category_values(
        x_values: list[Any],
        y_values: list[Any],
        category: str,
        resolved: float | list[float] | None,
    ) -> float | list[float] | None:
        """Slice a resolved numeric value for a specific category.

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — filter to entries matching this category.
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        return [
            val
            for x_val, y_val, val in zip(x_values, y_values, resolved, strict=False)
            if str(x_val) == category and y_val is not None
        ]

    @staticmethod
    def _pick_category_scalar(
        x_values: list[Any],
        category: str,
        resolved: str | list[str] | None,
    ) -> str | None:
        """Pick a single categorical value for a category (first match).

        If resolved is scalar or None — return as-is.
        If resolved is a per-entry list — return the first matching value.
        """
        if resolved is None or not isinstance(resolved, list):
            return resolved
        for x_val, val in zip(x_values, resolved, strict=False):
            if str(x_val) == category:
                return val
        return None

    @staticmethod
    def _pick_category_color(
        x_values: list[Any],
        y_values: list[Any],
        category: str,
        resolved_color: ResolvedColor,
    ) -> ResolvedColor:
        """Build a per-category ResolvedColor from a global ResolvedColor."""
        if resolved_color.values is not None:
            filtered = [
                c_val
                for x_val, y_val, c_val in zip(
                    x_values, y_values, resolved_color.values, strict=False
                )
                if str(x_val) == category and y_val is not None
            ]
            return ResolvedColor(values=filtered)

        if resolved_color.color is not None:
            if isinstance(resolved_color.color, list):
                for x_val, color in zip(x_values, resolved_color.color, strict=False):
                    if str(x_val) == category:
                        return ResolvedColor(color=color)
                return ResolvedColor()
            return resolved_color

        return resolved_color
