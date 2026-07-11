from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import plotly.graph_objects as go

from diffract.viz.data import (
    DataType,
    Entry,
    FieldRef,
)
from diffract.viz.plots.base.axis import AxisType, SupportsAxis
from diffract.viz.plots.base.coloraxis import SupportsColoraxis
from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import (
    CategoricalPropertyResolver,
    NumericPropertyResolver,
)


@dataclass(kw_only=True)
class HeatmapPlot(
    Plot,
    SupportsAxis("x", AxisType.CATEGORICAL),
    SupportsAxis("y", AxisType.CATEGORICAL),
    SupportsColoraxis("heatmap"),
):
    """Heatmap plot that pivots a scalar field by two categorical dimensions.

    Each cell in the heatmap represents a value of ``z`` at the intersection
    of a ``y`` row and an ``x`` column.  When multiple entries map to the
    same (x, y) pair the **last** value wins (no aggregation).

    Coloraxis settings (colorscale, cmin/cmax, colorbar) are inherited from
    the ``SupportsColoraxis("heatmap")`` mixin and can be controlled via
    ``heatmap_colorscale``, ``heatmap_cmin``, ``heatmap_cmax``, etc.
    """

    z: FieldRef
    z_rescale_range: tuple[float, float] | None = None

    y: FieldRef
    x: FieldRef

    fill_value: float = float("nan")

    show_text: bool = False
    text_format: str = ".2f"
    text_font_size: int = 10

    reverse_y: bool = True

    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        """Build a single trace-data dict containing the full z-matrix."""
        x_resolver = y_resolver = CategoricalPropertyResolver()
        z_resolver = NumericPropertyResolver(self.z_rescale_range)

        x_ref = replace(self.x, data_type=self.x.data_type or DataType.CATEGORICAL)
        y_ref = replace(self.y, data_type=self.y.data_type or DataType.CATEGORICAL)
        x_values = x_resolver.resolve(x_ref, entries)
        y_values = y_resolver.resolve(y_ref, entries)
        z_values = z_resolver.resolve(self.z, entries)

        sorted_x = self._get_sorted_unique(x_values, self.x)
        sorted_y = self._get_sorted_unique(y_values, self.y)

        cell_values = self._collect_cell_values(x_values, y_values, z_values)
        z_matrix = self._build_matrix(sorted_x, sorted_y, cell_values)

        traces_data: dict[str, dict[str, Any]] = {
            "heatmap": {
                "x_labels": sorted_x,
                "y_labels": sorted_y,
                "z_matrix": z_matrix,
            },
        }
        return traces_data

    @staticmethod
    def _get_sorted_unique(values: list[Any], ref: FieldRef) -> list[str]:
        """Return unique values sorted according to the field's ordering."""
        unique = list(dict.fromkeys(values))
        order_indices = ref.ordering.argsort(unique)
        return [str(unique[i]) for i in order_indices]

    @staticmethod
    def _collect_cell_values(
        x_values: list[Any],
        y_values: list[Any],
        z_values: list[Any],
    ) -> dict[tuple[str, str], float]:
        """Map (x, y) string pairs to scalar z-values."""
        cell_values: dict[tuple[str, str], float] = {}
        for x_val, y_val, z_val in zip(x_values, y_values, z_values, strict=False):
            if z_val is None:
                continue
            cell_values[(str(x_val), str(y_val))] = float(z_val)
        return cell_values

    def _build_matrix(
        self,
        sorted_x: list[str],
        sorted_y: list[str],
        cell_values: dict[tuple[str, str], float],
    ) -> np.ndarray:
        """Build a 2-D z-matrix (rows=y, cols=x) from cell values."""
        z_matrix = np.full(
            (len(sorted_y), len(sorted_x)),
            self.fill_value,
            dtype=np.float64,
        )
        x_idx = {label: j for j, label in enumerate(sorted_x)}
        y_idx = {label: i for i, label in enumerate(sorted_y)}

        for (x_label, y_label), value in cell_values.items():
            j = x_idx.get(x_label)
            i = y_idx.get(y_label)
            if i is not None and j is not None:
                z_matrix[i, j] = value

        return z_matrix

    def _build_figure(self) -> go.Figure:
        """Build the Plotly figure with a single Heatmap trace."""
        fig = go.Figure()

        if self._traces_data is None:
            return fig

        trace_data = self._traces_data["heatmap"]
        x_labels: list[str] = trace_data["x_labels"]
        y_labels: list[str] = trace_data["y_labels"]
        z_matrix: np.ndarray = trace_data["z_matrix"]

        heatmap_kwargs: dict[str, Any] = {
            "z": z_matrix,
            "x": x_labels,
            "y": y_labels,
            "coloraxis": self.resolve_coloraxis(fig),
        }

        if self.show_text:
            text = np.vectorize(
                lambda val: "" if np.isnan(val) else format(val, self.text_format),
            )(z_matrix)
            heatmap_kwargs["text"] = text
            heatmap_kwargs["texttemplate"] = "%{text}"
            heatmap_kwargs["textfont"] = {"size": self.text_font_size}

        fig.add_trace(go.Heatmap(**heatmap_kwargs))

        # Title
        if self.title:
            fig.update_layout(title=self.title)
        else:
            fig.update_layout(
                title=f"{self.z.field} by {self.y.field} x {self.x.field}"
            )

        # Reverse y-axis so the first row is at the top (matrix convention)
        if self.reverse_y:
            fig.update_yaxes(autorange="reversed")

        return fig
