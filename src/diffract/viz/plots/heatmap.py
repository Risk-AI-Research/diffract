"""Generic heatmaps from scalar fields pivoted by metadata keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.plots.common import as_float, fetch_data, get_field_value, sort_key

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


@dataclass(slots=True)
class HeatmapPivotPlot:
    """Build a heatmap by pivoting a scalar field by metadata keys.

    Example: if parameters have `layer_id` and `head_id` in metadata,
    create a layer x head heatmap for any scalar metric.
    """

    value_field: str
    row_by: str
    col_by: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    model_id: str | None = None
    model_ids: list[str] | None = None
    parameter_types: list[str] | None = None
    fill_value: float = float("nan")
    show_text: bool = False
    text_format: str = ".2f"
    text_font_size: int = 10
    colorscale: str = "Viridis"
    tickangle: int | None = None  # X-axis tick rotation, e.g. -45 or 90

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the heatmap using data from the session."""
        go = import_utils.require("plotly.graph_objects")
        from diffract.viz.themes import apply_theme

        ptypes = (
            [ParameterType.from_string(p) for p in self.parameter_types]
            if self.parameter_types
            else None
        )
        model_ids = self.model_ids
        if model_ids is None and self.model_id is not None:
            model_ids = [self.model_id]

        results = fetch_data(
            session,
            [self.value_field],
            parameter_uids=self.parameter_uids,
            parameter_names=self.parameter_names,
            parameter_types=ptypes,
            model_ids=model_ids,
        )

        rows_set: set[Any] = set()
        cols_set: set[Any] = set()
        values: dict[tuple[Any, Any], float] = {}

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            v = as_float(get_field_value(fields, self.value_field, **ctx))
            if v is None:
                continue

            r = meta.get(self.row_by)
            c = meta.get(self.col_by)
            if r is None or c is None:
                continue

            rows_set.add(r)
            cols_set.add(c)
            values[(r, c)] = v

        rows = sorted(rows_set, key=sort_key)
        cols = sorted(cols_set, key=sort_key)

        z = np.full((len(rows), len(cols)), self.fill_value, dtype=np.float64)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if (r, c) in values:
                    z[i, j] = values[(r, c)]

        text = None
        if self.show_text:
            text = np.vectorize(
                lambda x: ("" if np.isnan(x) else format(x, self.text_format))
            )(z)

        heatmap_kwargs: dict[str, Any] = dict(
            z=z,
            x=[str(c) for c in cols],
            y=[str(r) for r in rows],
            coloraxis="coloraxis",
        )
        if text is not None:
            heatmap_kwargs["text"] = text
            heatmap_kwargs["texttemplate"] = "%{text}"
            heatmap_kwargs["textfont"] = dict(size=self.text_font_size)

        fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])
        fig.update_layout(title=self.title or self.value_field, coloraxis=dict(colorscale=self.colorscale))

        xaxis_kwargs: dict[str, Any] = {"title": self.col_by if self.x_title is None else self.x_title}
        if self.tickangle is not None:
            xaxis_kwargs["tickangle"] = self.tickangle
        fig.update_xaxes(**xaxis_kwargs)

        fig.update_yaxes(title=self.row_by if self.y_title is None else self.y_title, autorange="reversed")
        return apply_theme(fig, self.theme)
