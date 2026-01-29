"""Generic scatter plots based on scalar fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.plots.common import (
    as_float,
    extract_meta_value,
    fetch_data,
    get_field_value,
)

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


@dataclass(slots=True)
class ScatterPlot:
    """Scatter plot for two scalar fields.

    Use `color_by` to color markers by a metadata key.

    Example:
        >>> plot = ScatterPlot(
        ...     x_field="frob_norm",
        ...     y_field="stable_rank",
        ...     color_by="model_id",
        ... )
        >>> fig = session.draw(plot=plot)
    """

    x_field: str
    y_field: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    group_by: str = "model_id"
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None
    opacity: float = 0.7
    marker_size: int = 6

    # Color customization
    color_by: str | None = None

    # Symbol customization
    marker_by: str | None = None

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the scatter plot using data from the session."""
        go = import_utils.require("plotly.graph_objects")
        from diffract.viz.colors import ColorMapper
        from diffract.viz.themes import apply_theme

        ptypes = (
            [ParameterType.from_string(p) for p in self.parameter_types]
            if self.parameter_types
            else None
        )

        results = fetch_data(
            session,
            [self.x_field, self.y_field],
            parameter_uids=self.parameter_uids,
            parameter_names=self.parameter_names,
            parameter_types=ptypes,
            model_ids=self.model_ids,
        )

        xs_by_group: dict[str, list[float]] = {}
        ys_by_group: dict[str, list[float]] = {}
        color_vals_by_group: dict[str, list] = {}
        marker_vals_by_group: dict[str, list] = {}

        all_color_vals: list = []
        all_marker_vals: list = []

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            x = as_float(get_field_value(fields, self.x_field, **ctx))
            y = as_float(get_field_value(fields, self.y_field, **ctx))
            if x is None or y is None:
                continue

            key = str(meta.get(self.group_by, "null")) if self.group_by else "all"
            xs_by_group.setdefault(key, []).append(x)
            ys_by_group.setdefault(key, []).append(y)

            if self.color_by:
                cv = extract_meta_value(entry, self.color_by)
                color_vals_by_group.setdefault(key, []).append(cv)
                if cv not in all_color_vals:
                    all_color_vals.append(cv)

            if self.marker_by:
                mv = extract_meta_value(entry, self.marker_by)
                marker_vals_by_group.setdefault(key, []).append(mv)
                if mv not in all_marker_vals:
                    all_marker_vals.append(mv)

        color_mapper = ColorMapper(theme=self.theme)

        fig = go.Figure()
        for g in sorted(xs_by_group.keys()):
            marker_cfg = dict(size=self.marker_size, opacity=self.opacity)

            # Apply colors if color_by is set
            if self.color_by and g in color_vals_by_group:
                cvs = color_vals_by_group[g]
                colors = [
                    color_mapper.get_color(self.color_by, cv, all_color_vals)
                    for cv in cvs
                ]
                marker_cfg["color"] = colors

            if self.marker_by and g in marker_vals_by_group:
                mvs = marker_vals_by_group[g]
                symbols = [
                    color_mapper.get_symbol_for_value(
                        self.marker_by, mv, all_marker_vals
                    )
                    for mv in mvs
                ]
                marker_cfg["symbol"] = symbols

            fig.add_trace(
                go.Scatter(
                    x=xs_by_group[g],
                    y=ys_by_group[g],
                    name=g,
                    mode="markers",
                    marker=marker_cfg,
                )
            )

        fig.update_layout(title=self.title or f"{self.y_field} vs {self.x_field}")
        fig.update_xaxes(title=self.x_title or self.x_field)
        fig.update_yaxes(title=self.y_title or self.y_field)
        return apply_theme(fig, self.theme)
