"""Line/sparkline-like plots based on scalar fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from statistics import mean, stdev
from matplotlib.colors import to_rgba

from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.plots.common import (
    as_float,
    extract_meta_value,
    fetch_data,
    get_field_value,
    sort_key,
)

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


@dataclass(slots=True)
class LineByMetaPlot:
    """Plot a scalar field as a function of a metadata key.

    This is a generic replacement for many "sparkline" uses where
    x-axis is something like step/layer/head stored in metadata.

    Use `color_by` to color lines by a metadata key.

    Example:
        >>> plot = LineByMetaPlot(
        ...     y_field="stable_rank",
        ...     x_by="in_model_idx",
        ...     color_by="model_id",
        ... )
        >>> fig = session.draw(plot=plot)
    """

    y_field: str
    x_by: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    group_by: str = "model_id"
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None
    mode: str = "lines"

    # Color customization
    color_by: str | None = None
    marker_by: str | None = None
    dash_by: str | None = None

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the line plot using data from the session."""
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
            [self.y_field],
            parameter_uids=self.parameter_uids,
            parameter_names=self.parameter_names,
            parameter_types=ptypes,
            model_ids=self.model_ids,
        )

        series: dict[str, list[tuple[object, float]]] = {}
        color_by_values: dict[str, object] = {}
        marker_by_values: dict[str, object] = {}
        dash_by_values: dict[str, object] = {}

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            y = as_float(get_field_value(fields | meta, self.y_field, **ctx))
            if y is None:
                continue
            x = meta.get(self.x_by)
            if x is None:
                continue
            g = str(meta.get(self.group_by, "null")) if self.group_by else "all"
            series.setdefault(g, []).append((x, y))

            if self.color_by:
                cv = extract_meta_value(entry, self.color_by)
                if cv is not None:
                    color_by_values[g] = cv
            if self.marker_by:
                mv = extract_meta_value(entry, self.marker_by)
                if mv is not None:
                    marker_by_values[g] = mv
            if self.dash_by:
                dv = extract_meta_value(entry, self.dash_by)
                if dv is not None:
                    dash_by_values[g] = dv

        color_mapper = ColorMapper(theme=self.theme)
        all_color_vals = list(color_by_values.values()) if self.color_by else []
        all_marker_vals = list(marker_by_values.values()) if self.marker_by else []
        all_dash_vals = list(dash_by_values.values()) if self.dash_by else []

        fig = go.Figure()
        for g in sorted(series.keys()):
            pts = sorted(series[g], key=lambda p: sort_key(p[0]))
            
            subgroups: dict[str, list[float]] = {}
            for x, y in pts:
                if x not in subgroups:
                    subgroups[x] = []
                subgroups[x].append(y)
                
            mean_vals: dict[str, float] = {}
            std_vals: dict[str, float] = {}
            for x, subgroup in subgroups.items():
                if len(subgroup) > 1:
                    mean_vals[x] = mean(subgroup)
                    std_vals[x] = stdev(subgroup)
                else:
                    mean_vals[x] = subgroup[0]
                    std_vals[x] = subgroup[0]

            line_cfg = {}
            if self.color_by and g in color_by_values:
                line_cfg["color"] = color_mapper.get_color(
                    self.color_by, color_by_values[g], all_color_vals
                )
            if self.dash_by and g in dash_by_values:
                line_cfg["dash"] = color_mapper.get_dash_for_value(
                    self.dash_by, dash_by_values[g], all_dash_vals
                )

            marker_cfg = None
            effective_mode = self.mode
            if self.marker_by:
                # Show markers if we have per-series symbol encoding.
                if "markers" not in effective_mode:
                    effective_mode = (
                        "lines+markers" if "lines" in effective_mode else "markers"
                    )
                if g in marker_by_values:
                    marker_cfg = dict(
                        symbol=color_mapper.get_symbol_for_value(
                            self.marker_by, marker_by_values[g], all_marker_vals
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=[x for x in mean_vals.keys()],
                    y=[y for y in mean_vals.values()],
                    name=g,
                    mode=effective_mode,
                    line=line_cfg if line_cfg else None,
                    marker=marker_cfg,
                    legendgroup=g,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[x for x in mean_vals.keys()],
                    y=[y + delta for y, delta in zip(mean_vals.values(), std_vals.values())],
                    name=g,
                    showlegend=False,
                    mode=effective_mode,
                    line=line_cfg | dict(width=0.5) if line_cfg else dict(width=0.5),
                    marker=marker_cfg,
                    legendgroup=g,
                )
            )
            
            r_, g_, b_, a_ = to_rgba(line_cfg["color"], alpha=0.3)
            fill_color = f"rgba({int(round(r_ * 255))},{int(round(g_ * 255))},{int(round(b_ * 255))},{a_})"
            
            fig.add_trace(
                go.Scatter(
                    x=[x for x in mean_vals.keys()],
                    y=[y - delta for y, delta in zip(mean_vals.values(), std_vals.values())],
                    name=g,
                    showlegend=False,
                    mode=effective_mode,
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=line_cfg | dict(width=0.5) if line_cfg else dict(width=0.5),
                    marker=marker_cfg,
                    legendgroup=g,
                )
            )

        fig.update_layout(title=self.title or f"{self.y_field} by {self.x_by}")
        fig.update_xaxes(title=self.x_by if self.x_title is None else self.x_title)
        fig.update_yaxes(title=self.y_field if self.y_title is None else self.y_title)
        return apply_theme(fig, self.theme)
