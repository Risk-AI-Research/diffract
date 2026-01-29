"""Scalar plots (BoxPlot) based on Session public APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.plots.common import (
    as_float,
    density_scaled_jitter,
    extract_meta_value,
    fetch_data,
    get_field_value,
)

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


def _as_color_scalar(value: Any) -> float | None:
    """Best-effort scalar extraction for jitter coloring."""
    f = as_float(value)
    if f is not None:
        return f
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        try:
            return float(np.nanmean(value.astype(np.float64)))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        xs = [as_float(x) for x in value]
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return float(np.mean(np.asarray(xs, dtype=np.float64)))
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError):
            return None
    return None


@dataclass(slots=True)
class BoxPlot:
    """Box plot for a single scalar field.

    Supports grouping, filtering, theming, and color customization.
    Use `color_by` to color boxes by a metadata key (e.g., "layer_id").

    Example:
        >>> plot = BoxPlot(
        ...     field="stable_rank",
        ...     title="Stable Rank Distribution",
        ...     group_by="model_id",
        ...     color_by="model_id",
        ... )
        >>> fig = session.draw(plot=plot)
    """

    field: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    group_by: str = "model_id"
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None

    # Box styling
    boxpoints: Literal["all", "outliers", False] = "outliers"
    box_width: float = 0.5  # Width of box (0-1 range, larger = wider box)

    # Color customization
    color_by: str | None = None  # Metadata key to color boxes by
    marker_by: str | None = None  # Metadata key to set jitter marker symbols by

    # Jitter overlay (separate scatter layer)
    jitter: bool = False
    jitter_width: float = 0.12  # Matches reference BOXPLOT_JITTER_WIDTH
    jitter_offset: float = -0.35  # Places jitter left of box
    jitter_seed: int = 42
    jitter_density_scale: bool = True
    jitter_marker_size: int = 4
    jitter_opacity: float = 0.7
    jitter_color_field: str | None = None
    jitter_colorscale: str = "Viridis"
    jitter_show_colorbar: bool = False

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the box plot using data from the session."""
        go = import_utils.require("plotly.graph_objects")
        from diffract.viz.colors import ColorMapper
        from diffract.viz.themes import apply_theme

        ptypes = (
            [ParameterType.from_string(p) for p in self.parameter_types]
            if self.parameter_types
            else None
        )

        fields_to_compute = [self.field]
        if self.jitter and self.jitter_color_field is not None:
            fields_to_compute.append(self.jitter_color_field)

        results = fetch_data(
            session,
            fields_to_compute,
            parameter_uids=self.parameter_uids,
            parameter_names=self.parameter_names,
            parameter_types=ptypes,
            model_ids=self.model_ids,
        )

        # Collect data by groups
        groups: dict[str, list[float]] = {}
        jitter_colors: dict[str, list[float]] = {}
        color_by_values: dict[str, Any] = {}  # group -> color_by value
        marker_by_values: dict[
            str, list[Any]
        ] = {}  # group -> per-point marker_by value

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            value = as_float(get_field_value(fields, self.field, **ctx))
            if value is None:
                continue

            color_val: float | None = None
            if self.jitter and self.jitter_color_field is not None:
                color_val = _as_color_scalar(
                    get_field_value(fields | meta, self.jitter_color_field, **ctx)
                )

            if not self.group_by:
                key = "all"
            elif self.group_by == "parameter_name":
                key = str(meta.get("name"))
            else:
                key = str(meta.get(self.group_by, "null"))

            groups.setdefault(key, []).append(value)
            if self.jitter and self.jitter_color_field is not None:
                jitter_colors.setdefault(key, []).append(
                    float("nan") if color_val is None else color_val
                )

            if self.marker_by:
                mv = extract_meta_value(entry, self.marker_by)
                marker_by_values.setdefault(key, []).append(mv)

            # Track color_by value for this group
            if self.color_by:
                cb_val = extract_meta_value(entry, self.color_by)
                if cb_val is not None:
                    color_by_values[key] = cb_val

        fig = go.Figure()
        names = sorted(groups.keys())

        # Color mapping
        color_mapper = ColorMapper(theme=self.theme)
        all_color_vals = list(color_by_values.values()) if self.color_by else []
        all_marker_vals: list[Any] = []
        if self.marker_by:
            for vs in marker_by_values.values():
                for v in vs:
                    if v not in all_marker_vals:
                        all_marker_vals.append(v)

        use_coloraxis = self.jitter and (self.jitter_color_field is not None)
        if use_coloraxis:
            fig.update_layout(
                coloraxis=dict(
                    colorscale=self.jitter_colorscale,
                    colorbar=(
                        dict(orientation="h") if self.jitter_show_colorbar else None
                    ),
                )
            )

        for idx, name in enumerate(names):
            ys = np.asarray(groups[name], dtype=np.float64)
            xs = np.full(ys.size, idx, dtype=np.float64)

            # Determine box color
            box_color = None
            if self.color_by and name in color_by_values:
                box_color = color_mapper.get_color(
                    self.color_by, color_by_values[name], all_color_vals
                )

            marker = dict(color=box_color) if box_color else None
            line = dict(color=box_color) if box_color else None

            fig.add_trace(
                go.Box(
                    name=name,
                    x=xs,
                    y=ys,
                    boxpoints=self.boxpoints,
                    width=self.box_width,
                    marker=marker,
                    line=line,
                )
            )

            if self.jitter and ys.size > 0:
                rng = np.random.default_rng(self.jitter_seed)
                j = rng.uniform(-self.jitter_width, self.jitter_width, size=ys.size)
                rng_idx = rng.permutation(np.arange(ys.size))
                if self.jitter_density_scale:
                    j = density_scaled_jitter(y=ys, jitter=j)

                marker_cfg: dict[str, Any] = dict(
                    size=self.jitter_marker_size, opacity=self.jitter_opacity
                )
                if self.jitter_color_field is not None:
                    cs = np.asarray(jitter_colors.get(name, []), dtype=np.float64)
                    if cs.size == ys.size:
                        marker_cfg["color"] = cs[rng_idx]
                        if use_coloraxis:
                            marker_cfg["coloraxis"] = "coloraxis"
                        else:
                            marker_cfg["colorscale"] = self.jitter_colorscale
                            if self.jitter_show_colorbar:
                                marker_cfg["colorbar"] = dict(orientation="h")

                if self.marker_by:
                    mvs = marker_by_values.get(name, [])
                    if len(mvs) == ys.size:
                        marker_cfg["symbol"] = [
                            color_mapper.get_symbol_for_value(
                                self.marker_by, mv, all_marker_vals
                            )
                            for mv in mvs
                        ]

                fig.add_trace(
                    go.Scatter(
                        x=(xs + self.jitter_offset + j)[rng_idx],
                        y=ys[rng_idx],
                        mode="markers",
                        showlegend=False,
                        marker=marker_cfg,
                        meta=dict(trace_type=f"{name} jitter"),
                    )
                )

        fig.update_xaxes(
            title=self.group_by if self.x_title is None else self.x_title,
            tickmode="array",
            tickvals=list(range(len(names))),
            ticktext=names,
            range=[-1, max(1, len(names))],
        )
        fig.update_yaxes(title=self.field if self.y_title is None else self.y_title)

        fig.update_layout(title=self.title or f"{self.field} by {self.group_by}")
        return apply_theme(fig, self.theme)
