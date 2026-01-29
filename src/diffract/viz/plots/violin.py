"""Violin plots based on Session public APIs.

Generic distribution plot for scalar or array-like fields. Array-like values
are flattened into samples.
"""

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


def _as_samples(value: Any) -> list[float]:
    """Convert a scalar / array-like to a list of float samples."""
    if value is None:
        return []

    if isinstance(value, (int, float)):
        f = as_float(value)
        return [] if f is None else [f]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        out: list[float] = []
        for x in value.ravel():
            f = as_float(x)
            if f is not None:
                out.append(f)
        return out

    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            f = as_float(x)
            if f is not None:
                out.append(f)
        return out

    f = as_float(value)
    return [] if f is None else [f]


@dataclass(slots=True)
class ViolinPlot:
    """Violin plot for a field (scalar or array-like).

    - For scalar fields: behaves similarly to BoxPlot but with violins.
    - For array-like fields (e.g. singular values): all elements are flattened
      into samples.

    Use `color_by` to color violins by a metadata key.

    Example:
        >>> plot = ViolinPlot(
        ...     field="esd",
        ...     title="ESD Distribution",
        ...     group_by="model_id",
        ...     jitter=True,
        ... )
        >>> fig = session.draw(plot=plot)
    """

    field: str
    title: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    group_by: str = "model_id"

    # Filtering
    parameter_uids: list[str] | None = None
    parameter_names: list[str] | None = None
    parameter_types: list[str] | None = None
    model_ids: list[str] | None = None

    # Visual options
    points: Literal["all", "outliers", False] = "outliers"
    box_visible: bool = True
    meanline_visible: bool = True
    side: Literal["positive", "negative", "both"] = "positive"
    bandwidth: float | None = None

    # Color customization
    color_by: str | None = None
    marker_by: str | None = None  # Metadata key to set jitter marker symbols by

    # Jitter overlay
    jitter: bool = False
    jitter_width: float = 0.17  # Matches reference VIOLINS_JITTER_WIDTH
    jitter_offset: float = -0.4  # Places jitter left of violin
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
        """Render the violin plot using data from the session."""
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

        groups: dict[str, list[float]] = {}
        jitter_colors: dict[str, list[float]] = {}
        color_by_values: dict[str, Any] = {}
        marker_by_values: dict[str, list[Any]] = {}

        for entry in results.values():
            meta = entry.get("metadata", {})
            fields = entry.get("fields", {})
            ctx = {"model_id": meta.get("model_id"), "parameter_name": meta.get("name")}
            samples = _as_samples(get_field_value(fields, self.field, **ctx))
            if not samples:
                continue

            colors: list[float] | None = None
            if self.jitter and self.jitter_color_field is not None:
                color_raw = get_field_value(fields, self.jitter_color_field, **ctx)
                if isinstance(color_raw, (int, float)):
                    c = as_float(color_raw)
                    if c is not None:
                        colors = [c] * len(samples)
                else:
                    c_samples = _as_samples(color_raw)
                    if len(c_samples) == len(samples):
                        colors = c_samples

                if colors is None:
                    colors = [float("nan")] * len(samples)

            if not self.group_by:
                key = "all"
            elif self.group_by == "parameter_name":
                key = str(meta.get("name"))
            else:
                key = str(meta.get(self.group_by, "null"))

            groups.setdefault(key, []).extend(samples)
            if self.jitter:
                jitter_colors.setdefault(key, []).extend(colors or [])

            if self.marker_by:
                mv = extract_meta_value(entry, self.marker_by)
                # replicate marker value for every sample coming from this entry
                marker_by_values.setdefault(key, []).extend([mv] * len(samples))

            if self.color_by:
                cb_val = extract_meta_value(entry, self.color_by)
                if cb_val is not None:
                    color_by_values[key] = cb_val

        fig = go.Figure()
        names = sorted(groups.keys())

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

            # Determine violin color
            line_color = None
            if self.color_by and name in color_by_values:
                line_color = color_mapper.get_color(
                    self.color_by, color_by_values[name], all_color_vals
                )

            fig.add_trace(
                go.Violin(
                    name=name,
                    x=xs,
                    y=ys,
                    points=self.points,
                    box_visible=self.box_visible,
                    meanline_visible=self.meanline_visible,
                    side=self.side,
                    bandwidth=self.bandwidth,
                    line_color=line_color,
                )
            )

            if self.jitter and ys.size > 0:
                rng = np.random.default_rng(self.jitter_seed)
                j = rng.uniform(-self.jitter_width, self.jitter_width, size=ys.size)
                if self.jitter_density_scale:
                    j = density_scaled_jitter(y=ys, jitter=j)

                marker: dict[str, Any] = dict(
                    size=self.jitter_marker_size, opacity=self.jitter_opacity
                )
                if self.jitter_color_field is not None:
                    cs = np.asarray(jitter_colors.get(name, []), dtype=np.float64)
                    if cs.size == ys.size:
                        marker["color"] = cs
                        if use_coloraxis:
                            marker["coloraxis"] = "coloraxis"
                        else:
                            marker["colorscale"] = self.jitter_colorscale
                            if self.jitter_show_colorbar:
                                marker["colorbar"] = dict(orientation="h")

                if self.marker_by:
                    mvs = marker_by_values.get(name, [])
                    if len(mvs) == ys.size:
                        marker["symbol"] = [
                            color_mapper.get_symbol_for_value(
                                self.marker_by, mv, all_marker_vals
                            )
                            for mv in mvs
                        ]

                fig.add_trace(
                    go.Scatter(
                        x=xs + self.jitter_offset + j,
                        y=ys,
                        mode="markers",
                        showlegend=False,
                        marker=marker,
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
