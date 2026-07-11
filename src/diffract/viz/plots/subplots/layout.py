"""Layout transfer helpers for subplot composition."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go

_AXIS_REF_RE = re.compile(r"^(?P<axis>[xy])(?P<id>\d+)?(?P<domain>\s+domain)?$")


def add_figure_to_subplot(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    row: int,
    col: int,
    transfer_layout: bool,
    seen_legend_names: set[str] | None = None,
) -> None:
    """Add traces from a child figure into a specific subplot cell."""
    if seen_legend_names is None:
        seen_legend_names = set()

    axis_refs = _get_subplot_axis_refs(fig_parent, row=row, col=col)
    overlay_xaxis_map: dict[str, str] = {}
    if axis_refs is not None:
        x_ref, y_ref = axis_refs
        overlay_xaxis_map = _register_overlay_xaxes(
            fig_parent,
            child_fig,
            x_ref=x_ref,
            y_ref=y_ref,
        )

    for trace in child_fig.data:
        source_xaxis = getattr(trace, "xaxis", None)
        source_yaxis = getattr(trace, "yaxis", None)

        name = getattr(trace, "name", None)
        if name and name in seen_legend_names:
            trace.showlegend = False
            if not getattr(trace, "legendgroup", None):
                trace.legendgroup = name
        elif name:
            seen_legend_names.add(name)
            if not getattr(trace, "legendgroup", None):
                trace.legendgroup = name

        fig_parent.add_trace(trace, row=row, col=col)

        if axis_refs is None:
            continue

        mapped_xaxis = _remap_trace_axis_reference(
            source_xaxis,
            axis="x",
            primary_axis=x_ref,
            overlay_map=overlay_xaxis_map,
        )
        mapped_yaxis = _remap_trace_axis_reference(
            source_yaxis,
            axis="y",
            primary_axis=y_ref,
        )

        fig_parent.data[-1].update(xaxis=mapped_xaxis, yaxis=mapped_yaxis)

    if transfer_layout:
        transfer_layout_to_subplot(fig_parent, child_fig, row=row, col=col)


def transfer_layout_to_subplot(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    row: int,
    col: int,
) -> None:
    """Transfer axis/layout overlays from child to the target subplot cell."""
    axis_refs = _get_subplot_axis_refs(fig_parent, row=row, col=col)
    if axis_refs is None:
        return
    x_ref, y_ref = axis_refs

    _transfer_axis_props(fig_parent, child_fig, row=row, col=col)
    _transfer_shapes(fig_parent, child_fig, x_ref=x_ref, y_ref=y_ref)
    _transfer_annotations(fig_parent, child_fig, x_ref=x_ref, y_ref=y_ref)
    _transfer_images(fig_parent, child_fig, x_ref=x_ref, y_ref=y_ref)


def _transfer_axis_props(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    row: int,
    col: int,
) -> None:
    axis_props = [
        "tickmode",
        "tickvals",
        "ticktext",
        "tickangle",
        "tickformat",
        "tickprefix",
        "ticksuffix",
        "showticklabels",
        "ticks",
        "tickfont",
        "tickcolor",
        "ticklen",
        "tickwidth",
        "showgrid",
        "gridcolor",
        "gridwidth",
        "zeroline",
        "zerolinecolor",
        "zerolinewidth",
        "type",
        "range",
        "autorange",
        "categoryorder",
        "categoryarray",
        "title",
        "titlefont",
        "showline",
        "linecolor",
        "linewidth",
        "mirror",
        "dtick",
        "tick0",
        "rangemode",
        "fixedrange",
        "automargin",
    ]

    for axis_type, child_axis in (
        ("x", child_fig.layout.xaxis),
        ("y", child_fig.layout.yaxis),
    ):
        if child_axis is None:
            continue

        props = {
            key: getattr(child_axis, key)
            for key in axis_props
            if getattr(child_axis, key, None) is not None
        }
        if not props:
            continue

        if axis_type == "x":
            fig_parent.update_xaxes(row=row, col=col, **props)
        else:
            fig_parent.update_yaxes(row=row, col=col, **props)


def _transfer_shapes(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    x_ref: str,
    y_ref: str,
) -> None:
    if not child_fig.layout.shapes:
        return

    new_shapes: list[dict[str, Any]] = []
    for shape in child_fig.layout.shapes:
        shape_dict = shape.to_plotly_json()
        _remap_xy_refs(shape_dict, x_ref=x_ref, y_ref=y_ref)
        new_shapes.append(shape_dict)

    current_shapes = list(fig_parent.layout.shapes or [])
    fig_parent.update_layout(shapes=current_shapes + new_shapes)


def _transfer_annotations(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    x_ref: str,
    y_ref: str,
) -> None:
    if not child_fig.layout.annotations:
        return

    new_annotations: list[dict[str, Any]] = []
    for annotation in child_fig.layout.annotations:
        annotation_dict = annotation.to_plotly_json()
        _remap_xy_refs(annotation_dict, x_ref=x_ref, y_ref=y_ref)
        new_annotations.append(annotation_dict)

    current_annotations = list(fig_parent.layout.annotations or [])
    fig_parent.update_layout(annotations=current_annotations + new_annotations)


def _transfer_images(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    x_ref: str,
    y_ref: str,
) -> None:
    if not child_fig.layout.images:
        return

    new_images: list[dict[str, Any]] = []
    for image in child_fig.layout.images:
        image_dict = image.to_plotly_json()
        _remap_xy_refs(image_dict, x_ref=x_ref, y_ref=y_ref)
        new_images.append(image_dict)

    current_images = list(fig_parent.layout.images or [])
    fig_parent.update_layout(images=current_images + new_images)


def _remap_xy_refs(item: dict[str, Any], *, x_ref: str, y_ref: str) -> None:
    x_value = item.get("xref")
    y_value = item.get("yref")
    ax_value = item.get("axref")
    ay_value = item.get("ayref")

    remapped_x = _remap_axis_reference(x_value, axis="x", target_axis=x_ref)
    remapped_y = _remap_axis_reference(y_value, axis="y", target_axis=y_ref)
    remapped_ax = _remap_axis_reference(ax_value, axis="x", target_axis=x_ref)
    remapped_ay = _remap_axis_reference(ay_value, axis="y", target_axis=y_ref)

    if remapped_x is not None:
        item["xref"] = remapped_x
    if remapped_y is not None:
        item["yref"] = remapped_y
    if remapped_ax is not None:
        item["axref"] = remapped_ax
    if remapped_ay is not None:
        item["ayref"] = remapped_ay


def _remap_axis_reference(
    value: Any,
    *,
    axis: str,
    target_axis: str,
) -> str | None:
    if not isinstance(value, str):
        return None

    match = _AXIS_REF_RE.fullmatch(value.strip())
    if match is None or match.group("axis") != axis:
        return value

    domain = match.group("domain") or ""
    return f"{target_axis}{domain}"


def _get_subplot_axis_refs(
    fig_parent: go.Figure,
    *,
    row: int,
    col: int,
) -> tuple[str, str] | None:
    """Get parent subplot axis references (for example ``('x3', 'y3')``)."""
    grid_refs = fig_parent._grid_ref[row - 1][col - 1]
    if not grid_refs:
        return None

    grid_ref = grid_refs[0]
    remap_ref = getattr(grid_ref, "trace_kwargs", None)
    if not isinstance(remap_ref, dict):
        return None

    x_ref = remap_ref.get("xaxis")
    y_ref = remap_ref.get("yaxis")
    if not isinstance(x_ref, str) or not isinstance(y_ref, str):
        return None

    return x_ref, y_ref


def _register_overlay_xaxes(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    x_ref: str,
    y_ref: str,
) -> dict[str, str]:
    """Create parent overlay x-axes for child overlay axes (x2, x3, ...)."""
    layout_dict = child_fig.layout.to_plotly_json()
    overlay_map: dict[str, str] = {}

    for layout_key, axis_cfg in layout_dict.items():
        if not layout_key.startswith("xaxis") or layout_key == "xaxis":
            continue
        if not isinstance(axis_cfg, dict):
            continue

        suffix = layout_key[5:]
        child_axis = "x" if not suffix else f"x{suffix}"
        overlaying = axis_cfg.get("overlaying")
        if overlaying not in {"x", "x1"}:
            continue

        parent_axis = _next_axis_ref(fig_parent, axis="x")
        parent_layout_key = _axis_layout_key(parent_axis)
        parent_cfg = dict(axis_cfg)
        parent_cfg["overlaying"] = x_ref
        parent_cfg["anchor"] = y_ref
        parent_cfg.pop("domain", None)

        fig_parent.update_layout(**{parent_layout_key: parent_cfg})
        overlay_map[child_axis] = parent_axis

    return overlay_map


def _remap_trace_axis_reference(
    value: Any,
    *,
    axis: str,
    primary_axis: str,
    overlay_map: dict[str, str] | None = None,
) -> str:
    """Map a child trace axis reference to parent subplot axis reference."""
    if not isinstance(value, str):
        return primary_axis

    match = _AXIS_REF_RE.fullmatch(value.strip())
    if match is None or match.group("axis") != axis:
        return primary_axis

    axis_id = match.group("id") or ""
    child_axis = f"{axis}{axis_id}"

    if overlay_map is not None and child_axis in overlay_map:
        return overlay_map[child_axis]

    if child_axis in {axis, f"{axis}1"}:
        return primary_axis

    return child_axis


def _next_axis_ref(fig_parent: go.Figure, *, axis: str) -> str:
    """Get next available axis reference in parent layout."""
    max_id = 0
    layout_dict = fig_parent.layout.to_plotly_json()
    axis_prefix = f"{axis}axis"

    for layout_key in layout_dict:
        if layout_key == axis_prefix:
            max_id = max(max_id, 1)
            continue
        if not layout_key.startswith(axis_prefix):
            continue

        suffix = layout_key[len(axis_prefix) :]
        if suffix.isdigit():
            max_id = max(max_id, int(suffix))

    next_id = max_id + 1
    return axis if next_id == 1 else f"{axis}{next_id}"


def _axis_layout_key(axis_ref: str) -> str:
    """Convert axis ref like ``x2`` to layout key like ``xaxis2``."""
    axis = axis_ref[0]
    suffix = axis_ref[1:]
    return f"{axis}axis{suffix}"
