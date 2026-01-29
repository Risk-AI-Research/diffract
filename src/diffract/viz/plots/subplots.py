"""Subplot composition for Plot objects (ported conceptually from notebooks_src)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from diffract.core.utils import imports as import_utils
from diffract.session import Session
from diffract.viz.renderer import Plot

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


@dataclass(slots=True)
class SubplotSpec:
    """Specification for a single subplot in a GridPlot."""

    row: int
    col: int
    title: str
    plot: Plot


@dataclass(slots=True)
class GridPlot:
    """Compose multiple plots into a grid layout.

    Supports theming which applies to the entire grid.

    Example:
        >>> grid = GridPlot(
        ...     subplots=[
        ...         SubplotSpec(1, 1, "Frobenius", BoxPlot(field="frob_norm")),
        ...         SubplotSpec(1, 2, "Stable Rank", BoxPlot(field="stable_rank")),
        ...     ],
        ...     make_subplots_kwargs={"rows": 1, "cols": 2},
        ... )
        >>> fig = session.draw(plot=grid)
    """

    subplots: list[SubplotSpec]
    make_subplots_kwargs: dict[str, Any]

    # Theming
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the grid of subplots using data from the session."""
        import_utils.require("plotly.graph_objects")
        subplots = import_utils.require("plotly.subplots")
        from diffract.viz.themes import apply_theme

        rows = [sp.row for sp in self.subplots]
        cols = [sp.col for sp in self.subplots]
        total_rows = max(rows) if rows else 1
        total_cols = max(cols) if cols else 1

        sorted_specs = sorted(
            self.subplots, key=lambda sp: sp.row * total_cols + sp.col
        )
        titles = [sp.title for sp in sorted_specs]

        # `make_subplots_kwargs` may also contain `rows`/`cols` (common in configs).
        # We compute rows/cols from `SubplotSpec` and enforce them here.
        kwargs = dict(self.make_subplots_kwargs or {})
        kwargs.pop("rows", None)
        kwargs.pop("cols", None)

        grid = subplots.make_subplots(
            rows=total_rows,
            cols=total_cols,
            subplot_titles=titles,
            **kwargs,
        )

        # Track which legend names we've already shown to avoid duplicates
        seen_legend_names: set[str] = set()

        for sp in self.subplots:
            child = sp.plot.render(session)
            add_figure_to_subplot(
                grid,
                child,
                row=sp.row,
                col=sp.col,
                transfer_layout=True,
                seen_legend_names=seen_legend_names,
            )

        return apply_theme(grid, self.theme)


def add_figure_to_subplot(
    fig_parent: go.Figure,
    child_fig: go.Figure,
    *,
    row: int,
    col: int,
    transfer_layout: bool,
    seen_legend_names: set[str] | None = None,
) -> None:
    """Add traces from child_fig to fig_parent at specified row/col.

    Args:
        fig_parent: Parent figure with subplot grid.
        child_fig: Child figure whose traces will be added.
        row: Target row in the subplot grid (1-indexed).
        col: Target column in the subplot grid (1-indexed).
        transfer_layout: Whether to copy axis properties from child.
        seen_legend_names: If provided, tracks legend names to avoid duplicates.
            Traces with a name already in this set get showlegend=False.
    """
    if seen_legend_names is None:
        seen_legend_names = set()

    for trace in child_fig.data:
        name = getattr(trace, "name", None)
        # Avoid duplicate legend entries: if name seen, hide from legend
        if name and name in seen_legend_names:
            trace.showlegend = False
            trace.legendgroup = name
        elif name:
            seen_legend_names.add(name)
            trace.legendgroup = name
        fig_parent.add_trace(trace, row=row, col=col)

    if transfer_layout:
        transfer_layout_to_subplot(fig_parent, child_fig, row=row, col=col)


def transfer_layout_to_subplot(
    fig_parent: go.Figure, child_fig: go.Figure, *, row: int, col: int
) -> None:
    """Transfer axis properties and shapes from child to parent subplot."""
    # Reuse the same layout-transfer strategy as in notebooks_src.
    (grid_ref,) = fig_parent._grid_ref[row - 1][col - 1]  # noqa: SLF001
    remap_ref = grid_ref.trace_kwargs
    x_ref, y_ref = remap_ref["xaxis"], remap_ref["yaxis"]

    axis_props = [
        "tickmode",
        "tickvals",
        "ticktext",
        "tickangle",
        "tickformat",
        "tickprefix",
        "ticksuffix",
        "showgrid",
        "gridcolor",
        "gridwidth",
        "zeroline",
        "zerolinecolor",
        "zerolinewidth",
        "type",
        "range",
        "categoryorder",
        "categoryarray",
        "title",
        "titlefont",
        "showline",
        "linecolor",
        "linewidth",
        "mirror",
        "dtick",
        "rangemode",
        "fixedrange",
        "tickfont",
        "tickcolor",
        "ticklen",
        "tickwidth",
    ]

    for axis_type, child_axis in [
        ("x", child_fig.layout.xaxis),
        ("y", child_fig.layout.yaxis),
    ]:
        props = {
            k: getattr(child_axis, k)
            for k in axis_props
            if getattr(child_axis, k, None) is not None
        }
        if props:
            if axis_type == "x":
                fig_parent.update_xaxes(row=row, col=col, **props)
            else:
                fig_parent.update_yaxes(row=row, col=col, **props)

    if child_fig.layout.shapes:
        new_shapes = []
        for shape in child_fig.layout.shapes:
            shape_dict = shape.to_plotly_json()
            shape_dict["xref"] = str(shape_dict.get("xref", "")).replace("x", x_ref)
            shape_dict["yref"] = str(shape_dict.get("yref", "")).replace("y", y_ref)
            new_shapes.append(shape_dict)
        current_shapes = list(fig_parent.layout.shapes or [])
        fig_parent.update_layout(shapes=current_shapes + new_shapes)

    if child_fig.layout.annotations:
        new_annotations = []
        for ann in child_fig.layout.annotations:
            ann_dict = ann.to_plotly_json()
            ann_dict["xref"] = str(ann_dict.get("xref", "")).replace("x", x_ref)
            ann_dict["yref"] = str(ann_dict.get("yref", "")).replace("y", y_ref)
            new_annotations.append(ann_dict)
        current_annotations = list(fig_parent.layout.annotations or [])
        fig_parent.update_layout(annotations=current_annotations + new_annotations)

    if child_fig.layout.images:
        new_images = []
        for img in child_fig.layout.images:
            img_dict = img.to_plotly_json()
            img_dict["xref"] = str(img_dict.get("xref", "")).replace("x", x_ref)
            img_dict["yref"] = str(img_dict.get("yref", "")).replace("y", y_ref)
            new_images.append(img_dict)
        current_images = list(fig_parent.layout.images or [])
        fig_parent.update_layout(images=current_images + new_images)
        
    for attr in dir(child_fig.layout):
        if attr.startswith("coloraxis"):
            fig_parent.update_layout({attr: getattr(child_fig.layout, attr)})
