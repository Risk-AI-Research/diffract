"""Grid plot composition for `viz` Plot objects."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from plotly.subplots import make_subplots

from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import Theme, apply_theme

from .coloraxis import (
    ColoraxisRegistry,
    apply_coloraxis_configs,
    distribute_colorbars,
    extract_coloraxis_share_keys,
    extract_legacy_coloraxis_name,
    remap_coloraxis,
)
from .layout import add_figure_to_subplot
from .spec import VALID_SESSION_FILTER_KEYS, VALUE_FILTER_SENTINEL, SubplotSpec

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from diffract.session import Session


@dataclass(kw_only=True)
class GridPlot:
    """Compose multiple plots into a grid layout."""

    subplots: list[SubplotSpec]
    make_subplots_kwargs: dict[str, Any]

    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render all subplot specs into a single Plotly grid figure."""
        _validate_subplots(self.subplots)

        total_rows = max((spec.row for spec in self.subplots), default=1)
        total_cols = max((spec.col for spec in self.subplots), default=1)
        sorted_specs = sorted(self.subplots, key=lambda spec: (spec.row, spec.col))
        titles = _build_subplot_titles(sorted_specs, total_rows, total_cols)

        kwargs = dict(self.make_subplots_kwargs or {})
        for forbidden_key in ("rows", "cols", "subplot_titles"):
            kwargs.pop(forbidden_key, None)

        grid = make_subplots(
            rows=total_rows,
            cols=total_cols,
            subplot_titles=titles,
            **kwargs,
        )

        seen_legend_names: set[str] = set()
        coloraxis_registry = ColoraxisRegistry()

        for spec in sorted_specs:
            render_session = session
            resolved_session_filter = _resolve_session_filter(spec)
            if resolved_session_filter:
                render_session = _apply_session_filter(session, resolved_session_filter)

            child_figure = _render_subplot_plot(
                plot=spec.plot,
                session=render_session,
                theme=theme,
                extra_value_filter=spec.value_filter,
            )

            remap_coloraxis(
                child_figure,
                coloraxis_registry,
                explicit_share_keys=extract_coloraxis_share_keys(spec.plot),
                legacy_share_name=extract_legacy_coloraxis_name(spec.plot),
            )

            add_figure_to_subplot(
                grid,
                child_figure,
                row=spec.row,
                col=spec.col,
                transfer_layout=True,
                seen_legend_names=seen_legend_names,
            )

        apply_coloraxis_configs(grid, coloraxis_registry)

        if theme is not None:
            apply_theme(grid, theme)

        distribute_colorbars(grid)

        return grid


def _build_subplot_titles(specs: list[SubplotSpec], rows: int, cols: int) -> list[str]:
    titles = [""] * (rows * cols)
    for spec in specs:
        idx = (spec.row - 1) * cols + (spec.col - 1)
        existing = titles[idx]
        if existing and spec.title and existing != spec.title:
            raise ValueError(
                "Conflicting subplot titles for the same cell "
                f"(row={spec.row}, col={spec.col}): '{existing}' vs '{spec.title}'."
            )
        if not existing and spec.title:
            titles[idx] = spec.title
    return titles


def _validate_subplots(subplots: list[SubplotSpec]) -> None:
    for spec in subplots:
        if spec.row < 1 or spec.col < 1:
            raise ValueError(
                "Subplot row/col must be 1-indexed positive integers; "
                f"got row={spec.row}, col={spec.col}."
            )

        filter_dict = _resolve_session_filter(spec)
        if filter_dict:
            invalid_keys = sorted(set(filter_dict) - VALID_SESSION_FILTER_KEYS)
            if invalid_keys:
                valid_keys = sorted(VALID_SESSION_FILTER_KEYS)
                raise ValueError(
                    f"Invalid filter keys {invalid_keys} in subplot filter. "
                    f"Valid keys are: {valid_keys}"
                )


def _resolve_session_filter(spec: SubplotSpec) -> dict[str, Any] | None:
    if spec.session_filter is None:
        return spec.filter
    if spec.filter is None:
        return spec.session_filter

    merged = dict(spec.filter)
    merged.update(spec.session_filter)
    return merged


def _render_subplot_plot(
    *,
    plot: Plot,
    session: Session,
    theme: Theme | None,
    extra_value_filter: dict[str, tuple[str, Any]] | None,
) -> go.Figure:
    original_value_filter = getattr(plot, "value_filter", VALUE_FILTER_SENTINEL)

    if original_value_filter is not VALUE_FILTER_SENTINEL:
        merged = _merge_value_filters(original_value_filter, extra_value_filter)
        plot.value_filter = merged

    try:
        return _render_plot_with_optional_theme(plot, session, theme)
    finally:
        if original_value_filter is not VALUE_FILTER_SENTINEL:
            plot.value_filter = original_value_filter


def _render_plot_with_optional_theme(
    plot: Plot,
    session: Session,
    theme: Theme | None,
) -> go.Figure:
    if theme is None:
        return plot.render(session)

    if _plot_supports_theme_kwarg(plot):
        return plot.render(session, theme=theme)
    return plot.render(session)


def _plot_supports_theme_kwarg(plot: Plot) -> bool:
    try:
        signature = inspect.signature(plot.render)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "theme":
            return True

    return False


def _merge_value_filters(
    base: dict[str, tuple[str, Any]] | None,
    extra: dict[str, tuple[str, Any]] | None,
) -> dict[str, tuple[str, Any]] | None:
    if base is None and extra is None:
        return None

    merged: dict[str, tuple[str, Any]] = {}
    if base:
        merged.update(base)
    if extra:
        merged.update(extra)
    return merged


def _apply_session_filter(session: Session, filter_dict: dict[str, Any]) -> Session:
    return session.filter(
        param_ids=filter_dict.get("param_ids"),
        param_names=filter_dict.get("param_names"),
        param_types=filter_dict.get("param_types"),
        model_ids=filter_dict.get("model_ids"),
    )
