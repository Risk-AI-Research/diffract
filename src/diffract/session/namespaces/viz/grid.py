"""Grid plot wrappers for Session.viz."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diffract.session.namespaces.viz._utils import _to_field_ref

if TYPE_CHECKING:
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.plots.base.plot import Plot
    from diffract.viz.plots.subplots import GridAxisBind, GridCellRule, SubplotSpec
    from diffract.viz.styling import Theme


def grid(
    self: Any,
    *,
    subplots: list[SubplotSpec],
    make_subplots_kwargs: dict[str, Any] | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Create and render a `GridPlot` from explicit subplot specifications."""
    from diffract.viz.plots.subplots import GridPlot

    plot = GridPlot(
        subplots=subplots,
        make_subplots_kwargs=dict(make_subplots_kwargs or {}),
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)


def bound_grid(
    self: Any,
    *,
    plot_template: Plot,
    row: GridAxisBind | None = None,
    col: GridAxisBind | None = None,
    cell_rules: list[GridCellRule] | None = None,
    title_template: str | None = None,
    make_subplots_kwargs: dict[str, Any] | None = None,
    base_session_filter: dict[str, Any] | None = None,
    base_value_filter: dict[str, tuple[str, Any]] | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Create and render a bound grid generated from row/column axis binds.

    For `plot` binds targeting `FieldRef` plot attributes, plain string bind
    values are automatically converted into `FieldRef(field=...)`. The same
    conversion is applied to `cell_rules[*].plot` overrides.
    """
    from diffract.viz.plots.subplots import build_bound_grid

    resolved_row = _coerce_plot_bind_field_refs(plot_template=plot_template, bind=row)
    resolved_col = _coerce_plot_bind_field_refs(plot_template=plot_template, bind=col)
    resolved_rules = _coerce_rule_plot_field_refs(
        plot_template=plot_template,
        rules=cell_rules,
    )

    plot = build_bound_grid(
        plot_template=plot_template,
        row=resolved_row,
        col=resolved_col,
        cell_rules=resolved_rules,
        title_template=title_template,
        make_subplots_kwargs=make_subplots_kwargs,
        base_session_filter=base_session_filter,
        base_value_filter=base_value_filter,
    )
    return self.draw(plot=plot, theme=theme, theme_path=theme_path)


def _coerce_plot_bind_field_refs(
    *,
    plot_template: Plot,
    bind: GridAxisBind | None,
) -> GridAxisBind | None:
    """Convert string plot-bind values to FieldRef when template field is FieldRef."""
    if bind is None or bind.target != "plot":
        return bind

    if not hasattr(plot_template, bind.key):
        return bind

    from diffract.viz.data import FieldRef

    template_value = getattr(plot_template, bind.key)
    if not isinstance(template_value, FieldRef):
        return bind

    changed = False
    converted_values: list[Any] = []
    for value in bind.values:
        if isinstance(value, str):
            converted_values.append(_to_field_ref(value))
            changed = True
            continue
        converted_values.append(value)

    if not changed:
        return bind

    return replace(bind, values=tuple(converted_values))


def _coerce_rule_plot_field_refs(
    *,
    plot_template: Plot,
    rules: list[GridCellRule] | None,
) -> list[GridCellRule] | None:
    """Convert string `cell_rules[*].plot` values to FieldRef when needed."""
    if not rules:
        return rules

    changed = False
    converted_rules: list[GridCellRule] = []
    for rule in rules:
        converted_rule, rule_changed = _coerce_single_rule_plot_field_refs(
            plot_template=plot_template,
            rule=rule,
        )
        converted_rules.append(converted_rule)
        changed = changed or rule_changed

    if not changed:
        return rules
    return converted_rules


def _coerce_single_rule_plot_field_refs(
    *,
    plot_template: Plot,
    rule: GridCellRule,
) -> tuple[GridCellRule, bool]:
    """Convert one rule's `plot` overrides where template fields are FieldRef."""
    if not rule.plot:
        return rule, False

    from diffract.viz.data import FieldRef

    changed = False
    converted_plot = dict(rule.plot)
    for key, value in rule.plot.items():
        if not hasattr(plot_template, key):
            continue

        template_value = getattr(plot_template, key)
        if isinstance(template_value, FieldRef) and isinstance(value, str):
            converted_plot[key] = _to_field_ref(value)
            changed = True

    if not changed:
        return rule, False
    return replace(rule, plot=converted_plot), True
