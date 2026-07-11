"""Factory helpers for building bound subplot grids."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

from diffract.viz.plots.base.plot import Plot
from diffract.viz.styling import Theme

from .grid import GridPlot
from .spec import VALID_SESSION_FILTER_KEYS, SubplotSpec

BindTarget = Literal["plot", "session_filter", "value_filter"]
ValueFilterOperator = Literal[">", "<", ">=", "<=", "==", "!="]
CellWhere = Literal["all", "first_row", "last_row", "first_col", "last_col"]

VALID_BIND_TARGETS = frozenset({"plot", "session_filter", "value_filter"})
VALID_VALUE_FILTER_OPERATORS = frozenset({">", "<", ">=", "<=", "==", "!="})
VALID_CELL_WHERE = frozenset({"all", "first_row", "last_row", "first_col", "last_col"})
_FILTER_CONDITION_ITEM_COUNT = 2


@dataclass(kw_only=True)
class GridAxisBind:
    """Axis binding definition for grid generation.

    A bind maps a sequence of values onto either rows or columns in a generated
    `GridPlot`, overriding one target key per subplot cell.

    Args:
        target: Target domain to override.
            - "plot": set an attribute on the per-cell plot copy.
            - "session_filter": set a key in `SubplotSpec.session_filter`.
            - "value_filter": set a key in `SubplotSpec.value_filter`.
        key: Target key within the selected domain.
        values: Ordered values to place along the axis.
        labels: Optional display labels for titles. If omitted, labels are
            derived from values via `str(value)`.
        op: Optional operator for `value_filter` binds. If provided, each entry
            in `values` is treated as a threshold and converted to
            `(op, threshold)`. If omitted, each value must already be
            `(operator, threshold)`.
    """

    target: BindTarget
    key: str
    values: Sequence[Any]
    labels: Sequence[str] | None = None
    op: ValueFilterOperator | None = None


@dataclass(kw_only=True)
class CellSelector:
    """Select cells where a rule should be applied.

    Args:
        where: Predefined positional selector.
        row: Optional explicit row index (1-based).
        col: Optional explicit column index (1-based).
    """

    where: CellWhere = "all"
    row: int | None = None
    col: int | None = None


@dataclass(kw_only=True)
class GridCellRule:
    """Conditional overrides applied to selected grid cells.

    Args:
        selector: Which cells to target.
        plot: Direct plot-attribute overrides for matching cells.
        plot_format: Format-string plot overrides. Values are formatted with
            placeholders from cell context:
            row, col, n_rows, n_cols,
            row_label, col_label, row_value, col_value, row_key, col_key,
            is_first_row, is_last_row, is_first_col, is_last_col.
        session_filter: Extra per-cell session filter values.
        value_filter: Extra per-cell value filter conditions.
    """

    selector: CellSelector = field(default_factory=CellSelector)
    plot: dict[str, Any] | None = None
    plot_format: dict[str, str] | None = None
    session_filter: dict[str, Any] | None = None
    value_filter: dict[str, tuple[str, Any] | list[Any]] | None = None


@dataclass
class _AxisPoint:
    value: Any
    label: str


@dataclass
class _CellContext:
    row: int
    col: int
    n_rows: int
    n_cols: int
    row_bind: GridAxisBind | None
    col_bind: GridAxisBind | None
    row_point: _AxisPoint
    col_point: _AxisPoint


def build_bound_grid(
    *,
    plot_template: Plot,
    row: GridAxisBind | None = None,
    col: GridAxisBind | None = None,
    cell_rules: Sequence[GridCellRule] | None = None,
    title_template: str | None = None,
    make_subplots_kwargs: dict[str, Any] | None = None,
    theme: Theme | None = None,
    base_session_filter: dict[str, Any] | None = None,
    base_value_filter: dict[str, tuple[str, Any]] | None = None,
) -> GridPlot:
    """Build a `GridPlot` by binding row/column values onto subplot templates.

    At least one axis bind (`row` or `col`) is required. For each generated cell:
    1. The plot template is deep-copied.
    2. Row bind (if provided) is applied.
    3. Column bind (if provided) is applied.
    4. Matching `cell_rules` are applied in order (last write wins).
    5. A `SubplotSpec` is created.

    Args:
        plot_template: Base plot object copied for each subplot cell.
        row: Optional row-axis bind.
        col: Optional column-axis bind.
        cell_rules: Optional positional rules for per-cell overrides.
        title_template: Optional format string for subplot titles.
            Available placeholders:
            - row_label, col_label
            - row_value, col_value
            - row_key, col_key
            - row, col
            If omitted, titles default to:
            - "{row_label} | {col_label}" when both binds exist,
            - "{row_label}" or "{col_label}" for single-axis grids.
        make_subplots_kwargs: Extra kwargs passed into `GridPlot`.
        theme: Optional theme for the resulting `GridPlot`.
        base_session_filter: Optional session filter merged into each subplot.
        base_value_filter: Optional value filter merged into each subplot.

    Returns:
        A `GridPlot` with auto-generated `SubplotSpec` entries.
    """
    _validate_axes(row=row, col=col)
    _validate_cell_rules(cell_rules=cell_rules, plot_template=plot_template)

    row_points = _materialize_axis_points(row)
    col_points = _materialize_axis_points(col)
    n_rows = len(row_points)
    n_cols = len(col_points)

    subplots: list[SubplotSpec] = []
    for row_idx, row_point in enumerate(row_points, start=1):
        for col_idx, col_point in enumerate(col_points, start=1):
            plot = deepcopy(plot_template)
            session_filter = dict(base_session_filter or {})
            value_filter = dict(base_value_filter or {})

            _apply_axis_bind(
                bind=row,
                point=row_point,
                plot=plot,
                session_filter=session_filter,
                value_filter=value_filter,
            )
            _apply_axis_bind(
                bind=col,
                point=col_point,
                plot=plot,
                session_filter=session_filter,
                value_filter=value_filter,
            )

            context = _CellContext(
                row=row_idx,
                col=col_idx,
                n_rows=n_rows,
                n_cols=n_cols,
                row_bind=row,
                col_bind=col,
                row_point=row_point,
                col_point=col_point,
            )
            _apply_cell_rules(
                rules=cell_rules,
                context=context,
                plot=plot,
                session_filter=session_filter,
                value_filter=value_filter,
            )

            title = _build_title(
                row=row_idx,
                col=col_idx,
                row_bind=row,
                col_bind=col,
                row_point=row_point,
                col_point=col_point,
                title_template=title_template,
            )

            subplots.append(
                SubplotSpec(
                    row=row_idx,
                    col=col_idx,
                    title=title,
                    plot=plot,
                    session_filter=session_filter or None,
                    value_filter=value_filter or None,
                )
            )

    return GridPlot(
        subplots=subplots,
        make_subplots_kwargs=dict(make_subplots_kwargs or {}),
    )


def _validate_axes(*, row: GridAxisBind | None, col: GridAxisBind | None) -> None:
    if row is None and col is None:
        raise ValueError("At least one bind must be provided: row=... or col=....")

    _validate_bind(row, axis_name="row")
    _validate_bind(col, axis_name="col")

    if row is not None and col is not None:
        same_target = row.target == col.target
        same_key = row.key == col.key
        if same_target and same_key:
            raise ValueError(
                "Row and column binds target the same key "
                f"('{row.target}.{row.key}'). Use different keys per axis."
            )


def _validate_bind(bind: GridAxisBind | None, *, axis_name: str) -> None:
    if bind is None:
        return

    if bind.target not in VALID_BIND_TARGETS:
        valid_targets = sorted(VALID_BIND_TARGETS)
        raise ValueError(
            f"Invalid target '{bind.target}' in {axis_name} bind. "
            f"Valid targets: {valid_targets}."
        )

    if not bind.key:
        raise ValueError(f"{axis_name} bind key must be non-empty.")

    if _is_scalar_like(bind.values):
        raise TypeError(
            f"{axis_name} bind values must be a sequence, got scalar "
            f"{type(bind.values).__name__}."
        )
    if len(bind.values) == 0:
        raise ValueError(f"{axis_name} bind values must not be empty.")

    if bind.labels is not None:
        if _is_scalar_like(bind.labels):
            raise TypeError(
                f"{axis_name} bind labels must be a sequence of strings, got scalar "
                f"{type(bind.labels).__name__}."
            )
        if len(bind.labels) != len(bind.values):
            raise ValueError(
                f"{axis_name} bind labels length must match values length: "
                f"{len(bind.labels)} != {len(bind.values)}."
            )

    if bind.target == "session_filter":
        if bind.key not in VALID_SESSION_FILTER_KEYS:
            valid_keys = sorted(VALID_SESSION_FILTER_KEYS)
            raise ValueError(
                f"Invalid session_filter key '{bind.key}' for {axis_name} bind. "
                f"Valid keys: {valid_keys}."
            )
        if bind.op is not None:
            raise ValueError(
                f"{axis_name} bind uses op='{bind.op}' with target='session_filter'. "
                "Operator is supported only for target='value_filter'."
            )
        return

    if bind.target == "plot":
        if bind.op is not None:
            raise ValueError(
                f"{axis_name} bind uses op='{bind.op}' with target='plot'. "
                "Operator is supported only for target='value_filter'."
            )
        return

    if bind.target == "value_filter":
        if bind.op is not None and bind.op not in VALID_VALUE_FILTER_OPERATORS:
            valid_ops = sorted(VALID_VALUE_FILTER_OPERATORS)
            raise ValueError(
                f"Invalid value_filter operator '{bind.op}' in {axis_name} bind. "
                f"Valid operators: {valid_ops}."
            )
        for value in bind.values:
            if bind.op is not None:
                continue
            _validate_value_filter_condition(value, axis_name=axis_name, key=bind.key)
        return


def _validate_cell_rules(
    *,
    cell_rules: Sequence[GridCellRule] | None,
    plot_template: Plot,
) -> None:
    if cell_rules is None:
        return

    if _is_scalar_like(cell_rules):
        raise TypeError(
            "cell_rules must be a sequence of GridCellRule, "
            f"got scalar {type(cell_rules).__name__}."
        )

    for idx, rule in enumerate(cell_rules, start=1):
        if not isinstance(rule, GridCellRule):
            raise TypeError(
                f"cell_rules[{idx}] must be GridCellRule, got {type(rule).__name__}."
            )
        _validate_cell_selector(rule.selector, rule_idx=idx)
        _validate_rule_plot_overrides(rule, plot_template=plot_template, rule_idx=idx)
        _validate_rule_session_filter(rule, rule_idx=idx)
        _validate_rule_value_filter(rule, rule_idx=idx)


def _validate_cell_selector(selector: CellSelector, *, rule_idx: int) -> None:
    if selector.where not in VALID_CELL_WHERE:
        valid_where = sorted(VALID_CELL_WHERE)
        raise ValueError(
            f"Invalid cell_rules[{rule_idx}].selector.where='{selector.where}'. "
            f"Valid values: {valid_where}."
        )

    for axis_name, value in (("row", selector.row), ("col", selector.col)):
        if value is None:
            continue
        if not _is_positive_int(value):
            raise ValueError(
                f"cell_rules[{rule_idx}].selector.{axis_name} must be positive "
                f"1-based index, got {value!r}."
            )


def _validate_rule_plot_overrides(
    rule: GridCellRule,
    *,
    plot_template: Plot,
    rule_idx: int,
) -> None:
    if rule.plot is not None and not isinstance(rule.plot, dict):
        raise TypeError(
            f"cell_rules[{rule_idx}].plot must be a dict, "
            f"got {type(rule.plot).__name__}."
        )
    if rule.plot_format is not None and not isinstance(rule.plot_format, dict):
        raise TypeError(
            f"cell_rules[{rule_idx}].plot_format must be a dict, "
            f"got {type(rule.plot_format).__name__}."
        )

    static_keys = set(rule.plot or {})
    formatted_keys = set(rule.plot_format or {})
    duplicate_keys = sorted(static_keys & formatted_keys)
    if duplicate_keys:
        raise ValueError(
            f"cell_rules[{rule_idx}] has duplicate keys in plot and plot_format: "
            f"{duplicate_keys}."
        )

    for key in sorted(static_keys | formatted_keys):
        if not key:
            raise ValueError(
                f"cell_rules[{rule_idx}] has empty key in plot/plot_format override."
            )
        if not hasattr(plot_template, key):
            available = _bindable_plot_keys(plot_template)
            raise ValueError(
                f"Unknown plot override key '{key}' in cell_rules[{rule_idx}] for "
                f"{type(plot_template).__name__}. Available fields: {available}."
            )

    for key, template in (rule.plot_format or {}).items():
        if not isinstance(template, str):
            raise TypeError(
                f"cell_rules[{rule_idx}].plot_format['{key}'] must be string "
                f"template, got {type(template).__name__}."
            )


def _validate_rule_session_filter(rule: GridCellRule, *, rule_idx: int) -> None:
    if rule.session_filter is None:
        return
    if not isinstance(rule.session_filter, dict):
        raise TypeError(
            f"cell_rules[{rule_idx}].session_filter must be a dict, "
            f"got {type(rule.session_filter).__name__}."
        )

    invalid_keys = sorted(set(rule.session_filter) - VALID_SESSION_FILTER_KEYS)
    if invalid_keys:
        valid_keys = sorted(VALID_SESSION_FILTER_KEYS)
        raise ValueError(
            f"Invalid session_filter keys {invalid_keys} in cell_rules[{rule_idx}]. "
            f"Valid keys: {valid_keys}."
        )


def _validate_rule_value_filter(rule: GridCellRule, *, rule_idx: int) -> None:
    if rule.value_filter is None:
        return
    if not isinstance(rule.value_filter, dict):
        raise TypeError(
            f"cell_rules[{rule_idx}].value_filter must be a dict, "
            f"got {type(rule.value_filter).__name__}."
        )

    for key, condition in rule.value_filter.items():
        _validate_value_filter_condition(
            condition,
            axis_name=f"cell_rules[{rule_idx}]",
            key=key,
        )


def _materialize_axis_points(bind: GridAxisBind | None) -> list[_AxisPoint]:
    if bind is None:
        return [_AxisPoint(value=None, label="")]

    labels = bind.labels
    points: list[_AxisPoint] = []
    for idx, value in enumerate(bind.values):
        label = labels[idx] if labels is not None else _default_label(value)
        points.append(_AxisPoint(value=value, label=label))
    return points


def _apply_axis_bind(
    *,
    bind: GridAxisBind | None,
    point: _AxisPoint,
    plot: Plot,
    session_filter: dict[str, Any],
    value_filter: dict[str, tuple[str, Any]],
) -> None:
    if bind is None:
        return

    if bind.target == "plot":
        _set_plot_attr(plot, key=bind.key, value=point.value)
        return

    if bind.target == "session_filter":
        session_filter[bind.key] = point.value
        return

    if bind.target == "value_filter":
        value_filter[bind.key] = _coerce_value_filter_condition(
            bind=bind,
            value=point.value,
        )
        return

    raise ValueError(
        f"Unsupported bind target '{bind.target}' while applying axis bind."
    )


def _apply_cell_rules(
    *,
    rules: Sequence[GridCellRule] | None,
    context: _CellContext,
    plot: Plot,
    session_filter: dict[str, Any],
    value_filter: dict[str, tuple[str, Any]],
) -> None:
    if not rules:
        return

    format_context = _build_cell_format_context(context)

    for rule in rules:
        if not _selector_matches(rule.selector, context):
            continue

        if rule.plot:
            for key, raw_value in rule.plot.items():
                _set_plot_attr(plot, key=key, value=raw_value)

        if rule.plot_format:
            for key, template in rule.plot_format.items():
                rendered = _format_template(
                    template=template,
                    format_context=format_context,
                    template_name=f"plot_format['{key}']",
                )
                _set_plot_attr(plot, key=key, value=rendered)

        if rule.session_filter:
            session_filter.update(rule.session_filter)

        if rule.value_filter:
            for key, raw_condition in rule.value_filter.items():
                value_filter[key] = _normalize_value_filter_condition(
                    raw_condition,
                    axis_name="cell rule",
                    key=key,
                )


def _selector_matches(selector: CellSelector, context: _CellContext) -> bool:
    if selector.row is not None and selector.row != context.row:
        return False
    if selector.col is not None and selector.col != context.col:
        return False

    where = selector.where
    if where == "all":
        return True
    if where == "first_row":
        return context.row == 1
    if where == "last_row":
        return context.row == context.n_rows
    if where == "first_col":
        return context.col == 1
    if where == "last_col":
        return context.col == context.n_cols
    return False


def _build_cell_format_context(context: _CellContext) -> dict[str, Any]:
    return {
        "row": context.row,
        "col": context.col,
        "n_rows": context.n_rows,
        "n_cols": context.n_cols,
        "row_label": context.row_point.label,
        "col_label": context.col_point.label,
        "row_value": context.row_point.value,
        "col_value": context.col_point.value,
        "row_key": context.row_bind.key if context.row_bind is not None else "",
        "col_key": context.col_bind.key if context.col_bind is not None else "",
        "is_first_row": context.row == 1,
        "is_last_row": context.row == context.n_rows,
        "is_first_col": context.col == 1,
        "is_last_col": context.col == context.n_cols,
    }


def _set_plot_attr(plot: Plot, *, key: str, value: Any) -> None:
    if not hasattr(plot, key):
        available = _bindable_plot_keys(plot)
        raise ValueError(
            f"Unknown plot bind key '{key}' for {type(plot).__name__}. "
            f"Available fields: {available}."
        )

    coerced_value = _coerce_plot_value(plot, key=key, value=value)
    setattr(plot, key, coerced_value)


def _coerce_plot_value(plot: Plot, *, key: str, value: Any) -> Any:
    from diffract.viz.data import FieldRef

    current_value = getattr(plot, key, None)
    if isinstance(current_value, FieldRef) and isinstance(value, str):
        return FieldRef(field=value)
    return value


def _coerce_value_filter_condition(
    *,
    bind: GridAxisBind,
    value: Any,
) -> tuple[str, Any]:
    if bind.op is not None:
        return (bind.op, value)

    return _normalize_value_filter_condition(value, axis_name="bind", key=bind.key)


def _normalize_value_filter_condition(
    value: Any,
    *,
    axis_name: str,
    key: str,
) -> tuple[str, Any]:
    if _is_scalar_like(value) or len(value) != _FILTER_CONDITION_ITEM_COUNT:
        raise TypeError(
            f"{axis_name} bind for value_filter key '{key}' expects each value to be "
            "(operator, threshold) pair when op is not provided."
        )

    operator = value[0]
    if not isinstance(operator, str) or operator not in VALID_VALUE_FILTER_OPERATORS:
        valid_ops = sorted(VALID_VALUE_FILTER_OPERATORS)
        raise ValueError(
            f"Invalid value_filter operator '{operator}' for key '{key}'. "
            f"Valid operators: {valid_ops}."
        )

    threshold = value[1]
    return (operator, threshold)


def _validate_value_filter_condition(
    value: Any,
    *,
    axis_name: str,
    key: str,
) -> None:
    _normalize_value_filter_condition(value, axis_name=axis_name, key=key)


def _build_title(
    *,
    row: int,
    col: int,
    row_bind: GridAxisBind | None,
    col_bind: GridAxisBind | None,
    row_point: _AxisPoint,
    col_point: _AxisPoint,
    title_template: str | None,
) -> str:
    row_label = row_point.label
    col_label = col_point.label

    if title_template is None:
        if row_bind is not None and col_bind is not None:
            return f"{row_label} | {col_label}"
        if row_bind is not None:
            return row_label
        if col_bind is not None:
            return col_label
        return ""

    format_context = {
        "row": row,
        "col": col,
        "row_label": row_label,
        "col_label": col_label,
        "row_value": row_point.value,
        "col_value": col_point.value,
        "row_key": row_bind.key if row_bind is not None else "",
        "col_key": col_bind.key if col_bind is not None else "",
    }
    return _format_template(
        template=title_template,
        format_context=format_context,
        template_name="title_template",
    )


def _format_template(
    *,
    template: str,
    format_context: dict[str, Any],
    template_name: str,
) -> str:
    try:
        return template.format(**format_context)
    except KeyError as exc:
        key = exc.args[0]
        available = sorted(format_context)
        raise ValueError(
            f"Unknown placeholder '{key}' in {template_name}. "
            f"Available placeholders: {available}."
        ) from exc


def _bindable_plot_keys(plot: Plot) -> list[str]:
    if is_dataclass(plot):
        return sorted(dataclass_field.name for dataclass_field in fields(plot))
    return sorted(name for name in dir(plot) if not name.startswith("_"))


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes)) or not isinstance(value, Sequence)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _default_label(value: Any) -> str:
    if isinstance(value, list) and len(value) == 1:
        return str(value[0])
    return str(value)


__all__ = [
    "BindTarget",
    "CellSelector",
    "CellWhere",
    "GridAxisBind",
    "GridCellRule",
    "ValueFilterOperator",
    "build_bound_grid",
]
