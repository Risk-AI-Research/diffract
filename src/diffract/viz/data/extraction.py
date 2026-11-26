from __future__ import annotations

from dataclasses import replace
from typing import Any

from diffract.session.resolver import FieldAddress, FieldSelector, render_label, resolve

from .detection import detect_field_meta
from .types import DataShape, DataType, Entry, EntryContext

_NAME_ALIASES = {"param_name", "parameter_name"}


class FieldNotFoundError(ValueError):
    """An entry has no value for a requested plot slot."""


def plot_address(field: FieldAddress) -> str | FieldSelector:
    """Map table coordinates to plot metadata, preserving selector constraints."""
    if isinstance(field, str) and "@" not in field:
        return "name" if field in _NAME_ALIASES else field
    selector = resolve(field)
    if selector.field in _NAME_ALIASES:
        if selector.is_contextual:
            return replace(selector, field="name")
        if not isinstance(field, str):
            return "name"
    return field if isinstance(field, str) else selector


def matching_fields(
    ctx: EntryContext, field: FieldAddress
) -> list[tuple[FieldSelector, str, Any]]:
    """Return the selector, label, and value for each matching field in this entry."""
    field = plot_address(field)
    if isinstance(field, str) and field in ctx.fields:
        return [(FieldSelector(field), field, ctx.fields[field])]
    requested = resolve(field)
    if requested.field in ctx.fields:
        scalar = FieldSelector(
            requested.field,
            (ctx.model_id,),
            (ctx.parameter_name,),
            (ctx.fields.get("step", 0),),
        )
        if requested.matches(scalar):
            return [(scalar, requested.field, ctx.fields[requested.field])]
    if ctx.field_contexts:
        return [
            (candidate, label, ctx.fields[label])
            for label, record in ctx.field_contexts.items()
            if requested.matches(candidate := FieldSelector.from_context(record))
        ]
    return [
        (candidate, key, value)
        for key, value in ctx.fields.items()
        if (candidate := resolve(key)).is_contextual and requested.matches(candidate)
    ]


def get_field_value(ctx: EntryContext, field: FieldAddress) -> Any:
    """Read one value and require an explicit operand order for directed pairs."""
    candidates = matching_fields(ctx, field)
    if not candidates:
        raise FieldNotFoundError(f"Field {field} not found in entry context")
    directions = {
        (s.roles, s.steps)
        for s, label, _ in candidates
        if s.roles and not ctx.field_contexts.get(label, {}).get("symmetric")
    }
    if len(directions) > 1:
        choices = ", ".join(repr(s) for s, _, _ in candidates if s.roles)
        raise ValueError(
            f"Ambiguous directed field. Select one ordered pair with {choices}"
        )
    return _select_best_contextual_field(candidates, ctx=ctx)


def field_label(entries: dict[str, Entry], field: FieldAddress) -> str:
    """Return the current display label for a plot slot."""
    requested = resolve(field)
    labels = {
        label
        for entry in entries.values()
        for label, record in entry.get("field_contexts", {}).items()
        if requested.matches(FieldSelector.from_context(record))
    }
    return next(iter(labels)) if len(labels) == 1 else render_label(requested)


def get_field_values(entries: dict[str, Entry], field: FieldAddress) -> list[Any]:
    """Return the field value for each entry.

    Args:
        entries: Mapping of uid to entry.
        field: Field name to extract.

    Returns:
        List of field values, one per entry.
    """
    values: list[Any] = []

    for entry in entries.values():
        value = get_field_value(EntryContext.from_entry(entry), field)
        values.append(value)

    return values


def get_field_data(
    entries: dict[str, Entry], field: FieldAddress
) -> tuple[list[Any], DataType, DataShape]:
    """Return field values along with their detected type and shape.

    Args:
        entries: Mapping of uid to entry.
        field: Field name to extract.

    Returns:
        Tuple of (values, data type, data shape).
    """
    values = get_field_values(entries, field)
    meta = detect_field_meta(values)
    return values, meta.data_type, meta.data_shape


def _select_best_contextual_field(
    candidates: list[tuple[FieldSelector, str, Any]],
    *,
    ctx: EntryContext,
) -> Any:
    """Select best contextual field from multiple candidates.

    Priority:
    1. Fields whose context includes current parameter
    2. Among those, prefer smaller context size (more specific)
    3. Fallback: deterministic sort by field name
    """
    scored: list[tuple[int, int, int, str, Any]] = []

    for selector, field_name, value in candidates:
        models = selector.models
        params = selector.params

        model_match = models is None or (
            ctx.model_id is not None and ctx.model_id in models
        )
        param_match = params is None or (
            ctx.parameter_name is not None and ctx.parameter_name in params
        )

        context_size = (len(models) if models else 0) + (len(params) if params else 0)

        scored.append(
            (
                1 if model_match else 0,
                1 if param_match else 0,
                -context_size,
                field_name,
                value,
            )
        )

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    return scored[0][4]
