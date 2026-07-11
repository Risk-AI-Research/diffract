from __future__ import annotations

import re
from typing import Any

from diffract.core.constants import CONTEXT_SEPARATOR, parse_contextual_field_name

from .detection import detect_field_meta
from .types import DataShape, DataType, Entry, EntryContext


def get_field_value(ctx: EntryContext, field: str) -> Any:
    """Return the value of a field, resolving contextual field variants.

    Args:
        ctx: Entry context to look the field up in.
        field: Field name to resolve.

    Returns:
        The direct field value if present, otherwise the best-matching
        contextual field value.

    Raises:
        ValueError: If no matching field is found in the context.
    """
    value = ctx.fields.get(field)
    if value is not None:
        return value

    agg_pattern = re.compile(rf"^{re.escape(field)}{re.escape(CONTEXT_SEPARATOR)}.+$")

    candidates: list[tuple[str, Any]] = []
    for key_, value_ in ctx.fields.items():
        if agg_pattern.match(key_):
            candidates.append((key_, value_))

    if not candidates:
        raise ValueError(f"Field {field} not found in entry context")

    return _select_best_contextual_field(candidates, ctx=ctx)


def get_field_values(entries: dict[str, Entry], field: str) -> list[Any]:
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
    entries: dict[str, Entry], field: str
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
    candidates: list[tuple[str, Any]],
    *,
    ctx: EntryContext,
) -> Any:
    """Select best contextual field from multiple candidates.

    Priority:
    1. Fields whose context includes current parameter
    2. Among those, prefer smaller context size (more specific)
    3. Fallback: deterministic sort by field name
    """
    scored: list[tuple[int, int, str, Any]] = []

    for field_name, value in candidates:
        _, models, params = parse_contextual_field_name(field_name)

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
