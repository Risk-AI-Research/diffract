from __future__ import annotations

from typing import Any

import numpy as np

from .extraction import get_field_value
from .types import Entry, EntryContext


def apply_filter(
    entries: dict[str, Entry],
    value_filter: dict[str, tuple[str, Any]],
) -> dict[str, Entry]:
    """Apply value-based filtering to entries."""
    return {
        uid: entry
        for uid, entry in entries.items()
        if _entry_passes_filter(entry, value_filter)
    }


def _entry_passes_filter(
    entry: Entry,
    value_filter: dict[str, tuple[str, Any]],
) -> bool:
    """Check if an entry passes all filter conditions."""
    ctx = EntryContext.from_entry(entry)
    for field_name, (op, threshold) in value_filter.items():
        value = get_field_value(ctx, field_name)
        if not _check_condition(value, op, threshold):
            return False
    return True


def _check_condition(value: Any, op: str, threshold: Any) -> bool:
    """Check a single filter condition."""
    if value is None:
        return False

    if isinstance(value, np.ndarray):
        value = float(np.nanmean(value))

    ops = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    op_func = ops.get(op)
    if op_func is None:
        raise ValueError(f"Unknown operator: {op}")

    return op_func(value, threshold)
