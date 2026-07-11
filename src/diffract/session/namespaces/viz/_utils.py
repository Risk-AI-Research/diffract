"""Shared helpers for viz namespace wrappers."""

from __future__ import annotations

from typing import Any


def _to_field_ref(value: str | Any) -> Any:
    """Convert str to FieldRef for plot constructor; pass through otherwise."""
    from diffract.viz.data import FieldRef

    if isinstance(value, str):
        return FieldRef(field=value)
    return value
