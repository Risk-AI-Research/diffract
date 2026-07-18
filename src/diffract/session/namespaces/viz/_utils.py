"""Shared helpers for viz namespace wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffract.viz.styling.sources import StyleLiteralKind


def _to_field_ref(value: str | Any) -> Any:
    """Convert str to FieldRef for plot constructor; pass through otherwise."""
    from diffract.viz.data import FieldRef

    if isinstance(value, str):
        return FieldRef(field=value)
    return value


def _to_style_source(value: Any, kind: StyleLiteralKind) -> Any:
    """Coerce a bare field name to FieldRef; keep valid plotly literals as-is.

    Applies the deterministic style-source rule: a string that is a valid plotly
    literal of ``kind`` stays a literal, any other string becomes a ``FieldRef``,
    and non-string values (``FieldRef``/``None``) pass through unchanged.
    """
    from diffract.viz.styling.sources import is_style_literal

    if isinstance(value, str) and not is_style_literal(value, kind):
        from diffract.viz.data import FieldRef

        return FieldRef(field=value)
    return value
