"""Single identity resolver for user-facing field addresses.

Maps every accepted address form — a plain field name, a legacy contextual
grammar string (``metric@models[m1,m2]@params[p1]``), a viz ``FieldRef``,
or a :class:`FieldSelector` — to structured selector components. This
module is the only interpreter of the legacy grammar; :func:`render_label`
renders labels from structure, never parses them back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FieldSelector",
    "render_label",
    "resolve",
]

_SEPARATOR = "@"
_MODELS_KEY = "models"
_PARAMS_KEY = "params"
_MODELS_PART = re.compile(rf"^{_MODELS_KEY}\[(.+)\]$")
_PARAMS_PART = re.compile(rf"^{_PARAMS_KEY}\[(.+)\]$")


@dataclass(frozen=True)
class FieldSelector:
    """Structured address of a stored value.

    Attributes:
        field: Base field name; never a grammar string.
        models: Model names constraining the address, or None when
            unconstrained.
        params: Parameter names constraining the address, or None when
            unconstrained.
        steps: Training-step axis slot; reserved, always None for now.
        roles: Directed role slots; reserved, always None for now.
    """

    field: str
    models: tuple[str, ...] | None = None
    params: tuple[str, ...] | None = None
    steps: tuple[int, ...] | None = None
    roles: tuple[str, ...] | None = None

    @property
    def is_contextual(self) -> bool:
        """True when the address names an aggregate context."""
        return self.models is not None or self.params is not None


def resolve(address: str | FieldSelector | Any) -> FieldSelector:
    """Resolve a user-facing field address into a structured selector.

    Args:
        address: A field name, a legacy contextual grammar string, an
            object carrying a string ``field`` attribute (``FieldRef``),
            or an already-built ``FieldSelector``.

    Returns:
        The structured selector, with component order as written.

    Raises:
        TypeError: If the address is none of the accepted kinds.
    """
    if isinstance(address, FieldSelector):
        return address
    if isinstance(address, str):
        return _resolve_string(address)
    ref_field = getattr(address, "field", None)
    if isinstance(ref_field, str):
        return _resolve_string(ref_field)
    msg = (
        f"Cannot resolve a field address of type {type(address).__name__}; "
        "expected a field name string, a FieldRef, or a FieldSelector"
    )
    raise TypeError(msg)


def render_label(selector: FieldSelector) -> str:
    """Render the canonical display label for a selector.

    Labels match the legacy contextual grammar (context members sorted)
    and so coincide with the names stored by existing sessions.

    Args:
        selector: Structured address to render.

    Returns:
        The base field name, plus sorted ``models[...]`` / ``params[...]``
        parts when the selector is contextual.
    """
    parts = [selector.field]
    if selector.models:
        parts.append(f"{_MODELS_KEY}[{','.join(sorted(selector.models))}]")
    if selector.params:
        parts.append(f"{_PARAMS_KEY}[{','.join(sorted(selector.params))}]")
    return _SEPARATOR.join(parts)


def _resolve_string(address: str) -> FieldSelector:
    """Parse a raw string address; the single interpreter of the grammar.

    A string without the separator is a plain field name; suffix parts
    that are not canonical ``models[...]`` / ``params[...]`` groups
    contribute no context.
    """
    if _SEPARATOR not in address:
        return FieldSelector(field=address)

    parts = address.split(_SEPARATOR)
    models: tuple[str, ...] | None = None
    params: tuple[str, ...] | None = None

    for part in parts[1:]:
        if match := _MODELS_PART.match(part):
            models = tuple(match.group(1).split(","))
        elif match := _PARAMS_PART.match(part):
            params = tuple(match.group(1).split(","))

    return FieldSelector(field=parts[0], models=models, params=params)
