"""Typed sources for style properties that accept literals or field names.

Each alias annotates ``FieldRef | str | None`` with the kind of plotly
literal the property accepts. The renderer's config-time coercion pass
uses the annotation to disambiguate bare strings deterministically: a
string that is a valid literal of that kind stays a literal, anything
else becomes a ``FieldRef``. Explicit ``FieldRef`` objects (e.g. the
refs-style configs) bypass the rule entirely.
"""

from enum import Enum
from typing import Annotated

from diffract.viz.data import FieldRef


class StyleLiteralKind(Enum):
    """Kind of plotly literal a style property accepts besides a FieldRef."""

    COLOR = "color"
    SYMBOL = "symbol"
    DASH = "dash"


ColorSource = Annotated[FieldRef | str | None, StyleLiteralKind.COLOR]
SymbolSource = Annotated[FieldRef | str | None, StyleLiteralKind.SYMBOL]
DashSource = Annotated[FieldRef | str | None, StyleLiteralKind.DASH]
