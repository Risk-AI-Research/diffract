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

from diffract.core.utils import imports as import_utils
from diffract.viz.data import FieldRef

_PLOTLY_GO = "plotly.graph_objects"


class StyleLiteralKind(Enum):
    """Kind of plotly literal a style property accepts besides a FieldRef."""

    COLOR = "color"
    SYMBOL = "symbol"
    DASH = "dash"


ColorSource = Annotated[FieldRef | str | None, StyleLiteralKind.COLOR]
SymbolSource = Annotated[FieldRef | str | None, StyleLiteralKind.SYMBOL]
DashSource = Annotated[FieldRef | str | None, StyleLiteralKind.DASH]


def is_style_literal(value: str, kind: StyleLiteralKind) -> bool:
    """Return True when ``value`` is a valid plotly literal of ``kind``.

    A string that is not a valid literal of ``kind`` is treated as a field name
    by the deterministic style-source rule.
    """
    go = import_utils.require(_PLOTLY_GO)
    try:
        if kind is StyleLiteralKind.COLOR:
            go.scatter.Marker(color=value)
        elif kind is StyleLiteralKind.SYMBOL:
            go.scatter.Marker(symbol=value)
        else:
            go.scatter.Line(dash=value)
    except ValueError:
        return False
    return True
