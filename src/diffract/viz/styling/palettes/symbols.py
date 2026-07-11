from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

MARKER_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "star",
    "hourglass",
    "bowtie",
]


class SymbolPalette(Protocol):
    """Protocol for marker symbol palettes."""

    @property
    def symbols(self) -> list[str]:
        """List of symbols in the palette."""
        ...


@dataclass(slots=True)
class DefaultSymbolPalette:
    """Default marker symbol palette."""

    _symbols: list[str] = field(default_factory=lambda: MARKER_SYMBOLS.copy())

    @property
    def symbols(self) -> list[str]:
        """List of symbols in the palette."""
        return self._symbols
