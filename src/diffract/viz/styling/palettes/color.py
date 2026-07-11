from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ColorPalette(Protocol):
    """Protocol for mapping categorical values to colors."""

    @property
    def colors(self) -> list[str]:
        """List of colors in the palette."""
        ...

    def get_color(self, value: Any, all_values: list[Any]) -> str:
        """Get color for a specific value given all values."""
        ...


@dataclass(slots=True)
class DefaultColorPalette:
    """Default discrete color palette."""

    _colors: list[str] = field(
        default_factory=lambda: [
            "navy",
            "crimson",
            "green",
            "chocolate",
            "orange",
            "violet",
            "purple",
            "blue",
            "grey",
            "teal",
        ]
    )

    @property
    def colors(self) -> list[str]:
        """List of colors in the palette."""
        return self._colors

    def get_color(self, value: Any, all_values: list[Any]) -> str:
        """Return the color assigned to a value given all values."""
        try:
            idx = list(all_values).index(value)
        except ValueError:
            idx = 0
        return self._colors[idx % len(self._colors)]


@dataclass(slots=True)
class DarkColorPalette:
    """Color palette optimized for dark themes."""

    _colors: list[str] = field(
        default_factory=lambda: [
            "#5dade2",
            "#f1948a",
            "#58d68d",
            "#f7dc6f",
            "#bb8fce",
            "#85c1e9",
            "#f8c471",
            "#abebc6",
            "#d7bde2",
            "#aed6f1",
        ]
    )

    @property
    def colors(self) -> list[str]:
        """List of colors in the palette."""
        return self._colors

    def get_color(self, value: Any, all_values: list[Any]) -> str:
        """Return the color assigned to a value given all values."""
        try:
            idx = list(all_values).index(value)
        except ValueError:
            idx = 0
        return self._colors[idx % len(self._colors)]
