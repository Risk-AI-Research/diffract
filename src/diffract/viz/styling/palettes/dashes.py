from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

LINE_DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]


class DashPalette(Protocol):
    """Protocol for line dash pattern palettes."""

    @property
    def dashes(self) -> list[str]:
        """List of dash patterns in the palette."""
        ...


@dataclass(slots=True)
class DefaultDashPalette:
    """Default line dash palette."""

    _dashes: list[str] = field(default_factory=lambda: LINE_DASHES.copy())

    @property
    def dashes(self) -> list[str]:
        """List of dash patterns in the palette."""
        return self._dashes
