from __future__ import annotations

from dataclasses import dataclass, field

from .color import ColorPalette, DefaultColorPalette
from .dashes import DashPalette, DefaultDashPalette
from .symbols import DefaultSymbolPalette, SymbolPalette


@dataclass(slots=True)
class PaletteBundle:
    """Collection of all palettes used for styling."""

    color: ColorPalette = field(default_factory=DefaultColorPalette)
    symbols: SymbolPalette = field(default_factory=DefaultSymbolPalette)
    dashes: DashPalette = field(default_factory=DefaultDashPalette)
