from __future__ import annotations

from dataclasses import dataclass, field

from diffract.viz.styling.palettes import PaletteBundle

from .components import (
    AxesStyle,
    BackgroundStyle,
    ColorbarStyle,
    LayoutStyle,
    LegendStyle,
    TypographyStyle,
)


@dataclass
class Theme:
    """Complete theme configuration with composition.

    Theme provides figure-level styling and palette configuration.
    Explicit user settings on Plot/Configurator fields take precedence.

    Example:
        >>> from diffract.viz.styling.theme import Theme, DARK_THEME
        >>> plot = BoxPlot(x=FieldRef("model"), y=FieldRef("accuracy"))
        >>> fig = plot.render(session, theme=DARK_THEME)
    """

    layout: LayoutStyle = field(default_factory=LayoutStyle)
    typography: TypographyStyle = field(default_factory=TypographyStyle)
    background: BackgroundStyle = field(default_factory=BackgroundStyle)
    axes: AxesStyle = field(default_factory=AxesStyle)
    legend: LegendStyle = field(default_factory=LegendStyle)
    colorbar: ColorbarStyle = field(default_factory=ColorbarStyle)
    palettes: PaletteBundle = field(default_factory=PaletteBundle)
