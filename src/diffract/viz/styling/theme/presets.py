from __future__ import annotations

from diffract.viz.styling.palettes import PaletteBundle
from diffract.viz.styling.palettes.color import DarkColorPalette

from .components import (
    AxesStyle,
    BackgroundStyle,
    LayoutStyle,
    LegendStyle,
    TypographyStyle,
)
from .theme import Theme

DEFAULT_THEME = Theme()


DARK_THEME = Theme(
    background=BackgroundStyle(
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#2d2d2d",
    ),
    axes=AxesStyle(
        grid_color="#444444",
        line_color="#666666",
        mirror=False,
    ),
    legend=LegendStyle(
        bgcolor="rgba(45,45,45,0.9)",
        border_color="#666666",
    ),
    palettes=PaletteBundle(
        color=DarkColorPalette(),
    ),
)


MINIMAL_THEME = Theme(
    layout=LayoutStyle(
        margin={"l": 50, "r": 20, "t": 40, "b": 50},
    ),
    typography=TypographyStyle(
        font_family="Arial",
        title_font_size=14,
        label_font_size=12,
        tick_font_size=10,
    ),
    axes=AxesStyle(
        show_grid=False,
        show_line=False,
        mirror=False,
    ),
    legend=LegendStyle(
        bgcolor="rgba(255,255,255,0)",
        border_color="rgba(0,0,0,0)",
        border_width=0,
    ),
)
