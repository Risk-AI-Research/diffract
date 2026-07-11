from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayoutStyle:
    """Figure layout dimensions and margins."""

    width: int | None = None
    height: int | None = None
    margin: dict[str, int] = field(
        default_factory=lambda: {"l": 80, "r": 40, "t": 60, "b": 80}
    )


@dataclass
class TypographyStyle:
    """Font settings for the figure."""

    font_family: str = "Times New Roman"
    title_font_size: int = 16
    label_font_size: int = 14
    tick_font_size: int = 12


@dataclass
class BackgroundStyle:
    """Background colors for the figure."""

    plot_bgcolor: str = "white"
    paper_bgcolor: str = "white"


@dataclass
class AxesStyle:
    """Default axes styling (can be overridden by SupportsAxis fields)."""

    grid_color: str = "lightgrey"
    line_color: str = "black"
    show_grid: bool = True
    show_line: bool = True
    mirror: bool = True


@dataclass
class LegendStyle:
    """Legend styling."""

    bgcolor: str = "rgba(255,255,255,0.9)"
    border_color: str = "gray"
    border_width: int = 1
    font_size: int = 12


@dataclass
class ColorbarStyle:
    """Colorbar positioning and appearance."""

    orientation: str = "h"
    x: float = 0.5
    y: float = -0.15
    xanchor: str = "center"
    yanchor: str = "top"
    thickness: int = 15
    len: float = 0.5
