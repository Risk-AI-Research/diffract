"""Subplot composition utilities for `viz` plots."""

from __future__ import annotations

from .coloraxis import ColoraxisRegistry
from .factory import (
    BindTarget,
    CellSelector,
    CellWhere,
    GridAxisBind,
    GridCellRule,
    ValueFilterOperator,
    build_bound_grid,
)
from .grid import GridPlot
from .layout import add_figure_to_subplot, transfer_layout_to_subplot
from .spec import SubplotSpec

__all__ = [
    "BindTarget",
    "CellSelector",
    "CellWhere",
    "ColoraxisRegistry",
    "GridAxisBind",
    "GridCellRule",
    "GridPlot",
    "SubplotSpec",
    "ValueFilterOperator",
    "add_figure_to_subplot",
    "build_bound_grid",
    "transfer_layout_to_subplot",
]
