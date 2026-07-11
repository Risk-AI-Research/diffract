"""Subplot specifications and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diffract.viz.plots.base.plot import Plot

VALID_SESSION_FILTER_KEYS = frozenset(
    {"param_ids", "param_names", "param_types", "model_ids"}
)
VALUE_FILTER_SENTINEL = object()


@dataclass(slots=True)
class SubplotSpec:
    """Specification for a single subplot in a `GridPlot`.

    Args:
        row: Row position (1-indexed).
        col: Column position (1-indexed).
        title: Title for this subplot.
        plot: Plot object to render.
        session_filter: Optional dict forwarded to `Session.filter(...)`.
            Valid keys: param_ids, param_names, param_types, model_ids.
        value_filter: Optional value filter merged into `plot.value_filter` for
            this subplot only: {field: (op, threshold)}.
        filter: Backward-compatible alias for `session_filter`.
    """

    row: int
    col: int
    title: str
    plot: Plot

    session_filter: dict[str, Any] | None = None
    value_filter: dict[str, tuple[str, Any]] | None = None
    filter: dict[str, Any] | None = None
