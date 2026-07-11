from __future__ import annotations

from dataclasses import field, make_dataclass
from enum import Flag, auto
from typing import Any, Literal

import plotly.graph_objects as go

from .configurator import Configurator


class AxisType(Flag):
    """Bit flags describing which axis modes an axis supports."""

    NUMERIC = auto()
    CATEGORICAL = auto()


AxisMode = Literal["numeric", "categorical"]


def SupportsAxis(axis_name: str, axis_type: AxisType) -> type[Configurator]:  # noqa: N802  factory returns a class
    """Factory that creates axis configurator mixins."""
    has_numeric = AxisType.NUMERIC in axis_type
    has_categorical = AxisType.CATEGORICAL in axis_type
    has_both = has_numeric and has_categorical

    fields = _common_fields(axis_name)

    if has_numeric:
        fields += _numeric_fields(axis_name)
    if has_categorical:
        fields += _categorical_fields(axis_name)
    if has_both:
        fields += _mode_fields(axis_name)

    def configure(self: Any, fig: go.Figure) -> None:
        kwargs: dict[str, Any] = {}

        _configure_common(self, axis_name, kwargs)

        if has_both:
            mode = getattr(self, f"{axis_name}_axis_mode")
            if mode is None:
                mode = getattr(self, f"_{axis_name}_resolved_mode", "categorical")
            if mode == "numeric":
                _configure_numeric(self, axis_name, kwargs)
            else:
                _configure_categorical(self, axis_name, kwargs)
        elif has_numeric:
            _configure_numeric(self, axis_name, kwargs)
        elif has_categorical:
            _configure_categorical(self, axis_name, kwargs)

        fig.update_layout(**{f"{axis_name}axis": kwargs})

    return make_dataclass(
        f"SupportsAxis_{axis_name}",
        fields=fields,
        bases=(Configurator,),
        namespace={
            "axis_name": axis_name,
            "axis_type": axis_type,
            "configure": configure,
        },
        kw_only=True,
    )


# --- Field definitions ---


def _common_fields(axis_name: str) -> list[tuple[str, type, Any]]:
    return [
        (f"{axis_name}_title", str | None, field(default=None)),
        (f"{axis_name}_showticklabels", bool, field(default=True)),
        (f"{axis_name}_tickangle", int | None, field(default=None)),
        (f"{axis_name}_tickfont_size", int | None, field(default=None)),
        (f"{axis_name}_tickfont_family", str | None, field(default=None)),
        (f"{axis_name}_tickfont_color", str | None, field(default=None)),
        (f"{axis_name}_showgrid", bool, field(default=True)),
        (f"{axis_name}_gridcolor", str | None, field(default=None)),
        (f"{axis_name}_showline", bool, field(default=True)),
        (f"{axis_name}_linecolor", str | None, field(default=None)),
    ]


def _numeric_fields(axis_name: str) -> list[tuple[str, type, Any]]:
    return [
        (f"{axis_name}_range", tuple[float, float] | None, field(default=None)),
        (f"{axis_name}_dtick", float | str | None, field(default=None)),
        (f"{axis_name}_tick0", float | None, field(default=None)),
        (f"{axis_name}_tickformat", str | None, field(default=None)),
        (f"{axis_name}_tickmode", str | None, field(default=None)),
        (f"{axis_name}_tickvals", list[float] | None, field(default=None)),
        (f"{axis_name}_ticktext", list[str] | None, field(default=None)),
        (f"{axis_name}_zeroline", bool, field(default=False)),
        (f"{axis_name}_zerolinecolor", str | None, field(default=None)),
    ]


def _categorical_fields(axis_name: str) -> list[tuple[str, type, Any]]:
    return [
        (f"{axis_name}_categoryorder", str | None, field(default=None)),
        (f"{axis_name}_categoryarray", list[str] | None, field(default=None)),
    ]


def _mode_fields(axis_name: str) -> list[tuple[str, type, Any]]:
    # None means "infer from the data"; the plot stores the outcome in
    # _{axis}_resolved_mode before configuration runs.
    return [
        (f"{axis_name}_axis_mode", AxisMode | None, field(default=None)),
    ]


# --- Configure helpers ---


def _configure_common(self: Any, axis_name: str, kwargs: dict[str, Any]) -> None:
    def _get(suffix: str) -> Any:
        return getattr(self, f"{axis_name}_{suffix}")

    # Get theme axes style for defaults
    theme = getattr(self, "_theme", None)
    axes_style = theme.axes if theme else None

    title = _get("title")
    if title is not None:
        kwargs["title"] = title

    kwargs["showticklabels"] = _get("showticklabels")

    # showgrid: explicit value or theme default
    show_grid = _get("showgrid")
    if show_grid is not None:
        kwargs["showgrid"] = show_grid
    elif axes_style:
        kwargs["showgrid"] = axes_style.show_grid

    # showline: explicit value or theme default
    show_line = _get("showline")
    if show_line is not None:
        kwargs["showline"] = show_line
    elif axes_style:
        kwargs["showline"] = axes_style.show_line

    # mirror: from theme if available
    if axes_style:
        kwargs["mirror"] = axes_style.mirror

    tickangle = _get("tickangle")
    if tickangle is not None:
        kwargs["tickangle"] = tickangle

    # gridcolor: explicit value or theme default
    gridcolor = _get("gridcolor")
    if gridcolor is not None:
        kwargs["gridcolor"] = gridcolor
    elif axes_style:
        kwargs["gridcolor"] = axes_style.grid_color

    # linecolor: explicit value or theme default
    linecolor = _get("linecolor")
    if linecolor is not None:
        kwargs["linecolor"] = linecolor
    elif axes_style:
        kwargs["linecolor"] = axes_style.line_color

    tickfont: dict[str, Any] = {}
    if (size := _get("tickfont_size")) is not None:
        tickfont["size"] = size
    if (family := _get("tickfont_family")) is not None:
        tickfont["family"] = family
    if (color := _get("tickfont_color")) is not None:
        tickfont["color"] = color
    if tickfont:
        kwargs["tickfont"] = tickfont


def _configure_numeric(self: Any, axis_name: str, kwargs: dict[str, Any]) -> None:
    def _get(suffix: str) -> Any:
        return getattr(self, f"{axis_name}_{suffix}")

    if (rng := _get("range")) is not None:
        kwargs["range"] = rng

    if (dtick := _get("dtick")) is not None:
        kwargs["dtick"] = dtick

    if (tick0 := _get("tick0")) is not None:
        kwargs["tick0"] = tick0

    if (tickmode := _get("tickmode")) is not None:
        kwargs["tickmode"] = tickmode

    if (tickvals := _get("tickvals")) is not None:
        kwargs["tickvals"] = tickvals

    if (ticktext := _get("ticktext")) is not None:
        kwargs["ticktext"] = ticktext

    if (fmt := _get("tickformat")) is not None:
        kwargs["tickformat"] = fmt

    kwargs["zeroline"] = _get("zeroline")
    if (zcolor := _get("zerolinecolor")) is not None:
        kwargs["zerolinecolor"] = zcolor


def _configure_categorical(self: Any, axis_name: str, kwargs: dict[str, Any]) -> None:
    def _get(suffix: str) -> Any:
        return getattr(self, f"{axis_name}_{suffix}")

    if (order := _get("categoryorder")) is not None:
        kwargs["categoryorder"] = order

    if (arr := _get("categoryarray")) is not None:
        kwargs["categoryarray"] = arr
