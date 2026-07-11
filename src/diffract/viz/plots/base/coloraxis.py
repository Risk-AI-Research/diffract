from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any

import plotly.graph_objects as go

from .configurator import Configurator


def SupportsColoraxis(prefix: str) -> type[Configurator]:  # noqa: N802  factory returns a class
    """Factory that creates coloraxis configurator mixins."""
    fields = _coloraxis_fields(prefix)

    def configure(self: Any, fig: go.Figure) -> None:
        _configure_coloraxis(self, fig, prefix)

    def resolve_coloraxis(self: Any, fig: go.Figure) -> str:
        return _resolve_coloraxis(self, fig, prefix)

    return make_dataclass(
        f"SupportsColoraxis_{prefix}",
        fields=fields,
        bases=(Configurator,),
        namespace={
            "prefix": prefix,
            "configure": configure,
            "resolve_coloraxis": resolve_coloraxis,
        },
        kw_only=True,
    )


# --- Field definitions ---


def _coloraxis_fields(prefix: str) -> list[tuple[str, type, Any]]:
    return [
        (f"{prefix}_coloraxis_id", int | None, field(default=None)),
        (f"{prefix}_colorscale", str, field(default="Viridis")),
        (f"{prefix}_showscale", bool, field(default=True)),
        (f"{prefix}_cmin", float | None, field(default=None)),
        (f"{prefix}_cmax", float | None, field(default=None)),
        (f"{prefix}_colorbar_title", str | None, field(default=None)),
        (f"{prefix}_coloraxis_override", bool, field(default=False)),
    ]


# --- Configure helper ---


def _configure_coloraxis(self: Any, fig: go.Figure, prefix: str) -> None:
    def _get(suffix: str) -> Any:
        return getattr(self, f"{prefix}_{suffix}")

    coloraxis_name = _resolve_coloraxis(self, fig, prefix)

    if _coloraxis_exists(fig, coloraxis_name) and not _get("coloraxis_override"):
        return

    config: dict[str, Any] = {
        "colorscale": _get("colorscale"),
        "showscale": _get("showscale"),
    }

    cmin = _get("cmin")
    if cmin is not None:
        config["cmin"] = cmin

    cmax = _get("cmax")
    if cmax is not None:
        config["cmax"] = cmax

    colorbar_title = _get("colorbar_title")
    if colorbar_title is not None:
        config["colorbar"] = {"title": colorbar_title}

    fig.update_layout(**{coloraxis_name: config})


def _resolve_coloraxis(self: Any, fig: go.Figure, prefix: str) -> str:
    """Resolve coloraxis id and return the coloraxis name (e.g. 'coloraxis2')."""
    cache_attr = f"__{prefix}_resolved_coloraxis"

    cached = getattr(self, cache_attr, None)
    if cached is not None:
        return cached

    coloraxis_id = getattr(self, f"{prefix}_coloraxis_id")

    if coloraxis_id is not None:
        result = _coloraxis_name(coloraxis_id)
    else:
        result = _next_coloraxis_name(fig)

    object.__setattr__(self, cache_attr, result)
    return result


# --- Helpers ---


def _coloraxis_name(coloraxis_id: int) -> str:
    """Convert coloraxis id to Plotly coloraxis name."""
    if coloraxis_id == 1:
        return "coloraxis"
    return f"coloraxis{coloraxis_id}"


def _coloraxis_exists(fig: go.Figure, coloraxis_name: str) -> bool:
    """Check if coloraxis already exists in figure layout."""
    return getattr(fig.layout, coloraxis_name, None) is not None


def _next_coloraxis_name(fig: go.Figure) -> str:
    """Find next available coloraxis name in figure."""
    layout = fig.layout
    existing_ids: list[int] = []

    for key in dir(layout):
        if key == "coloraxis":
            if getattr(layout, key, None) is not None:
                existing_ids.append(1)
        elif (
            key.startswith("coloraxis")
            and key[9:].isdigit()
            and getattr(layout, key, None) is not None
        ):
            existing_ids.append(int(key[9:]))

    next_id = max(existing_ids, default=0) + 1

    return _coloraxis_name(next_id)
