"""Coloraxis remapping and aggregation for subplot composition."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from diffract.viz.plots.base.plot import Plot

if TYPE_CHECKING:
    import plotly.graph_objects as go

_COLORAXIS_RE = re.compile(r"^coloraxis(?P<id>[1-9]\d*)?$")
_MIN_COLORBAR_BOTTOM_MARGIN = 110


class ColoraxisRegistry:
    """Registry for remapping child coloraxis references to parent coloraxis."""

    __slots__ = ("_configs", "_counter", "_share_key_to_axis")

    def __init__(self) -> None:
        self._counter = 0
        self._share_key_to_axis: dict[str, str] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    def get_or_create(self, *, share_key: str | None = None) -> str:
        """Get a target coloraxis name, optionally shared across subplots."""
        if share_key is not None and share_key in self._share_key_to_axis:
            return self._share_key_to_axis[share_key]

        self._counter += 1
        axis_name = _coloraxis_name(self._counter)
        if share_key is not None:
            self._share_key_to_axis[share_key] = axis_name
        return axis_name

    def store_config(self, axis_name: str, config: dict[str, Any]) -> None:
        """Store/merge coloraxis layout config for final parent layout."""
        normalized = _normalize_coloraxis_ref(axis_name)
        if normalized is None:
            return

        existing = self._configs.get(normalized)
        if existing is None:
            self._configs[normalized] = dict(config)
            return

        self._configs[normalized] = _merge_shared_coloraxis_config(
            existing,
            config,
            axis_name=normalized,
        )

    def get_all_configs(self) -> dict[str, dict[str, Any]]:
        """Get all merged configs sorted in Plotly coloraxis order."""
        result: dict[str, dict[str, Any]] = {}
        for axis_name in sorted(self._configs, key=_coloraxis_sort_key):
            result[axis_name] = dict(self._configs[axis_name])
        return result


def extract_coloraxis_share_keys(plot: Plot) -> dict[str, str]:
    """Collect explicit coloraxis share keys from `*_coloraxis_id` fields."""
    share_keys: dict[str, str] = {}

    for attr_name in _iter_coloraxis_id_attrs(plot):
        coloraxis_id = getattr(plot, attr_name, None)
        if coloraxis_id is None:
            continue
        if not isinstance(coloraxis_id, int) or coloraxis_id < 1:
            raise ValueError(
                f"Invalid '{attr_name}={coloraxis_id}'. coloraxis_id must be >= 1."
            )

        axis_name = _coloraxis_name(coloraxis_id)
        share_keys[axis_name] = f"id:{coloraxis_id}"

    return share_keys


def extract_legacy_coloraxis_name(plot: Plot) -> str | None:
    """Read legacy `coloraxis_name` field for backward compatibility."""
    value = getattr(plot, "coloraxis_name", None)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def remap_coloraxis(
    fig: go.Figure,
    registry: ColoraxisRegistry,
    *,
    explicit_share_keys: dict[str, str],
    legacy_share_name: str | None,
) -> None:
    """Remap child coloraxis refs and store merged configs in registry."""
    layout_configs = _extract_layout_coloraxis_configs(fig)
    remap: dict[str, str] = {}

    for trace in fig.data:
        for location, old_axis_raw in _iter_trace_coloraxis_refs(trace):
            old_axis = _normalize_coloraxis_ref(old_axis_raw)
            if old_axis is None:
                continue

            target_axis = remap.get(old_axis)
            if target_axis is None:
                share_key = explicit_share_keys.get(old_axis)
                if share_key is None and legacy_share_name is not None:
                    share_key = f"name:{legacy_share_name}"
                target_axis = registry.get_or_create(share_key=share_key)
                remap[old_axis] = target_axis

            if location == "marker":
                trace.marker.coloraxis = target_axis
            elif location == "line":
                trace.line.coloraxis = target_axis
            else:
                trace.coloraxis = target_axis

    if not remap:
        return

    for old_axis, new_axis in remap.items():
        registry.store_config(new_axis, layout_configs.get(old_axis, {}))
        if hasattr(fig.layout, old_axis):
            fig.update_layout(**{old_axis: None})


def apply_coloraxis_configs(fig: go.Figure, registry: ColoraxisRegistry) -> None:
    """Apply merged coloraxis configs to the parent figure."""
    for axis_name, config in registry.get_all_configs().items():
        fig.update_layout(**{axis_name: config})


def distribute_colorbars(fig: go.Figure) -> None:
    """Spread multiple colorbars horizontally and ensure bottom margin."""
    coloraxis_names = _collect_visible_coloraxis_names(fig)
    count = len(coloraxis_names)
    if count <= 1:
        return

    # Leave visible gaps between the bars: several identically scaled
    # colorbars packed edge to edge read as one bar with garbled ticks.
    spacing = 0.9 / count
    colorbar_len = max(0.12, min(0.45, 0.72 / count))

    for idx, axis_name in enumerate(coloraxis_names):
        coloraxis = getattr(fig.layout, axis_name, None)
        if coloraxis is None:
            continue

        colorbar_obj = getattr(coloraxis, "colorbar", None)
        colorbar = colorbar_obj.to_plotly_json() if colorbar_obj is not None else {}

        colorbar.update(
            orientation="h",
            len=colorbar_len,
            x=0.05 + spacing * idx + (spacing / 2),
            xanchor="center",
            y=-0.15,
            yanchor="top",
        )

        fig.update_layout(**{axis_name: {"colorbar": colorbar}})

    _ensure_bottom_margin_for_colorbars(fig)


def _iter_coloraxis_id_attrs(plot: Plot) -> list[str]:
    if is_dataclass(plot):
        return [
            field.name for field in fields(plot) if field.name.endswith("_coloraxis_id")
        ]

    return [
        name
        for name in dir(plot)
        if name.endswith("_coloraxis_id") and not name.startswith("_")
    ]


def _iter_trace_coloraxis_refs(trace: go.BaseTraceType) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []

    marker = getattr(trace, "marker", None)
    marker_axis = getattr(marker, "coloraxis", None)
    if marker_axis:
        refs.append(("marker", marker_axis))

    line = getattr(trace, "line", None)
    line_axis = getattr(line, "coloraxis", None)
    if line_axis:
        refs.append(("line", line_axis))

    trace_axis = getattr(trace, "coloraxis", None)
    if trace_axis:
        refs.append(("trace", trace_axis))

    return refs


def _extract_layout_coloraxis_configs(fig: go.Figure) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}

    for key, value in fig.layout.to_plotly_json().items():
        axis_name = _normalize_coloraxis_ref(key)
        if axis_name is None or not isinstance(value, dict):
            continue

        existing = configs.get(axis_name)
        if existing is None:
            configs[axis_name] = dict(value)
        else:
            configs[axis_name] = _merge_dict(existing, value)

    return configs


def _normalize_coloraxis_ref(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _COLORAXIS_RE.fullmatch(value.strip())
    if match is None:
        return None

    suffix = match.group("id")
    if suffix is None or suffix == "1":
        return "coloraxis"
    return f"coloraxis{int(suffix)}"


def _coloraxis_name(axis_id: int) -> str:
    if axis_id < 1:
        raise ValueError(f"Coloraxis id must be >= 1, got {axis_id}.")
    if axis_id == 1:
        return "coloraxis"
    return f"coloraxis{axis_id}"


def _coloraxis_sort_key(axis_name: str) -> int:
    if axis_name == "coloraxis":
        return 1
    return int(axis_name.removeprefix("coloraxis"))


def _merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, extra_value in extra.items():
        if key not in merged:
            merged[key] = extra_value
            continue

        base_value = merged[key]
        if base_value is None:
            merged[key] = extra_value
            continue

        if isinstance(base_value, dict) and isinstance(extra_value, dict):
            merged[key] = _merge_dict(base_value, extra_value)

    return merged


def _merge_shared_coloraxis_config(
    base: dict[str, Any],
    extra: dict[str, Any],
    *,
    axis_name: str,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Merge shared coloraxis configs and fail on conflicting non-null values."""
    merged = dict(base)

    for key, extra_value in extra.items():
        if key not in merged:
            merged[key] = extra_value
            continue

        base_value = merged[key]
        if base_value is None:
            merged[key] = extra_value
            continue
        if extra_value is None:
            continue

        next_path = (*path, key)

        if isinstance(base_value, dict) and isinstance(extra_value, dict):
            merged[key] = _merge_shared_coloraxis_config(
                base_value,
                extra_value,
                axis_name=axis_name,
                path=next_path,
            )
            continue

        if base_value != extra_value:
            path_label = ".".join(next_path)
            raise ValueError(
                "Conflicting shared coloraxis config for "
                f"'{axis_name}' at '{path_label}': "
                f"{base_value!r} vs {extra_value!r}."
            )

    return merged


def _collect_visible_coloraxis_names(fig: go.Figure) -> list[str]:
    axis_names: list[str] = []

    for key in fig.layout.to_plotly_json():
        axis_name = _normalize_coloraxis_ref(key)
        if axis_name is None or axis_name in axis_names:
            continue

        coloraxis = getattr(fig.layout, axis_name, None)
        if coloraxis is None:
            continue

        if getattr(coloraxis, "showscale", None) is False:
            continue

        axis_names.append(axis_name)

    return sorted(axis_names, key=_coloraxis_sort_key)


def _ensure_bottom_margin_for_colorbars(fig: go.Figure) -> None:
    margin_obj = fig.layout.margin
    current_margin = margin_obj.to_plotly_json() if margin_obj is not None else {}
    if current_margin.get("b", 0) >= _MIN_COLORBAR_BOTTOM_MARGIN:
        return

    current_margin["b"] = _MIN_COLORBAR_BOTTOM_MARGIN
    fig.update_layout(margin=current_margin)
