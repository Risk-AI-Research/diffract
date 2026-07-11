from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from diffract.viz.data import DataType, Entry, FieldRef
from diffract.viz.data.extraction import get_field_data
from diffract.viz.styling.palettes import ColorPalette, DefaultColorPalette
from diffract.viz.styling.sources import ColorSource

__all__ = ["ColorResolver", "ColorSource", "ResolvedColor"]


@dataclass
class ResolvedColor:
    """Resolved color result: concrete colors and/or numeric values."""

    color: str | list[str] | None = None
    values: list[float] | None = None


@dataclass
class ColorResolver:
    """Resolves color sources into concrete colors or coloraxis config."""

    palette: ColorPalette = field(default_factory=DefaultColorPalette)

    def resolve(
        self,
        source: ColorSource,
        entries: dict[str, Entry] | None = None,
    ) -> ResolvedColor:
        """Resolve a color source into concrete colors."""
        match source:
            case None:
                return ResolvedColor()
            case str():
                return ResolvedColor(color=source)
            case FieldRef():
                return self._resolve_field_ref(source, entries)

    def _resolve_field_ref(
        self,
        ref: FieldRef,
        entries: dict[str, Entry] | None,
    ) -> ResolvedColor:
        if entries is None:
            return ResolvedColor()

        values, data_type, _ = get_field_data(entries, ref.field)
        if ref.data_type is not None:
            data_type = ref.data_type

        match data_type:
            case DataType.CATEGORICAL:
                return self._resolve_categorical(values, ref)
            case DataType.NUMERIC:
                return self._resolve_numeric(values)

    def _resolve_categorical(self, values: list[Any], ref: FieldRef) -> ResolvedColor:
        unique_values = list(dict.fromkeys(values))
        order_indices = ref.ordering.argsort(unique_values)
        sorted_unique = [unique_values[i] for i in order_indices]
        colors = [self.palette.get_color(v, sorted_unique) for v in values]
        return ResolvedColor(color=colors)

    def _resolve_numeric(self, values: list[Any]) -> ResolvedColor:
        float_values: list[Any] = []
        for v in values:
            if v is None:
                float_values.append(float("nan"))
            else:
                try:
                    float_values.append(float(v))
                except (TypeError, ValueError):
                    float_values.append(v)
        return ResolvedColor(values=float_values)
