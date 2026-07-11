from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from diffract.viz.data import DataShape, DataType, Entry, FieldRef
from diffract.viz.data.extraction import get_field_data


@dataclass
class NumericPropertyResolver:
    """Resolves numeric FieldRef values with optional normalization."""

    output_range: tuple[float, float] | None = None

    def resolve(
        self,
        source: FieldRef | float | None,
        entries: dict[str, Entry] | None = None,
    ) -> float | list[float] | None:
        """Resolve a numeric source into a value or list of values.

        Args:
            source: A literal number, a FieldRef, or None.
            entries: Entries used to resolve a FieldRef, if given.

        Returns:
            The literal number, the resolved per-entry values, or None.
        """
        match source:
            case None:
                return None
            case int() | float():
                return source
            case FieldRef():
                return self._resolve_field_ref(source, entries)

    def _resolve_field_ref(
        self,
        ref: FieldRef,
        entries: dict[str, Entry] | None,
    ) -> list[float] | None:
        if entries is None:
            return None

        values, data_type, data_shape = get_field_data(entries, ref.field)
        if ref.data_type is not None:
            data_type = ref.data_type

        if data_type != DataType.NUMERIC:
            raise ValueError(
                f"Field '{ref.field}' must be numeric for this property, "
                f"got {data_type.name}"
            )

        if data_shape == DataShape.SCALAR:
            float_values = [float(v) if v is not None else float("nan") for v in values]
        else:
            float_values = [
                v if v is not None else [float("nan")] * len(v) for v in values
            ]

        return self._normalize(float_values)

    def _normalize(
        self,
        values: list[float | list[float]],
        data_min: float | None = None,
        data_max: float | None = None,
    ) -> list[float | list[float]]:
        if self.output_range is None:
            return values

        if isinstance(values[0], Iterable):
            for v in values:
                valid = [v_item for v_item in v if not math.isnan(v_item)]
                partial_min, partial_max = min(valid), max(valid)
                if data_min is None or partial_min < data_min:
                    data_min = partial_min
                if data_max is None or partial_max > data_max:
                    data_max = partial_max
            return [self._normalize(v, data_min, data_max) for v in values]
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            return values

        if data_min is None:
            data_min = min(valid)

        if data_max is None:
            data_max = max(valid)

        if data_min == data_max:
            mid = (self.output_range[0] + self.output_range[1]) / 2
            return [mid if not math.isnan(v) else float("nan") for v in values]

        out_min, out_max = self.output_range
        scale = (out_max - out_min) / (data_max - data_min)

        return np.array(
            [
                out_min + (v - data_min) * scale if not math.isnan(v) else float("nan")
                for v in values
            ]
        )
