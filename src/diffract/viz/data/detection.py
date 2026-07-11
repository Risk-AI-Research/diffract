from __future__ import annotations

import re
from typing import Any

import numpy as np

from .types import DataShape, DataType, FieldMeta

RE_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def is_number_regex(string: str) -> bool:
    """Return True if the string fully matches a numeric pattern."""
    return RE_NUMBER_PATTERN.fullmatch(string) is not None


def detect_field_meta(values: list[Any]) -> FieldMeta:
    """Detect DataType and DataShape from values."""
    data_type = detect_data_type(values)
    data_shape = detect_data_shape(values)
    return FieldMeta(data_type, data_shape)


def detect_data_type(values: list[Any]) -> DataType:
    """Detect if values are numeric or categorical."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return DataType.CATEGORICAL
        if isinstance(v, str):
            if is_number_regex(v):
                continue
            return DataType.CATEGORICAL
        if isinstance(v, (int, float, np.integer, np.floating)):
            continue
        if isinstance(v, np.ndarray):
            if not np.issubdtype(v.dtype, np.number):
                return DataType.CATEGORICAL
            continue
        return DataType.CATEGORICAL

    return DataType.NUMERIC


def detect_data_shape(values: list[Any]) -> DataShape:
    """Detect if field represents scalar or vector per entry."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, np.ndarray) and v.size > 1:
            return DataShape.VECTOR
        if isinstance(v, (list, tuple)) and len(v) > 1:
            return DataShape.VECTOR

    return DataShape.SCALAR
