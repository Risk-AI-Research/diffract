"""Helpers for capturing lightweight value metadata (dtype, shape, kind)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

Kind = Literal["scalar", "vector", "matrix", "ndarray", "object"]


_SCALAR_NDIM = 0
_VECTOR_NDIM = 1
_MATRIX_NDIM = 2


@dataclass(slots=True)
class ValueMetadata:
    """Normalized description of a stored value."""

    kind: Kind
    dtype: str | None
    shape: tuple[int, ...] | None
    ndim: int | None
    is_numeric: bool

    def to_jsonable(self) -> dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary."""
        return {
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": self.shape,
            "ndim": self.ndim,
            "is_numeric": self.is_numeric,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> ValueMetadata:
        """Reconstruct metadata from JSON-serializable dictionary."""
        return cls(
            kind=data.get("kind", "object"),
            dtype=data.get("dtype"),
            shape=tuple(data["shape"]) if data.get("shape") is not None else None,
            ndim=data.get("ndim"),
            is_numeric=bool(data.get("is_numeric", False)),
        )


def _kind_from_ndim(ndim: int) -> Kind:
    if ndim == _SCALAR_NDIM:
        return "scalar"
    if ndim == _VECTOR_NDIM:
        return "vector"
    if ndim == _MATRIX_NDIM:
        return "matrix"
    return "ndarray"


@lru_cache(maxsize=128)
def _dtype_metadata(dtype: np.dtype) -> tuple[str, bool]:
    return str(dtype), bool(np.issubdtype(dtype, np.number))


def infer_value_metadata(value: Any) -> ValueMetadata:
    """Infer basic metadata for a value that will be stored.

    Keeps the schema minimal while still being enough for visualization/signature
    purposes.
    """
    if isinstance(value, np.ndarray):
        ndim = value.ndim
        dtype, is_numeric = _dtype_metadata(value.dtype)
        return ValueMetadata(
            kind=_kind_from_ndim(ndim),
            dtype=dtype,
            shape=value.shape,
            ndim=ndim,
            is_numeric=is_numeric,
        )

    if isinstance(value, (int, float, bool, complex, np.generic)):
        # Scalars treated as 0-D.
        dtype_name = type(value).__name__
        is_numeric = isinstance(value, (int, float, complex, np.generic, bool))
        return ValueMetadata(
            kind="scalar",
            dtype=dtype_name,
            shape=None,
            ndim=0,
            is_numeric=is_numeric,
        )

    # Fallback: treat as opaque object.
    return ValueMetadata(
        kind="object",
        dtype=type(value).__name__,
        shape=None,
        ndim=None,
        is_numeric=False,
    )
