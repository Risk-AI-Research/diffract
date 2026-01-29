"""Helpers for capturing lightweight value metadata (dtype, shape, kind)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        return asdict(self)

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


def infer_value_metadata(value: Any) -> ValueMetadata:
    """Infer basic metadata for a value that will be stored.

    Keeps the schema minimal while still being enough for visualization/signature
    purposes.
    """
    if isinstance(value, np.ndarray):
        ndim = int(value.ndim)
        return ValueMetadata(
            kind=_kind_from_ndim(ndim),
            dtype=str(value.dtype),
            shape=tuple(int(x) for x in value.shape),
            ndim=ndim,
            is_numeric=bool(np.issubdtype(value.dtype, np.number)),
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
