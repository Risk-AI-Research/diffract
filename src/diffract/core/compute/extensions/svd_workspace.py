"""Estimate the transient host memory used by the CPU SVD."""

from __future__ import annotations

import logging
from functools import cache
from math import prod
from typing import TYPE_CHECKING

from diffract.core.compute.execution.host_memory import available_host_bytes

if TYPE_CHECKING:
    import numpy as np

_MIB = 1024**2
_MATRIX_NDIM = 2
_HOST_HEADROOM_FIFTHS = 4
logger = logging.getLogger(__name__)


def estimate_svd_workspace_bytes(matrix: np.ndarray, *, vectors: bool) -> int:
    """Budget scratch space, input copies, and sorted economy factors for GESDD.

    The estimate reserves eight bytes per real element. For shorter dimension k,
    64k + max(m, n) covers linear work and 3mn covers matrix-sized temporaries.
    Vectors add mn + 10k² for factor copies, sorting and quadratic workspace.
    These coefficients are conservative allowances, not LAPACK workspace minima.
    """
    if matrix.ndim < _MATRIX_NDIM or not matrix.size:
        return 0
    rows, cols = matrix.shape[-_MATRIX_NDIM:]
    short = min(rows, cols)
    matrices = prod(matrix.shape[:-_MATRIX_NDIM])
    scalar = 16 if matrix.dtype.kind == "c" else 8
    linear_workspace = 64 * short + max(rows, cols)
    workspace = 3 * rows * cols + linear_workspace
    if vectors:
        workspace += rows * cols + 10 * short**2
    return scalar * matrices * workspace


@cache
def _warn_unavailable() -> None:
    logger.warning(
        "Available host RAM could not be read. SVD inputs are streamed "
        "one at a time, but the workspace estimate cannot be checked."
    )


def require_svd_workspace(matrix: np.ndarray, *, vectors: bool) -> None:
    """Raise before calling LAPACK when its estimated workspace cannot fit."""
    required = estimate_svd_workspace_bytes(matrix, vectors=vectors)
    if not required:
        return
    available = available_host_bytes()
    if available is None:
        _warn_unavailable()
        return
    budget = available * _HOST_HEADROOM_FIFTHS // 5
    if required > budget:
        raise MemoryError(
            f"CPU SVD for {matrix.shape} ({matrix.dtype}) needs approximately "
            f"{required / _MIB:.1f} MiB of working memory. "
            f"The current budget is {budget / _MIB:.1f} MiB "
            f"of {available / _MIB:.1f} MiB available host RAM. "
            "Free RAM, reduce the matrix size, or run on a larger host."
        )
