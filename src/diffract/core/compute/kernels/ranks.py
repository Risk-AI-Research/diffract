import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel

EPS = 1e-16


@kernel
def effective_rank(weights_svals: NDArray[np.floating[Any]]) -> float:
    """Compute effective rank from singular values using entropy."""
    probabilities = weights_svals / cast("float", weights_svals.sum().item() + EPS)
    entropy = cast("float", -np.sum(probabilities * np.log(probabilities + EPS)).item())
    return math.exp(entropy)


@kernel
def hard_rank(esd: NDArray[np.floating[Any]], *, threshold: float = 1e-5) -> int:
    """Count eigenvalues above threshold."""
    return cast("int", np.sum((esd > threshold).astype(int)).item())


@kernel
def mp_soft_rank(mp_esd_max: float, esd_max: float) -> float:
    """Compute soft rank from MP bulk maximum relative to ESD maximum."""
    return mp_esd_max / (esd_max + EPS)


@kernel
def stable_rank(frob_norm: float, l2_norm: float) -> float:
    """Compute stable rank (Frobenius norm / spectral norm)^2."""
    return (frob_norm / (l2_norm + EPS)) ** 2
