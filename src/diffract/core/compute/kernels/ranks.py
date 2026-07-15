import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel

EPS = 1e-16


@kernel
def effective_rank(weights_svals: NDArray[np.floating[Any]]) -> float:
    r"""Entropy-based effective rank.

    :math:`\exp\!\big(-\sum_i p_i \ln p_i\big),\ p_i = \sigma_i / \sum_j \sigma_j`
    """
    probabilities = weights_svals / cast("float", weights_svals.sum().item() + EPS)
    entropy = cast("float", -np.sum(probabilities * np.log(probabilities + EPS)).item())
    return math.exp(entropy)


@kernel
def hard_rank(esd: NDArray[np.floating[Any]], *, rtol: float = 1e-5) -> int:
    r"""Count of eigenvalues above a relative threshold.

    :math:`\#\{i : \lambda_i > \texttt{rtol}\cdot\lambda_{\max}\}`
    """
    threshold = rtol * cast("float", esd.max())
    return cast("int", np.sum((esd > threshold).astype(int)).item())


@kernel
def mp_soft_rank(mp_esd_max: float, esd_max: float) -> float:
    r"""Soft rank :math:`\lambda_+ / \lambda_{\max}` (MP edge over spectrum max)."""
    return mp_esd_max / (esd_max + EPS)


@kernel
def stable_rank(frob_norm: float, l2_norm: float) -> float:
    r"""Stable rank :math:`\lVert W\rVert_F^2 / \lVert W\rVert_2^2`."""
    return (frob_norm / (l2_norm + EPS)) ** 2
