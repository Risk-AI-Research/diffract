import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import entr

from diffract.core.compute.decorator import kernel

EPS = 1e-16


def _relative_spectrum(weights_svals: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    values = np.asarray(weights_svals, dtype=np.float64)
    peak = float(np.max(np.abs(values), initial=0.0))
    with np.errstate(invalid="ignore", under="ignore"):
        return values / peak if peak else values


@kernel(version="2")
def effective_rank(weights_svals: NDArray[np.floating[Any]]) -> float:
    r"""Entropy-based effective rank, equal to one for a zero spectrum.

    :math:`\exp\!\big(-\sum_i p_i \ln p_i\big),\ p_i = \sigma_i / \sum_j \sigma_j`
    """
    relative = _relative_spectrum(weights_svals)
    total = float(relative.sum())
    if total == 0.0:
        return 1.0
    with np.errstate(under="ignore"):
        return math.exp(float(entr(relative / total).sum()))


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


def stable_rank(frob_norm: float, l2_norm: float) -> float:
    r"""Stable rank :math:`\lVert W\rVert_F^2 / \lVert W\rVert_2^2`."""
    if l2_norm == 0:
        return 0.0 if frob_norm == 0 else math.nan
    return (frob_norm / l2_norm) ** 2


@kernel(name="stable_rank", version="2")
def _spectral_stable_rank(weights_svals: NDArray[np.floating[Any]]) -> float:
    r"""Stable rank for a nonzero spectrum, or zero for an all-zero spectrum.

    :math:`\sum_i (\sigma_i / \sigma_{\max})^2`
    """
    relative = _relative_spectrum(weights_svals)
    return float(relative @ relative)
