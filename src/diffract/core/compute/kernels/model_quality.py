import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from diffract.core.compute.decorator import kernel


@kernel
def pl_alpha_weighted(esd_max: float, pl_alpha: float) -> float:
    r"""Alpha weighted by the log10 spectrum maximum.

    :math:`\alpha_{\mathrm{PL}}\,\log_{10}\lambda_{\max}`
    """
    return pl_alpha * math.log10(esd_max)


@kernel
def w1_rand_distance(
    esd: NDArray[np.floating[Any]], esd_rand: NDArray[np.floating[Any]]
) -> float:
    r"""Scale-invariant Wasserstein-1 distance from the ESD to its randomized null.

    :math:`\mathcal{W}_1(\lambda, \lambda^{\mathrm{rand}}) / \langle\lambda\rangle` --
    the earth-mover distance between the spectrum and its permutation null,
    divided by the shared mean eigenvalue so the index is dimensionless and
    comparable across layers of different scale.
    """
    mean = float(esd.mean())
    if np.isnan(mean):
        return float("nan")
    if mean <= 0:
        return 0.0
    return float(stats.wasserstein_distance(esd, esd_rand) / mean)
