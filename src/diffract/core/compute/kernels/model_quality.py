import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from diffract.core.compute.decorator import kernel


@kernel
def pl_alpha_weighted(esd_max: float, pl_alpha: float) -> float:
    """Compute power law alpha weighted by log ESD max."""
    return pl_alpha * math.log(esd_max)


@kernel
def rand_distance(
    esd: NDArray[np.floating[Any]], esd_rand: NDArray[np.floating[Any]]
) -> float:
    """Compute Jensen-Shannon divergence between ESD and randomized ESD."""
    avg = 0.5 * (esd + esd_rand)
    divergence = 0.5 * (
        cast("float", stats.entropy(esd, avg))
        + cast("float", stats.entropy(esd_rand, avg))
    )
    return math.sqrt(divergence)
