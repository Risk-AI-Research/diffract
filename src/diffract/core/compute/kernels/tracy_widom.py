import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.extensions.rmt import tracy_widom_ppf


@kernel
def tw_esd_bound(
    greater_dim: int,
    lower_dim: int,
    mp_bulk_std: float,
    *,
    p_value_threshold: float = 0.005,
) -> float:
    """Compute Tracy-Widom bound for ESD spike detection."""
    g_dim, l_dim = greater_dim, lower_dim

    loc = math.sqrt(g_dim - 1) + math.sqrt(l_dim)
    inv_loc = 1 / math.sqrt(g_dim - 1) + 1 / math.sqrt(l_dim)
    scale = (mp_bulk_std**2) / g_dim

    mu: float = scale * (loc**2)
    sigma: float = scale * (loc) * (inv_loc ** (1 / 3))
    pure_twd_bound = tracy_widom_ppf(1 - p_value_threshold)

    return pure_twd_bound * sigma + mu


@kernel
def tw_num_spikes(esd: NDArray[np.floating[Any]], tw_esd_bound: float) -> int:
    """Count number of spikes above Tracy-Widom bound."""
    return cast("int", np.sum((esd > tw_esd_bound).astype(int)).item())
