import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import diffract.core.utils.imports as import_utils

_skrmt_spectral_law = import_utils.get_module("skrmt.ensemble.spectral_law")
TracyWidomDistribution = (
    None
    if _skrmt_spectral_law is None
    else getattr(_skrmt_spectral_law, "TracyWidomDistribution", None)
)
_SKRMT_AVAILABLE = TracyWidomDistribution is not None

from diffract.core.compute.decorator import kernel


@kernel
def tw_esd_bound(
    greater_dim: int,
    lower_dim: int,
    mp_bulk_std: float,
    *,
    p_value_threshold: float = 0.005,
) -> float:
    """Compute Tracy-Widom bound for ESD spike detection."""
    if not _SKRMT_AVAILABLE:
        raise ModuleNotFoundError(
            "Missing optional dependency 'skrmt'. Install scikit-rmt (skrmt) to use "
            "tw_esd_bound."
        )

    g_dim, l_dim = greater_dim, lower_dim

    loc = math.sqrt(g_dim - 1) + math.sqrt(l_dim)
    inv_loc = 1 / math.sqrt(g_dim - 1) + 1 / math.sqrt(l_dim)
    scale = (mp_bulk_std**2) / g_dim

    mu: float = scale * (loc**2)
    sigma: float = scale * (loc) * (inv_loc ** (1 / 3))
    pure_twd_bound = cast(
        "float",
        TracyWidomDistribution().ppf(1 - p_value_threshold),  # type: ignore[misc]
    )

    return pure_twd_bound * sigma + mu


@kernel
def tw_num_spikes(esd: NDArray[np.floating[Any]], tw_esd_bound: float) -> int:
    """Count number of spikes above Tracy-Widom bound."""
    return cast("int", np.sum((esd > tw_esd_bound).astype(int)).item())
