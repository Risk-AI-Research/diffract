from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.extensions.rmt import marchenko_pastur_cdf


@kernel(produce_fields=("mp_esd_max", "mp_esd_min", "mp_bulk_std"))
def marchenko_pastur_fit(
    esd_rand: NDArray[np.floating[Any]],
    esd_max: float,
    weights_std: NDArray[np.floating[Any]],
    aspect_ratio: float,
    lower_dim: int,
) -> tuple[float, float, float]:
    """Fit Marchenko-Pastur distribution to random ESD."""
    sigma_2_fit = weights_std**2
    eig_max_fit = sigma_2_fit * (1 + (1 / np.sqrt(aspect_ratio))) ** 2

    eigenvals_bleeding_out = esd_rand[esd_rand > eig_max_fit]
    if eigenvals_bleeding_out.size > 0:
        sigma_2_corrected = sigma_2_fit - (np.sum(eigenvals_bleeding_out) / lower_dim)
        eig_max_corrected = sigma_2_corrected * (1 + (1 / np.sqrt(aspect_ratio))) ** 2
        mp_esd_max = eig_max_corrected
        mp_bulk_std = np.sqrt(sigma_2_corrected)
    else:
        mp_esd_max = eig_max_fit
        mp_bulk_std = weights_std

    mp_esd_max = min(mp_esd_max, esd_max)

    mp_esd_min = (mp_bulk_std**2) * (1 - (1 / np.sqrt(aspect_ratio))) ** 2

    return mp_esd_max, mp_esd_min, mp_bulk_std


@kernel
def mp_sval_max(mp_esd_max: float, greater_dim: int) -> float:
    """Compute maximum singular value from MP ESD max."""
    return np.sqrt(mp_esd_max * greater_dim)


@kernel
def mp_ks(
    aspect_ratio: float,
    mp_bulk_std: float,
    esd: NDArray[np.floating[Any]],
    mp_esd_max: float,
    mp_esd_min: float,
) -> float:
    """Compute Kolmogorov-Smirnov distance for Marchenko-Pastur fit."""
    ratio = 1 / (aspect_ratio + 1e-8)
    sigma = mp_bulk_std  # we consider the fitted bulk std as the standard deviation

    esd_mask = (mp_esd_min < esd) & (esd < mp_esd_max)
    if np.sum(esd_mask).astype(int).item() == 0:
        ks_distance = 1
    else:
        esd_filtered = esd[esd_mask]
        model_cdf = marchenko_pastur_cdf(esd_filtered, ratio, sigma)

        empirical_cdf = np.arange(esd_filtered.size) / esd_filtered.size

        ks_distance = np.max(np.abs(empirical_cdf - model_cdf))

    return ks_distance


@kernel
def mp_concentration(
    esd: NDArray[np.floating[Any]], mp_esd_max: float, mp_esd_min: float
) -> float:
    """Compute MP bulk concentration (bulk size / total size)."""
    mask = (mp_esd_min <= esd) & (esd <= mp_esd_max)
    bulk_size = cast("int", np.sum(mask.astype(int)).item())

    return bulk_size / esd.size


@kernel
def mp_presence(
    esd_min: float,
    esd_max: float,
    mp_esd_max: float,
    mp_esd_min: float,
) -> float:
    """Compute MP bulk presence (bulk width / total width)."""
    esd_width = esd_max - esd_min
    mp_width = mp_esd_max - mp_esd_min

    return mp_width / esd_width


@kernel
def mp_num_spikes(esd: NDArray[np.floating[Any]], mp_esd_max: float) -> int:
    """Count number of spikes above MP bulk maximum."""
    return cast("int", np.sum((esd > mp_esd_max).astype(int)).item())
