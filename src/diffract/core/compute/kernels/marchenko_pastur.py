from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.extensions.rmt import marchenko_pastur_cdf

EPS = 1e-16


@kernel(produce_fields=("mp_esd_max", "mp_esd_min", "mp_bulk_std"))
def marchenko_pastur_fit(
    esd_rand: NDArray[np.floating[Any]],
    esd_max: float,
    weights_std: NDArray[np.floating[Any]],
    aspect_ratio: float,
    lower_dim: int,
) -> tuple[float, float, float]:
    r"""Fit the Marchenko-Pastur bulk edges to the randomized ESD.

    :math:`\lambda_\pm = \sigma_{\mathrm{b}}^2\,(1 \pm 1/\sqrt{Q})^2`
    """
    sigma_2_fit = weights_std**2
    eig_max_fit = sigma_2_fit * (1 + (1 / np.sqrt(aspect_ratio))) ** 2

    eigenvals_bleeding_out = esd_rand[esd_rand > eig_max_fit]
    if eigenvals_bleeding_out.size > 0:
        bulk = esd_rand[esd_rand <= eig_max_fit]
        sigma_2_corrected = np.sum(bulk) / lower_dim
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
    r"""Singular value at the MP bulk edge.

    :math:`\sigma_+^{\mathrm{MP}} = \sqrt{\lambda_+ N}`
    """
    return np.sqrt(mp_esd_max * greater_dim)


@kernel
def mp_ks(
    aspect_ratio: float,
    mp_bulk_std: float,
    esd: NDArray[np.floating[Any]],
    mp_esd_max: float,
    mp_esd_min: float,
) -> float:
    r"""Two-sided Kolmogorov-Smirnov distance for the MP fit.

    :math:`D = \sup_\lambda \lvert \hat{F}(\lambda) - F_{\mathrm{MP}}(\lambda)\rvert`
    """
    ratio = 1 / (aspect_ratio + 1e-8)
    sigma = mp_bulk_std  # we consider the fitted bulk std as the standard deviation

    esd_mask = (mp_esd_min < esd) & (esd < mp_esd_max)
    n = np.sum(esd_mask).astype(int).item()
    if n == 0:
        return 1.0

    # Condition the model CDF on the same (mp_esd_min, mp_esd_max) window the
    # sample is truncated to; the sentinel guards a window with no model mass.
    f_min = float(marchenko_pastur_cdf(mp_esd_min, ratio, sigma))
    f_max = float(marchenko_pastur_cdf(mp_esd_max, ratio, sigma))
    window = f_max - f_min
    if window <= EPS:
        return 1.0

    esd_filtered = esd[esd_mask]
    model_cdf = (marchenko_pastur_cdf(esd_filtered, ratio, sigma) - f_min) / window

    ranks = np.arange(1, n + 1)
    d_plus = np.max(ranks / n - model_cdf)
    d_minus = np.max(model_cdf - (ranks - 1) / n)

    return float(max(d_plus, d_minus))


@kernel
def mp_concentration(
    esd: NDArray[np.floating[Any]], mp_esd_max: float, mp_esd_min: float
) -> float:
    r"""MP bulk concentration (bulk fraction of the spectrum).

    :math:`\#\{i : \lambda_- \le \lambda_i \le \lambda_+\} / M`
    """
    if np.isnan(mp_esd_max) or np.isnan(mp_esd_min) or np.isnan(esd).any():
        return float("nan")

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
    r"""MP bulk presence (bulk width as a fraction of the spectrum width).

    :math:`(\lambda_+ - \lambda_-) / (\lambda_{\max} - \lambda_{\min})`
    """
    esd_width = esd_max - esd_min
    mp_width = mp_esd_max - mp_esd_min

    return float(np.clip(mp_width / (esd_width + EPS), 0.0, 1.0))


@kernel
def mp_num_spikes(esd: NDArray[np.floating[Any]], mp_esd_max: float) -> float:
    r"""Spikes above the MP bulk edge :math:`\#\{i : \lambda_i > \lambda_+\}`."""
    if np.isnan(mp_esd_max) or np.isnan(esd).any():
        return float("nan")

    return float(np.sum((esd > mp_esd_max).astype(int)).item())
