"""Random matrix theory distributions for the spectral kernels.

Implements the closed-form cumulative distribution function of the
Marchenko-Pastur law and the quantile function of the Tracy-Widom law for
beta = 1 (GOE). The Marchenko-Pastur CDF integrates the density through the
standard antiderivative of ``sqrt((x - a)(b - x)) / x``. The Tracy-Widom
distribution is computed from its Painleve II representation: the
Hastings-McLeod solution ``q'' = s q + 2 q^3`` with ``q(s) ~ Ai(s)`` as
``s -> +inf`` is integrated numerically together with the auxiliary
integrals that give ``F2(s) = exp(-int (x - s) q^2 dx)`` and
``F1(s)^2 = F2(s) exp(-int q dx)``; quantiles invert the resulting CDF with
a monotone PCHIP spline built once per process.

References:
    - Marchenko, V. A. and Pastur, L. A. "Distribution of eigenvalues for
      some sets of random matrices". Mathematics of the USSR-Sbornik (1967).
    - Tracy, C. A. and Widom, H. "Level-spacing distributions and the Airy
      kernel". Communications in Mathematical Physics (1994).
    - Tracy, C. A. and Widom, H. "On orthogonal and symplectic matrix
      ensembles". Communications in Mathematical Physics (1996).
    - Hastings, S. P. and McLeod, J. B. "A boundary value problem associated
      with the second Painleve transcendent and the Korteweg-de Vries
      equation". Archive for Rational Mechanics and Analysis (1980).
    - Bejan, A. "Largest eigenvalues and sample covariance matrices".
      M.Sc. dissertation, The University of Warwick (2005).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import airy

_PAINLEVE_S_START = 10.0
_PAINLEVE_S_END = -5.0
_PAINLEVE_RTOL = 1e-12
_PAINLEVE_ATOL = 1e-18
_TW_GRID_STEP = 0.005


def marchenko_pastur_cdf(
    x: NDArray[np.floating[Any]] | float,
    ratio: float,
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate the Marchenko-Pastur CDF for beta = 1.

    Args:
        x: Value or array of values to evaluate the CDF at.
        ratio: Dimension ratio lambda in (0, 1] of the underlying matrix.
        sigma: Standard deviation of the matrix entries.

    Returns:
        CDF values as a float64 array of the same shape as ``x``.

    Raises:
        ValueError: If ``ratio`` is outside (0, 1] or ``sigma`` is not positive.
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    sqrt_ratio = np.sqrt(ratio)
    edge_low = (1 - sqrt_ratio) ** 2
    edge_high = (1 + sqrt_ratio) ** 2

    z = np.asarray(x, dtype=np.float64) / sigma**2
    result = np.zeros_like(z)
    result[z >= edge_high] = 1.0

    inside = (z > edge_low) & (z < edge_high)
    if np.any(inside):
        zi = z[inside]
        root = np.sqrt((zi - edge_low) * (edge_high - zi))
        span = edge_high - edge_low
        first_arcsin = np.arcsin((2 * zi - edge_low - edge_high) / span)
        product = edge_low * edge_high
        second_arcsin = np.arcsin(
            ((edge_low + edge_high) * zi - 2 * product) / (zi * span)
        )
        result[inside] = (
            root
            + (1 + ratio) * first_arcsin
            - (1 - ratio) * second_arcsin
            + np.pi * ratio
        ) / (2 * np.pi * ratio)

    return np.clip(result, 0.0, 1.0)


@cache
def _tw_beta1_cdf_spline() -> PchipInterpolator:
    s_start = _PAINLEVE_S_START
    ai_start, ai_prime_start, *_ = airy(s_start)
    airy_tail_sq = quad(lambda t: airy(t)[0] ** 2, s_start, np.inf)[0]
    airy_tail_weighted = quad(
        lambda t: (t - s_start) * airy(t)[0] ** 2, s_start, np.inf
    )[0]
    airy_tail = quad(lambda t: airy(t)[0], s_start, np.inf)[0]

    def rhs(s: float, y: NDArray[np.float64]) -> list[float]:
        q, q_prime, _, k_val, _ = y
        return [q_prime, s * q + 2 * q**3, -k_val, -(q**2), -q]

    solution = solve_ivp(
        rhs,
        [s_start, _PAINLEVE_S_END],
        [ai_start, ai_prime_start, airy_tail_weighted, airy_tail_sq, airy_tail],
        method="DOP853",
        rtol=_PAINLEVE_RTOL,
        atol=_PAINLEVE_ATOL,
        dense_output=True,
    )
    if solution.status != 0:
        raise RuntimeError(f"Painleve II integration failed: {solution.message}")

    s_grid = np.arange(_PAINLEVE_S_END, s_start, _TW_GRID_STEP)
    states = solution.sol(s_grid)
    cdf = np.exp(-(states[2] + states[4]) / 2)

    increasing = np.concatenate(([True], np.diff(cdf) > 0))
    return PchipInterpolator(cdf[increasing], s_grid[increasing], extrapolate=False)


def tracy_widom_ppf(q: float) -> float:
    """Evaluate the Tracy-Widom (beta = 1) quantile function.

    Args:
        q: Lower-tail probability inside the computed CDF range.

    Returns:
        The quantile corresponding to ``q``.

    Raises:
        ValueError: If ``q`` falls outside the computed CDF range.
    """
    spline = _tw_beta1_cdf_spline()
    q_low, q_high = spline.x[0], spline.x[-1]
    if not q_low <= q <= q_high:
        raise ValueError(
            f"q must be within the computed CDF range "
            f"[{q_low:.2e}, {q_high:.10f}], got {q}"
        )
    return float(spline(q))
