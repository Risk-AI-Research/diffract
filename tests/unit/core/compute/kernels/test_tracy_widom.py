"""Analytic and property tests for the Tracy-Widom spike kernels.

tw_esd_bound is the Johnstone largest-eigenvalue threshold: the decomposition
test recovers its (mu, sigma) from two evaluations and checks them against the
closed forms (validating the full dimension formula at one point), and a
homogeneity property pins its sigma**2 scaling across the input space.
tw_num_spikes is a strict count above the bound, pinned by analytic edge
cases. ``tracy_widom_ppf`` from ``extensions.rmt`` is a reference oracle, not
a kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import (
    given,
    strategies as st,
)

from diffract.core.compute.extensions.rmt import tracy_widom_ppf

pytestmark = pytest.mark.unit

positive_std = st.floats(
    min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False
)


@st.composite
def matrix_dims(draw: st.DrawFn) -> tuple[int, int]:
    """A valid (greater_dim >= 2, 1 <= lower_dim <= greater_dim) pair."""
    greater = draw(st.integers(min_value=2, max_value=4000))
    lower = draw(st.integers(min_value=1, max_value=greater))
    return greater, lower


def test_tw_esd_bound_matches_reference_decomposition() -> None:
    """bound(p) = mu + sigma * twd(1 - p) is affine in the TW quantile with
    mu, sigma independent of p. Evaluating at two thresholds recovers mu and
    sigma, which must equal their analytic (loc, inv_loc, scale) forms."""
    from diffract.core.compute.kernels.tracy_widom import tw_esd_bound

    greater_dim, lower_dim, std = 101, 64, 1.3  # sqrt(100)=10, sqrt(64)=8
    p_a, p_b = 0.005, 0.1
    twd_a = tracy_widom_ppf(1 - p_a)
    twd_b = tracy_widom_ppf(1 - p_b)
    bound_a = tw_esd_bound(greater_dim, lower_dim, std, p_value_threshold=p_a)
    bound_b = tw_esd_bound(greater_dim, lower_dim, std, p_value_threshold=p_b)

    sigma_hat = (bound_a - bound_b) / (twd_a - twd_b)
    mu_hat = bound_a - sigma_hat * twd_a

    loc = math.sqrt(greater_dim - 1) + math.sqrt(lower_dim)
    inv_loc = 1 / math.sqrt(greater_dim - 1) + 1 / math.sqrt(lower_dim)
    scale = std**2 / greater_dim
    assert mu_hat == pytest.approx(scale * loc**2)
    assert sigma_hat == pytest.approx(scale * loc * inv_loc ** (1 / 3))


@given(dims=matrix_dims(), std=positive_std, factor=st.floats(0.1, 10.0))
def test_tw_esd_bound_scales_quadratically_with_std(
    dims: tuple[int, int], std: float, factor: float
) -> None:
    """scale = sigma^2 / g feeds both mu and the TW term, so the whole bound
    is proportional to mp_bulk_std**2 (this also pins its monotone growth in
    the bulk std)."""
    from diffract.core.compute.kernels.tracy_widom import tw_esd_bound

    greater_dim, lower_dim = dims
    base = tw_esd_bound(greater_dim, lower_dim, std)
    scaled = tw_esd_bound(greater_dim, lower_dim, factor * std)

    assert scaled == pytest.approx(factor**2 * base, rel=1e-9)


def test_tw_num_spikes_counts_values_strictly_above_bound() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_num_spikes

    esd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spikes_above_three = 2  # 4.0 and 5.0 exceed the bound

    result = tw_num_spikes(esd, 3.0)
    assert result == spikes_above_three


def test_tw_num_spikes_uses_a_strict_boundary() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_num_spikes

    # A value exactly on the bound is bulk, not a spike (only 3.1 counts).
    assert tw_num_spikes(np.array([2.9, 3.0, 3.1]), 3.0) == 1
    # Degenerate ends stay well-defined.
    assert tw_num_spikes(np.array([]), 3.0) == 0
    assert tw_num_spikes(np.array([1.0, 2.0]), 3.0) == 0


def test_tw_esd_bound_propagates_nan_bulk_std() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_esd_bound

    # A failed MP fit yields mp_bulk_std = nan; the bound carries it through
    # instead of fabricating a finite threshold.
    assert np.isnan(tw_esd_bound(101, 64, float("nan")))


def test_tw_esd_bound_is_zero_for_a_dead_layer() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_esd_bound

    # A dead layer fits mp_bulk_std = 0; scale collapses to 0, so the bound is
    # a finite 0.0 with no 0/0 warning.
    assert tw_esd_bound(101, 64, 0.0) == 0.0


def test_tw_num_spikes_propagates_nan_bound() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_num_spikes

    # A failed MP fit gives tw_esd_bound = nan; the count propagates nan instead
    # of confidently reporting 0 spikes for a failed fit.
    assert np.isnan(tw_num_spikes(np.array([1.0, 2.0, 5.0]), float("nan")))


def test_tw_num_spikes_propagates_nan_spectrum() -> None:
    from diffract.core.compute.kernels.tracy_widom import tw_num_spikes

    # A nan eigenvalue must propagate rather than silently count as non-spike.
    assert np.isnan(tw_num_spikes(np.array([np.nan, np.nan, 5.0]), 3.0))
