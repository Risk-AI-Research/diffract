"""Analytic and property tests for Marchenko-Pastur kernels.

mp_sval_max inverts the ESD definition, so it carries an inversion property;
mp_concentration and mp_num_spikes are masked counts (order- and
fraction-bounded by construction), so analytic points on the interval and the
strict edge are what actually pin them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import (
    given,
    strategies as st,
)

pytestmark = pytest.mark.unit

positive_scalar = st.floats(
    min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
)


def _mp_producer_inputs(matrix: np.ndarray, *, seed: int = 42) -> dict:
    """Reproduce the wired producer chain that feeds marchenko_pastur_fit.

    The bleeding-correction bug only manifests through the interaction of the
    randomized spectrum, the real spectrum max, and the entry std, so the
    guarding inputs come from the real producer kernels rather than hand-built
    arrays.
    """
    from diffract.core.compute.kernels.mat_decomposition import (
        esd as esd_kernel,
        svd,
    )
    from diffract.core.compute.kernels.mat_properties import weights_rand, weights_std

    greater = max(matrix.shape)
    lower = min(matrix.shape)
    randomized = weights_rand(matrix, seed=seed)
    _, svals, _ = svd(matrix, allow_cuda=False)
    _, svals_rand, _ = svd(randomized, allow_cuda=False)
    real_esd = esd_kernel(svals, greater_dim=greater)
    esd_rand = esd_kernel(svals_rand, greater_dim=greater)
    return {
        "esd_rand": esd_rand,
        "esd_max": float(real_esd[-1]),
        "weights_std": weights_std(matrix),
        "aspect_ratio": greater / lower,
        "lower_dim": lower,
        "greater_dim": greater,
        "esd": real_esd,
    }


def test_mp_sval_max_matches_closed_form() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_sval_max

    assert mp_sval_max(2.0, 8) == pytest.approx(4.0)
    assert mp_sval_max(0.0, 5) == pytest.approx(0.0)


@given(sval=positive_scalar, greater_dim=st.integers(min_value=1, max_value=1000))
def test_mp_sval_max_inverts_esd_definition(sval: float, greater_dim: int) -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_sval_max

    esd_max = sval**2 / greater_dim
    assert mp_sval_max(esd_max, greater_dim) == pytest.approx(sval, rel=1e-9)


def test_mp_concentration_counts_closed_interval() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_concentration

    esd = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 10.0])
    # [1.0, 4.0] closed -> {1.0, 2.0, 3.0, 4.0} = 4 of 6 entries.
    assert mp_concentration(esd, 4.0, 1.0) == pytest.approx(4.0 / 6.0)


def test_mp_concentration_includes_both_endpoints() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_concentration

    edges = np.array([1.0, 4.0])
    # Both endpoints sit exactly on the interval; a strict comparison would
    # drop them and report 0.0.
    assert mp_concentration(edges, 4.0, 1.0) == pytest.approx(1.0)


def test_mp_num_spikes_counts_strictly_above_edge() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_num_spikes

    esd = np.array([1.0, 2.0, 5.0, 5.0, 9.0])
    # Only 9.0 exceeds 5.0; the two entries equal to the edge are bulk.
    spikes = mp_num_spikes(esd, 5.0)
    assert spikes == 1


def test_mp_num_spikes_zero_when_bulk_covers_everything() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_num_spikes

    esd = np.array([0.2, 0.5, 1.0, 2.0])
    assert mp_num_spikes(esd, 2.0) == 0


def test_mp_sval_max_propagates_nan_fit_edge() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_sval_max

    # A degenerate MP fit yields mp_esd_max = nan; the sqrt carries it through
    # instead of fabricating a finite singular value.
    assert np.isnan(mp_sval_max(float("nan"), 8))


def test_mp_masked_counts_survive_rank_collapsed_spectrum() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import (
        mp_concentration,
        mp_num_spikes,
    )

    # A dead / zero-initialised layer collapses the spectrum to zeros; against
    # a positive bulk edge above them the masked counts stay finite (0 bulk,
    # 0 spikes) and warning-free.
    zeros = np.zeros(4)
    assert mp_concentration(zeros, 4.0, 0.1) == pytest.approx(0.0)
    assert mp_num_spikes(zeros, 4.0) == 0


def test_mp_presence_is_clipped_to_unit_interval() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_presence

    # The MP bulk can be wider than the observed spectrum -> raw ratio > 1;
    # presence is a fraction and must clip to [0, 1].
    assert mp_presence(esd_min=0.0, esd_max=1.0, mp_esd_max=1.2, mp_esd_min=0.0) == (
        pytest.approx(1.0)
    )


def test_mp_presence_is_finite_on_zero_width_spectrum() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_presence

    # A constant / dead layer has zero spectral width; division must be guarded.
    result = mp_presence(esd_min=5.0, esd_max=5.0, mp_esd_max=5.0, mp_esd_min=5.0)
    assert np.isfinite(result)


def test_mp_concentration_propagates_nan_fit_edge() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_concentration

    esd = np.array([0.5, 1.0, 2.0])
    # A degenerate MP fit yields nan bounds; the bulk fraction must propagate
    # nan instead of collapsing the mask to an all-False, plausible 0.0.
    assert np.isnan(mp_concentration(esd, float("nan"), 1.0))
    assert np.isnan(mp_concentration(esd, 4.0, float("nan")))


def test_mp_concentration_propagates_nan_spectrum() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_concentration

    # A nan eigenvalue must propagate rather than silently drop out of the bulk
    # numerator while remaining in the denominator (a plausible 2/3).
    assert np.isnan(mp_concentration(np.array([1.0, np.nan, 2.0]), 4.0, 1.0))


def test_mp_num_spikes_propagates_nan_fit_edge() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_num_spikes

    # A degenerate MP fit yields mp_esd_max = nan; the count propagates nan
    # rather than confidently reporting 0 spikes for a failed fit.
    assert np.isnan(mp_num_spikes(np.array([1.0, 2.0, 9.0]), float("nan")))


def test_mp_num_spikes_propagates_nan_spectrum() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_num_spikes

    # A nan eigenvalue must propagate rather than silently count as non-spike.
    assert np.isnan(mp_num_spikes(np.array([1.0, np.nan, 9.0]), 4.0))


def test_mp_fit_bulk_variance_is_trace_identity() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import marchenko_pastur_fit

    # Q=4 -> bulk edge factor (1 + 1/sqrt(Q))^2 = 2.25; weights_std=1 places the
    # edge at 2.25. In esd_rand the 3.0 bleeds past the edge and [1, 2, 2] form
    # the bulk. The trace identity estimates the bulk variance as the mean of the
    # bulk eigenvalues: sigma^2 = (1 + 2 + 2) / lower_dim = 5 / 4 = 1.25, so the
    # fitted edge is 1.25 * 2.25 = 2.8125.
    mp_esd_max, mp_esd_min, mp_bulk_std = marchenko_pastur_fit(
        esd_rand=np.array([1.0, 2.0, 2.0, 3.0]),
        esd_max=3.0,
        weights_std=1.0,
        aspect_ratio=4.0,
        lower_dim=4,
    )

    assert mp_bulk_std == pytest.approx(math.sqrt(1.25))
    assert mp_esd_max == pytest.approx(2.8125)
    assert mp_esd_min == pytest.approx(1.25 * 0.25)


def test_mp_fit_bulk_std_recovers_planted_mean_sigma() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import marchenko_pastur_fit

    # A nonzero entry mean plants a rank-one spike above the MP bulk. The trace
    # identity recovers the true sigma because the bulk mean excludes the spike
    # mass that bleeds past the edge.
    rng = np.random.default_rng(0)
    sigma = 0.02
    matrix = rng.normal(sigma / 2, sigma, size=(400, 200))
    inputs = _mp_producer_inputs(matrix)

    _, _, mp_bulk_std = marchenko_pastur_fit(
        esd_rand=inputs["esd_rand"],
        esd_max=inputs["esd_max"],
        weights_std=inputs["weights_std"],
        aspect_ratio=inputs["aspect_ratio"],
        lower_dim=inputs["lower_dim"],
    )

    assert mp_bulk_std == pytest.approx(sigma, rel=0.05)


def test_mp_fit_keeps_edge_non_negative_through_outlier_pipeline() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import (
        marchenko_pastur_fit,
        mp_concentration,
        mp_num_spikes,
        mp_sval_max,
    )

    # A small single-outlier matrix is a library-profile input that drives the
    # bulk variance toward the edge; the trace identity keeps mp_esd_max
    # non-negative and the whole cascade finite under filterwarnings=error.
    matrix = np.full((6, 3), 1e-3)
    matrix[0, 0] = 5.0
    inputs = _mp_producer_inputs(matrix)

    mp_esd_max, mp_esd_min, mp_bulk_std = marchenko_pastur_fit(
        esd_rand=inputs["esd_rand"],
        esd_max=inputs["esd_max"],
        weights_std=inputs["weights_std"],
        aspect_ratio=inputs["aspect_ratio"],
        lower_dim=inputs["lower_dim"],
    )

    # A non-negative, finite edge keeps the whole cascade well-defined.
    assert mp_esd_max >= 0.0
    assert np.isfinite(mp_esd_max)
    assert np.isfinite(mp_bulk_std)
    # mp_sval_max takes the sqrt of a non-negative edge, so it stays finite.
    assert np.isfinite(mp_sval_max(mp_esd_max, inputs["greater_dim"]))
    # The spike count stays below the full spectrum size.
    assert mp_num_spikes(inputs["esd"], mp_esd_max) < inputs["esd"].size
    # Concentration stays a valid fraction in [0, 1].
    assert 0.0 <= mp_concentration(inputs["esd"], mp_esd_max, mp_esd_min) <= 1.0


def test_mp_ks_matches_two_sided_conditioned_reference() -> None:
    from scipy import stats

    from diffract.core.compute.extensions.rmt import marchenko_pastur_cdf
    from diffract.core.compute.kernels.marchenko_pastur import mp_ks

    rng = np.random.default_rng(0)
    esd = np.sort(
        np.linalg.svd(rng.standard_normal((400, 200)), compute_uv=False) ** 2 / 400
    )
    aspect_ratio = 2.0
    ratio = 1 / (aspect_ratio + 1e-8)
    sigma = float(np.sqrt(np.median(esd)))
    mp_esd_min = sigma**2 * (1 - np.sqrt(ratio)) ** 2
    # mp_esd_max is clipped below the theoretical edge (the reachable case where
    # esd_max < lambda_plus), so F(mp_esd_max) < 1 and conditioning is not a
    # no-op.
    mp_esd_max = 0.9 * sigma**2 * (1 + np.sqrt(ratio)) ** 2

    # Independent oracle: SciPy's two-sided KS against the MP CDF conditioned on
    # the same filtering window the kernel truncates to.
    mask = (mp_esd_min < esd) & (esd < mp_esd_max)
    f_a = float(marchenko_pastur_cdf(mp_esd_min, ratio, sigma))
    f_b = float(marchenko_pastur_cdf(mp_esd_max, ratio, sigma))

    def conditioned(x: np.ndarray) -> np.ndarray:
        return (marchenko_pastur_cdf(x, ratio, sigma) - f_a) / (f_b - f_a)

    reference = stats.kstest(esd[mask], conditioned).statistic

    assert mp_ks(aspect_ratio, sigma, esd, mp_esd_max, mp_esd_min) == pytest.approx(
        reference, rel=1e-9
    )


def test_mp_ks_is_sentinel_when_bulk_window_is_empty() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_ks

    # No eigenvalue falls inside (mp_esd_min, mp_esd_max): the fit is
    # inapplicable and the distance is the sentinel 1.0.
    esd = np.array([10.0, 20.0, 30.0])
    assert mp_ks(2.0, 0.5, esd, mp_esd_max=1.0, mp_esd_min=0.1) == 1.0


def test_mp_ks_is_sentinel_on_degenerate_fit() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_ks

    # A dead layer (mp_bulk_std=0) and a failed fit (nan bounds) both collapse
    # the bulk window; the sentinel must be reached before the MP CDF (which
    # rejects sigma <= 0) is evaluated, and stay warning-free.
    assert mp_ks(2.0, 0.0, np.zeros(5), mp_esd_max=0.0, mp_esd_min=0.0) == 1.0
    assert (
        mp_ks(
            2.0,
            float("nan"),
            np.array([1.0, 2.0, 3.0]),
            mp_esd_max=float("nan"),
            mp_esd_min=float("nan"),
        )
        == 1.0
    )


def test_mp_ks_is_sentinel_when_window_carries_no_model_mass() -> None:
    from diffract.core.compute.kernels.marchenko_pastur import mp_ks

    # Eigenvalues do fall inside (mp_esd_min, mp_esd_max), but the fitted MP
    # support (a tiny bulk std) sits far below the window, so the model CDF is
    # flat across it (F(max) - F(min) <= EPS). With no model mass to condition
    # on, the distance degrades to the sentinel instead of dividing by ~0.
    assert mp_ks(1.0, 0.01, np.array([1.5]), mp_esd_max=2.0, mp_esd_min=1.0) == 1.0
