"""Analytic and property tests for norm kernels.

Each homogeneous aggregation (frob_norm, alpha_norm, param_norm) carries its
degree-of-homogeneity property alongside an analytic anchor; l2_norm is a pass
-through and model_alpha_norm is a log-mean (not homogeneous), so both are
pinned by analytic points only.

The stacked bodies are tested once through the underlying function:
``alpha_norm`` backs pl_alpha_norm and tpl_alpha_norm, ``model_alpha_norm``
backs model_pl_alpha_norm and model_tpl_alpha_norm (the names differ only by
which input field feeds the same math).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = pytest.mark.unit

positive_svals = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=50),
    elements=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
positive_tuples = st.lists(
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=50,
).map(tuple)
scales = st.floats(min_value=0.1, max_value=10.0)


def test_frob_norm_is_euclidean_length_of_spectrum() -> None:
    from diffract.core.compute.kernels.norms import frob_norm

    assert frob_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert frob_norm(np.array([1.0, 2.0, 2.0])) == pytest.approx(3.0)


@given(svals=positive_svals, scale=scales)
def test_frob_norm_is_scale_equivariant(
    svals: NDArray[np.float64], scale: float
) -> None:
    from diffract.core.compute.kernels.norms import frob_norm

    assert frob_norm(scale * svals) == pytest.approx(scale * frob_norm(svals), rel=1e-9)


def test_l2_norm_passes_through_max_singular_value() -> None:
    from diffract.core.compute.kernels.norms import l2_norm

    assert l2_norm(3.5) == pytest.approx(3.5)


def test_alpha_norm_is_power_sum_of_spectrum() -> None:
    from diffract.core.compute.kernels.norms import alpha_norm

    esd = np.array([1.0, 2.0, 3.0])
    assert alpha_norm(esd, 2.0) == pytest.approx(14.0)
    assert alpha_norm(esd, 1.0) == pytest.approx(6.0)


@given(
    esd=positive_svals,
    alpha=st.floats(min_value=1.0, max_value=4.0),
    scale=st.floats(min_value=0.5, max_value=2.0),
)
def test_alpha_norm_is_homogeneous_of_degree_alpha(
    esd: NDArray[np.float64], alpha: float, scale: float
) -> None:
    from diffract.core.compute.kernels.norms import alpha_norm

    assert alpha_norm(scale * esd, alpha) == pytest.approx(
        scale**alpha * alpha_norm(esd, alpha), rel=1e-9
    )


def test_alpha_norm_is_nan_on_isometric_underflow() -> None:
    from diffract.core.compute.kernels.norms import alpha_norm

    # A near-isometric (orthogonal-init) layer has all esd < 1 and an unfittable
    # garbage-huge alpha; the weighted sum underflows to 0.0. That is a healthy
    # but unmeasurable layer, not a real zero -> nan (so the model mean skips it).
    esd = np.full(64, 1.0 / 64)
    assert np.isnan(alpha_norm(esd, 3.0e15))


def test_alpha_norm_is_finite_for_measurable_spectrum() -> None:
    from diffract.core.compute.kernels.norms import alpha_norm

    # A real spectrum (esd_max > 1, plausible alpha) stays finite and is NOT
    # skipped -- bad-but-measurable layers must still reach the model mean.
    result = alpha_norm(np.array([0.01, 0.1, 1.0, 5.0, 20.0]), 4.0)
    assert np.isfinite(result)
    assert result > 0


def test_param_norm_sums_squared_frobenius_norms() -> None:
    from diffract.core.compute.kernels.norms import param_norm

    assert param_norm((3.0, 4.0)) == pytest.approx(25.0)
    assert param_norm((1.0, 2.0, 2.0)) == pytest.approx(9.0)


@given(frobs=positive_tuples, scale=scales)
def test_param_norm_is_scale_squared_homogeneous(
    frobs: tuple[float, ...], scale: float
) -> None:
    from diffract.core.compute.kernels.norms import param_norm

    scaled = tuple(scale * frob for frob in frobs)
    assert param_norm(scaled) == pytest.approx(scale**2 * param_norm(frobs), rel=1e-9)


def test_model_alpha_norm_is_mean_log10_across_parameters() -> None:
    from diffract.core.compute.kernels.norms import model_alpha_norm

    assert model_alpha_norm((10.0, 100.0, 1000.0)) == pytest.approx(2.0)


def test_norm_kernels_propagate_nan() -> None:
    from diffract.core.compute.kernels.norms import (
        alpha_norm,
        frob_norm,
        param_norm,
    )

    # Per-parameter norms propagate nan from a corrupted spectrum or a failed
    # power-law fit. Model-level aggregators instead skip degenerate layers,
    # pinned separately by the model_alpha_norm / log-norm tests.
    assert np.isnan(frob_norm(np.full(3, np.nan)))
    assert np.isnan(alpha_norm(np.array([1.0, 2.0, 3.0]), np.nan))
    assert np.isnan(param_norm((3.0, np.nan, 4.0)))


def test_nuclear_norm_is_sum_of_singular_values() -> None:
    from diffract.core.compute.kernels.norms import nuclear_norm

    assert nuclear_norm(np.array([3.0, 4.0])) == pytest.approx(7.0)


def test_log_norm_is_mean_log10_of_squared_frobenius() -> None:
    from diffract.core.compute.kernels.norms import log_norm

    # mean(log10(frob^2)) = mean(2 log10 frob); frobs (10, 100) -> mean(2, 4) = 3.
    assert log_norm((10.0, 100.0)) == pytest.approx(3.0)


def test_log_norm_is_monotone_in_norm() -> None:
    from diffract.core.compute.kernels.norms import log_norm

    # log10 of the squared norm is strictly monotone in the norm: it decreases
    # as the norm shrinks below 1 (0.25 < 0.5 < 1.0).
    assert log_norm((0.25,)) < log_norm((0.5,)) < log_norm((1.0,))


def test_log_spectral_norm_is_mean_log10_of_squared_spectral() -> None:
    from diffract.core.compute.kernels.norms import log_spectral_norm

    assert log_spectral_norm((10.0, 100.0)) == pytest.approx(3.0)


def test_log_prod_frob_norm_is_sum_of_log10() -> None:
    from diffract.core.compute.kernels.norms import log_prod_frob_norm

    # log10(prod(10, 100, 1000)) = log10(1e6) = 6 = sum(1, 2, 3).
    assert log_prod_frob_norm((10.0, 100.0, 1000.0)) == pytest.approx(6.0)


def test_log_prod_frob_norm_survives_llm_scale() -> None:
    from diffract.core.compute.kernels.norms import log_prod_frob_norm

    # np.prod overflows to inf on hundreds of layers; the log domain stays finite.
    assert np.isfinite(log_prod_frob_norm(tuple([60.0] * 220)))


def test_log_prod_spectral_norm_is_sum_of_log10() -> None:
    from diffract.core.compute.kernels.norms import log_prod_spectral_norm

    assert log_prod_spectral_norm((10.0, 100.0, 1000.0)) == pytest.approx(6.0)


def test_model_alpha_norm_skips_degenerate_layers() -> None:
    import warnings

    from diffract.core.compute.kernels.norms import model_alpha_norm

    # An orthogonal-init layer's alpha_norm underflows to 0.0 and a diverged
    # layer is nan; both are unmeasurable, so they are skipped (no fabricated
    # sentinel) and the mean is over the healthy layers, warning-free.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model_alpha_norm((1000.0, 100.0, 0.0)) == pytest.approx(
            np.nanmean([3.0, 2.0])
        )
        assert model_alpha_norm((1000.0, np.nan)) == pytest.approx(3.0)
        # A fully degenerate model has nothing to measure -> nan, warning-free.
        assert np.isnan(model_alpha_norm((0.0, float("nan"))))


def test_log_norms_skip_dead_layers() -> None:
    from diffract.core.compute.kernels.norms import (
        log_norm,
        log_prod_frob_norm,
        log_prod_spectral_norm,
        log_spectral_norm,
    )

    # A dead layer (frob = 0) is not measurable; it is skipped rather than
    # dragging the model aggregate to -inf. Kept frobs (10, 100) -> mean(2, 4).
    assert log_norm((10.0, 100.0, 0.0)) == pytest.approx(3.0)
    assert log_spectral_norm((10.0, 100.0, 0.0)) == pytest.approx(3.0)
    # The log-product skips the zero factor instead of collapsing to -inf.
    assert log_prod_frob_norm((10.0, 100.0, 0.0)) == pytest.approx(3.0)
    assert log_prod_spectral_norm((10.0, 100.0, 0.0)) == pytest.approx(3.0)


def test_log_norms_are_nan_when_all_layers_dead() -> None:
    import warnings

    from diffract.core.compute.kernels.norms import log_norm, log_prod_frob_norm

    # No measurable layer -> nan (not -inf, not 0), warning-free.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(log_norm((0.0, 0.0)))
        assert np.isnan(log_prod_frob_norm((0.0, 0.0)))
