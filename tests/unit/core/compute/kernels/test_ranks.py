"""Analytic and property tests for rank kernels.

effective_rank and stable_rank are scale-invariant by construction, so each
carries a homogeneity property in addition to its analytic anchor;
mp_soft_rank is a plain edge-to-max ratio and needs only analytic points.
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
scales = st.floats(min_value=0.1, max_value=10.0)


def test_effective_rank_counts_equal_components() -> None:
    from diffract.core.compute.kernels.ranks import effective_rank

    assert effective_rank(np.ones(7)) == pytest.approx(7.0)
    assert effective_rank(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


@given(svals=positive_svals, scale=scales)
def test_effective_rank_is_scale_invariant(
    svals: NDArray[np.float64], scale: float
) -> None:
    from diffract.core.compute.kernels.ranks import effective_rank

    assert effective_rank(scale * svals) == pytest.approx(
        effective_rank(svals), rel=1e-6
    )


def test_stable_rank_matches_closed_form() -> None:
    from diffract.core.compute.kernels.ranks import stable_rank

    assert stable_rank(5.0, 4.0) == pytest.approx(1.5625)


@given(svals=positive_svals, scale=scales)
def test_stable_rank_is_scale_invariant(
    svals: NDArray[np.float64], scale: float
) -> None:
    from diffract.core.compute.kernels.ranks import stable_rank

    frob = float(np.sqrt(np.sum(svals**2)))
    spectral = float(np.max(svals))
    assert stable_rank(scale * frob, scale * spectral) == pytest.approx(
        stable_rank(frob, spectral), rel=1e-6
    )


def test_mp_soft_rank_is_bulk_edge_ratio() -> None:
    from diffract.core.compute.kernels.ranks import mp_soft_rank

    assert mp_soft_rank(2.0, 8.0) == pytest.approx(0.25)
    assert mp_soft_rank(5.0, 5.0) == pytest.approx(1.0)


def test_rank_kernels_propagate_nan() -> None:
    from diffract.core.compute.kernels.ranks import (
        effective_rank,
        mp_soft_rank,
        stable_rank,
    )

    # A diverged checkpoint (inf weights) makes the whole spectrum nan, so
    # frob and l2 are nan together; a failed MP fit makes mp_esd_max nan on an
    # otherwise finite spectrum. Mixed finite/nan spectra are not producible.
    assert np.isnan(effective_rank(np.full(3, np.nan)))
    assert np.isnan(stable_rank(np.nan, np.nan))
    assert np.isnan(mp_soft_rank(np.nan, 2.0))


def test_rank_kernels_survive_degenerate_spectra() -> None:
    from diffract.core.compute.kernels.ranks import (
        effective_rank,
        mp_soft_rank,
        stable_rank,
    )

    # EPS guards keep the zero (rank-collapsed) spectrum finite and warning-free.
    assert effective_rank(np.zeros(3)) == pytest.approx(1.0)
    assert stable_rank(0.0, 0.0) == 0.0
    assert mp_soft_rank(0.0, 0.0) == 0.0


def test_hard_rank_counts_relative_to_spectrum_max() -> None:
    from diffract.core.compute.kernels.ranks import hard_rank

    # esd_max = 100 -> threshold = 1e-5 * 100 = 1e-3; 1e-4 is below it (noise),
    # every larger entry is significant.
    esd = np.array([1e-4, 1.0, 50.0, 100.0])
    assert hard_rank(esd) == esd.size - 1


def test_hard_rank_is_scale_invariant() -> None:
    from diffract.core.compute.kernels.ranks import hard_rank

    # rank(cW) == rank(W): scaling the weights scales the whole esd, and a
    # threshold relative to the spectrum max keeps the count fixed.
    esd = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    assert hard_rank(0.01 * esd) == hard_rank(esd)
    assert hard_rank(100.0 * esd) == hard_rank(esd)
