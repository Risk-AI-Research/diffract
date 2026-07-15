"""Analytic tests for model-quality kernels."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_pl_alpha_weighted_uses_log10() -> None:
    from diffract.core.compute.kernels.model_quality import pl_alpha_weighted

    # alpha-hat = pl_alpha * log10(esd_max) (Martin-Peng-Mahoney convention);
    # alpha=2, esd_max=100 -> 2 * 2 = 4.
    assert pl_alpha_weighted(esd_max=100.0, pl_alpha=2.0) == pytest.approx(4.0)


def test_w1_rand_distance_is_normalized_wasserstein() -> None:
    from diffract.core.compute.kernels.model_quality import w1_rand_distance

    # Sorted W1 = mean(|1-1.5|, |2-2.5|, |3-2.5|, |4-3.5|) = 0.5, normalized by
    # the mean eigenvalue 2.5 -> 0.2. Hand value, independent of the kernel body.
    esd = np.array([1.0, 2.0, 3.0, 4.0])
    esd_rand = np.array([1.5, 2.5, 2.5, 3.5])
    assert w1_rand_distance(esd, esd_rand) == pytest.approx(0.2)


def test_w1_rand_distance_is_scale_invariant() -> None:
    from diffract.core.compute.kernels.model_quality import w1_rand_distance

    # Weights scaled by c scale the ESD by c^2; normalizing W1 by the shared mean
    # eigenvalue makes the index invariant, so it is comparable across layers.
    rng = np.random.default_rng(0)
    esd = np.sort(rng.random(200) + 0.1)
    esd_rand = np.sort(rng.random(200) + 0.1)
    base = w1_rand_distance(esd, esd_rand)

    assert base > 0
    assert w1_rand_distance(100.0 * esd, 100.0 * esd_rand) == pytest.approx(base)


def test_w1_rand_distance_is_permutation_invariant() -> None:
    from diffract.core.compute.kernels.model_quality import w1_rand_distance

    # A distribution distance depends on the multiset, not the order: comparing a
    # spectrum to a permutation of itself is zero.
    rng = np.random.default_rng(1)
    esd = np.sort(rng.random(200) + 0.1)
    assert w1_rand_distance(esd, rng.permutation(esd)) == pytest.approx(0.0)


def test_w1_rand_distance_edge_cases() -> None:
    from diffract.core.compute.kernels.model_quality import w1_rand_distance

    # A dead layer collapses the spectrum to zero energy -> distance 0,
    # warning-free (no division by a zero mean).
    assert w1_rand_distance(np.zeros(4), np.zeros(4)) == 0.0
    # A nan-corrupted spectrum propagates nan rather than fabricating a value.
    assert np.isnan(w1_rand_distance(np.full(4, np.nan), np.full(4, np.nan)))
