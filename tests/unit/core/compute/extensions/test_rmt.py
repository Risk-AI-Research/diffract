"""Tests for the vendored RMT distributions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diffract.core.compute.extensions.rmt import (
    marchenko_pastur_cdf,
    tracy_widom_ppf,
)

pytestmark = pytest.mark.unit

GOLDEN = json.loads((Path(__file__).parent / "_golden" / "rmt_values.json").read_text())

TW_LITERATURE_QUANTILES = {
    0.5: -1.2686225,
    0.95: 0.9793,
    0.995: 2.4224,
    0.01: -3.8954,
}


def test_marchenko_pastur_cdf_matches_scikit_rmt_golden() -> None:
    for case in GOLDEN["marchenko_pastur_cdf"]:
        ours = marchenko_pastur_cdf(np.array(case["x"]), case["ratio"], case["sigma"])
        reference = np.array(case["cdf"])
        finite = np.isfinite(reference)
        np.testing.assert_allclose(ours[finite], reference[finite], atol=1e-6)


def test_marchenko_pastur_cdf_derivative_matches_density() -> None:
    for ratio, sigma in [(0.25, 1.0), (0.5, 2.0), (0.999999, 0.7)]:
        sqrt_ratio = np.sqrt(ratio)
        lo = sigma**2 * (1 - sqrt_ratio) ** 2
        hi = sigma**2 * (1 + sqrt_ratio) ** 2
        span = hi - lo
        x = np.linspace(lo + 1e-4 * span, hi - 1e-4 * span, 501)
        h = span * 1e-7

        numeric = (
            marchenko_pastur_cdf(x + h, ratio, sigma)
            - marchenko_pastur_cdf(x - h, ratio, sigma)
        ) / (2 * h)
        analytic = np.sqrt((hi - x) * (x - lo)) / (2 * np.pi * ratio * sigma**2 * x)
        np.testing.assert_allclose(numeric, analytic, atol=1e-6 * np.max(analytic))


def test_marchenko_pastur_cdf_bounds_and_monotonicity() -> None:
    x = np.linspace(0.0, 5.0, 1001)
    cdf = marchenko_pastur_cdf(x, 0.5, 1.0)

    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= 0)


def test_marchenko_pastur_cdf_validates_arguments() -> None:
    with pytest.raises(ValueError, match="ratio"):
        marchenko_pastur_cdf(np.array([1.0]), 1.5, 1.0)
    with pytest.raises(ValueError, match="ratio"):
        marchenko_pastur_cdf(np.array([1.0]), 0.0, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        marchenko_pastur_cdf(np.array([1.0]), 0.5, 0.0)


def test_tracy_widom_ppf_matches_scikit_rmt_golden() -> None:
    case = GOLDEN["tracy_widom_ppf"]
    for q, reference in zip(case["q"], case["ppf"], strict=True):
        assert tracy_widom_ppf(q) == pytest.approx(reference, abs=5e-3)


def test_tracy_widom_ppf_matches_literature() -> None:
    for q, reference in TW_LITERATURE_QUANTILES.items():
        assert tracy_widom_ppf(q) == pytest.approx(reference, abs=5e-4)


def test_tracy_widom_ppf_validates_domain() -> None:
    with pytest.raises(ValueError, match="range"):
        tracy_widom_ppf(1e-12)
    with pytest.raises(ValueError, match="range"):
        tracy_widom_ppf(1.0)
