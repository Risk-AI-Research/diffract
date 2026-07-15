"""Tests for floating-point environment preservation across native initializers."""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FLOAT32_SUBNORMAL = np.float32(1e-20)
"""Squares to 1e-40, below the ~1.18e-38 float32 subnormal threshold."""


def _float32_subnormal_survives() -> bool:
    return float(FLOAT32_SUBNORMAL * FLOAT32_SUBNORMAL) > 0.0


def _float64_subnormal_survives() -> bool:
    return (sys.float_info.min / 2.0) > 0.0


def _skip_without_taichi() -> None:
    from diffract.core.compute.kernels import heavy_tailed

    if not heavy_tailed._DIFFRACT_FIT_AVAILABLE:
        pytest.skip("taichi extra not installed")


def test_subnormals_resolve_by_default() -> None:
    """The baseline the rest of the module is measured against: an untouched
    process resolves subnormal float32 and float64 values rather than flushing
    them to zero."""
    assert _float32_subnormal_survives()
    assert _float64_subnormal_survives()


def test_accelerated_fit_leaves_subnormals_resolvable(ram_container) -> None:
    """Running the accelerated fit must not change the floating-point semantics
    of unrelated numpy work.

    The taichi runtime is initialized once per process, on first import of the
    extension, and any flush-to-zero flag it sets is a thread control-register
    bit that persists for the rest of the process. So this assertion holds no
    matter which test imported the extension first. It matters because squared
    singular values of a small-magnitude layer fall in the subnormal range: a
    flushed environment reports a Frobenius norm of exactly zero for a layer
    whose weights are tiny but nonzero, making it indistinguishable from a dead
    layer.
    """
    _skip_without_taichi()
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    esd = np.random.default_rng(0).pareto(1.5, 4000) + 1.0
    power_law_fit(esd, fit_method="diffract")

    assert _float32_subnormal_survives()
    assert _float64_subnormal_survives()

    weights = np.diag([1e-160, 1e-170, 1e-180, 1e-200]).astype(np.float64)
    spectrum = np.linalg.svd(weights, compute_uv=False)
    assert float(np.sum(spectrum**2)) > 0.0


def test_accelerated_fit_is_unaffected_by_the_restored_environment(
    ram_container,
) -> None:
    """The fitted parameters do not depend on the flush-to-zero flag: repeated
    fits of the same sample agree exactly with the environment restored."""
    _skip_without_taichi()
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    esd = np.random.default_rng(0).pareto(1.5, 4000) + 1.0

    first = power_law_fit(esd, fit_method="diffract")
    second = power_law_fit(esd, fit_method="diffract")

    assert first == pytest.approx(second, rel=1e-12)
