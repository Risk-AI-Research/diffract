"""Unit tests for heavy-tailed fit kernels."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from diffract.core import WEIGHTS_FIELD
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.session import Session

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = pytest.mark.unit


def _pareto_sample(alpha: float, size: int, seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.pareto(alpha - 1.0, size) + 1.0


def _tpl_sample(size: int = 2000, seed: int = 10) -> NDArray[np.float64]:
    """Rejection-sample TPL(alpha=2.5, lambda=0.1, xmin=1)."""
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    while len(samples) < size:
        candidate = 1.0 + rng.exponential(10.0, 100_000)
        accepted = candidate[rng.random(100_000) < candidate ** (-2.5)]
        samples.extend(accepted.tolist())
    return np.array(samples[:size])


def test_power_law_fit_recovers_pareto_alpha(ram_container) -> None:
    """Pure Pareto(alpha=2.5, xmin=1): the exponent is recovered and the fitted
    support starts at the generative xmin. The KS distance of a correct fit to
    its own generative law is small — a bound of 0.05 is ~4x the spread observed
    across seeds, so a fit that misses the law fails it."""
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    esd = _pareto_sample(alpha=2.5, size=4000)

    pl_alpha, pl_esd_xmin, pl_ks = power_law_fit(esd)

    assert pl_alpha == pytest.approx(2.5, abs=0.2)
    assert pl_esd_xmin == pytest.approx(1.0, rel=0.05)
    assert pl_ks < 0.05


def test_power_law_fit_locates_the_tail_onset(ram_container) -> None:
    """A Pareto tail grafted onto a lognormal body puts the true xmin (3.0)
    strictly above the data minimum (~0.1). The fit must locate the tail onset
    and exclude the body: an xmin left at min(esd) admits the lognormal bulk and
    cannot recover the tail exponent. The xmin estimator is biased upward (it may
    always cut deeper into the tail), so the tolerance follows the spread measured
    across seeds rather than the point estimate."""
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    rng = np.random.default_rng(0)
    body = rng.lognormal(0.0, 0.6, 4000)
    body = body[body < 3.0]
    tail = 3.0 * (rng.pareto(1.5, 1500) + 1.0)
    esd = np.concatenate([body, tail])

    pl_alpha, pl_esd_xmin, pl_ks = power_law_fit(esd)

    assert pl_esd_xmin == pytest.approx(3.0, rel=0.1)
    assert pl_esd_xmin > 10.0 * float(esd.min())
    assert pl_alpha == pytest.approx(2.5, abs=0.15)
    assert pl_ks < 0.05


def test_resolve_fit_method(ram_container, monkeypatch) -> None:
    from diffract.core.compute.kernels import heavy_tailed as ht

    assert ht._resolve_fit_method("powerlaw", 10**6) == "powerlaw"
    assert ht._resolve_fit_method("diffract", 10) == "diffract"

    monkeypatch.setattr(ht, "_accelerated_fit_ready", lambda: False)
    assert ht._resolve_fit_method("auto", 10**6) == "powerlaw"

    monkeypatch.setattr(ht, "_accelerated_fit_ready", lambda: True)
    assert ht._resolve_fit_method("auto", ht.AUTO_DIFFRACT_MIN_SIZE) == "diffract"
    assert ht._resolve_fit_method("auto", ht.AUTO_DIFFRACT_MIN_SIZE - 1) == "powerlaw"


def test_resolve_fit_method_falls_back_when_taichi_broken(
    ram_container, monkeypatch
) -> None:
    """taichi may be importable yet fail to initialize; `auto` must
    degrade to the powerlaw library instead of raising."""
    from diffract.core.compute.kernels import heavy_tailed as ht

    def _broken() -> None:
        msg = "taichi failed to initialize"
        raise ModuleNotFoundError(msg)

    monkeypatch.setattr(ht, "_DIFFRACT_FIT_AVAILABLE", True)
    monkeypatch.setattr(ht, "_DIFFRACT_FIT_BROKEN", False)
    monkeypatch.setattr(ht, "_get_fit_class", _broken)

    assert ht._resolve_fit_method("auto", 10**6) == "powerlaw"
    assert ht._DIFFRACT_FIT_BROKEN is True
    # The failure is cached: no repeated import attempts.
    assert ht._resolve_fit_method("auto", 10**6) == "powerlaw"


def test_fit_kernels_auto_uses_powerlaw_below_threshold(
    ram_container, monkeypatch
) -> None:
    from diffract.core.compute.kernels import heavy_tailed as ht

    def _fail(esd):
        msg = "auto must not pick diffract below the threshold"
        raise AssertionError(msg)

    monkeypatch.setattr(ht, "power_law_fit_diffract_implementation", _fail)
    monkeypatch.setattr(ht, "exponential_fit_diffract_implementation", _fail)

    esd = _pareto_sample(alpha=3.0, size=ht.AUTO_DIFFRACT_MIN_SIZE - 1, seed=1)

    for fit in (ht.power_law_fit, ht.exponential_fit):
        results = fit(esd)
        assert all(np.isfinite(value) for value in results)


def test_fit_kernels_auto_uses_diffract_above_threshold(
    ram_container, monkeypatch
) -> None:
    from diffract.core.compute.kernels import heavy_tailed as ht

    calls: list[int] = []
    monkeypatch.setattr(ht, "_accelerated_fit_ready", lambda: True)
    monkeypatch.setattr(
        ht,
        "power_law_fit_diffract_implementation",
        lambda esd: (calls.append(esd.size), (2.0, 1.0, 0.1))[1],
    )

    esd = _pareto_sample(alpha=3.0, size=ht.AUTO_DIFFRACT_MIN_SIZE, seed=1)
    assert ht.power_law_fit(esd) == (2.0, 1.0, 0.1)
    assert calls == [esd.size]


@pytest.mark.slow
def test_truncated_power_law_fit_auto_works_on_small_esd(ram_container) -> None:
    from diffract.core.compute.kernels.heavy_tailed import truncated_power_law_fit

    esd = _pareto_sample(alpha=3.0, size=200, seed=1)

    results = truncated_power_law_fit(esd)
    assert all(np.isfinite(value) for value in results)


def _skip_without_taichi(ram_container) -> None:
    from diffract.core.compute.kernels import heavy_tailed

    if not heavy_tailed._DIFFRACT_FIT_AVAILABLE:
        pytest.skip("taichi extra not installed")


@pytest.mark.slow
def test_diffract_power_law_fit_matches_powerlaw(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    esd = _pareto_sample(alpha=2.5, size=5000)

    ref_alpha, ref_xmin, ref_ks = power_law_fit(esd, fit_method="powerlaw")
    acc_alpha, acc_xmin, acc_ks = power_law_fit(esd, fit_method="diffract")

    assert acc_alpha == pytest.approx(ref_alpha, abs=0.15)
    assert acc_xmin == pytest.approx(ref_xmin, rel=0.25)
    assert acc_ks == pytest.approx(ref_ks, abs=0.02)


@pytest.mark.slow
def test_diffract_exponential_fit_matches_powerlaw(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import exponential_fit

    rng = np.random.default_rng(3)
    esd = 1.0 + rng.exponential(0.05, 5000)

    ref_lambda, _, _ = exponential_fit(esd, fit_method="powerlaw")
    acc_lambda, _, _ = exponential_fit(esd, fit_method="diffract")

    assert acc_lambda == pytest.approx(ref_lambda, rel=0.1)


@pytest.mark.slow
def test_diffract_p_value_accepts_power_law_data(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import pl_p_value, power_law_fit

    esd = _pareto_sample(alpha=2.5, size=5000)
    pl_alpha, pl_esd_xmin, pl_ks = power_law_fit(esd)

    p_value = pl_p_value(esd, pl_alpha, pl_esd_xmin, pl_ks)

    # Clauset et al. plausibility threshold; clean Pareto data sits far above.
    assert p_value > 0.1


@pytest.mark.slow
def test_diffract_p_value_rejects_exponential_data_under_power_law(
    ram_container,
) -> None:
    """At a fixed xmin covering the whole sample, exponential data must be
    decisively rejected as a power law. (With a free xmin the CSN procedure
    may legitimately retreat to a tiny tail where PL is plausible.)"""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import pl_p_value

    rng = np.random.default_rng(6)
    esd = np.sort(1.0 + rng.exponential(1.0, 5000))
    xmin = 1.0

    pl_alpha = 1.0 + esd.size / np.sum(np.log(esd / xmin))
    model_cdf = 1.0 - (esd / xmin) ** (1.0 - pl_alpha)
    pl_ks = float(np.max(np.abs(np.arange(esd.size) / esd.size - model_cdf)))

    p_value = pl_p_value(esd, pl_alpha, xmin, pl_ks)

    assert p_value < 0.01


@pytest.mark.slow
def test_diffract_p_value_is_deterministic(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import pl_p_value, power_law_fit

    esd = _pareto_sample(alpha=2.5, size=3000, seed=8)
    pl_alpha, pl_esd_xmin, pl_ks = power_law_fit(esd)

    first = pl_p_value(esd, pl_alpha, pl_esd_xmin, pl_ks)
    second = pl_p_value(esd, pl_alpha, pl_esd_xmin, pl_ks)

    assert first == second


@pytest.mark.slow
def test_diffract_p_value_is_nan_for_tiny_tails(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import pl_p_value

    esd = _pareto_sample(alpha=2.5, size=40, seed=9)

    p_value = pl_p_value(esd, 2.5, float(esd.min()), 0.05)

    assert np.isnan(p_value)


@pytest.mark.slow
def test_diffract_fit_handles_degenerate_data(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import (
        power_law_fit_diffract_implementation,
    )

    pl_alpha, pl_esd_xmin, pl_ks = power_law_fit_diffract_implementation(
        np.full(100, 3.0)
    )

    assert np.isnan(pl_alpha)
    assert np.isnan(pl_esd_xmin)
    assert np.isnan(pl_ks)


@pytest.mark.slow
def test_diffract_truncated_power_law_fit_recovers_parameters(ram_container) -> None:
    """The TPL fit is a maximum-likelihood estimate; on synthetic
    TPL(alpha=2.5, lambda=0.1) data both parameters must be recovered
    away from their bounds — lambda in particular must not collapse onto
    MIN_LAMBDA."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import (
        truncated_power_law_fit_diffract_implementation,
    )

    tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks = (
        truncated_power_law_fit_diffract_implementation(_tpl_sample())
    )

    assert tpl_alpha == pytest.approx(2.5, abs=0.5)
    assert tpl_lambda == pytest.approx(0.1, rel=0.6)
    assert np.isfinite(tpl_esd_xmin)
    assert 0 <= tpl_ks <= 1


@pytest.mark.slow
def test_diffract_fit_matches_powerlaw_on_body_tail_data(ram_container) -> None:
    """Canonical CSN scenario: a non-power-law body plus a power-law tail.
    xmin selection must find the tail onset (global KS minimum)."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import power_law_fit

    rng = np.random.default_rng(0)
    body = rng.lognormal(0.0, 0.6, 4000)
    body = body[body < 3.0]
    tail = 3.0 * (rng.pareto(1.5, 1500) + 1.0)
    esd = np.concatenate([body, tail])

    ref_alpha, ref_xmin, ref_ks = power_law_fit(esd, fit_method="powerlaw")
    acc_alpha, acc_xmin, acc_ks = power_law_fit(esd, fit_method="diffract")

    assert acc_xmin == pytest.approx(ref_xmin, rel=0.05)
    assert acc_alpha == pytest.approx(ref_alpha, abs=0.1)
    assert acc_ks == pytest.approx(ref_ks, abs=0.005)


@pytest.mark.slow
def test_diffract_exponential_fit_survives_large_shift(ram_container) -> None:
    """Shifted-form CDFs: at lambda*xmin ~ 4000 the f32 evaluation stays in
    range, so the fit must come back finite."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import exponential_fit

    rng = np.random.default_rng(3)
    esd = 201.0 + rng.exponential(0.05, 5000)

    expon_lambda, expon_esd_xmin, _ = exponential_fit(esd, fit_method="diffract")

    assert expon_lambda == pytest.approx(20.0, rel=0.1)
    assert expon_esd_xmin == pytest.approx(201.0, rel=0.01)


@pytest.mark.slow
def test_diffract_fit_rejects_empty_data_without_poisoning_runtime(
    ram_container,
) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import (
        _get_fit_class,
        power_law_fit_diffract_implementation,
    )

    with pytest.raises(ValueError, match="empty"):
        _get_fit_class()(np.array([]), "power_law")

    # The failed construction must not break subsequent fits.
    pl_alpha, _, _ = power_law_fit_diffract_implementation(
        _pareto_sample(alpha=2.5, size=1000, seed=11)
    )
    assert np.isfinite(pl_alpha)


@pytest.mark.slow
def test_diffract_fit_raises_after_close(ram_container) -> None:
    """A closed Fit must raise rather than reach freed native memory."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    fit = _get_fit_class()(_pareto_sample(alpha=2.5, size=200, seed=12), "power_law")
    fit.fit_params()
    fit.close()
    fit.close()

    with pytest.raises(RuntimeError, match="closed"):
        fit.fit_params()
    with pytest.raises(RuntimeError, match="closed"):
        fit.p_value_test()


@pytest.mark.slow
def test_diffract_fit_rejects_invalid_input(ram_container) -> None:
    """Inf values and 2-D input are rejected at construction, so no fit can
    report confident parameters for them."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    fit_class = _get_fit_class()
    data = _pareto_sample(alpha=2.5, size=1000, seed=13)

    with pytest.raises(ValueError, match="finite"):
        fit_class(np.concatenate([data, [np.inf]]), "power_law")
    with pytest.raises(ValueError, match="1-D"):
        fit_class(data.reshape(50, 20), "power_law")
    with pytest.raises(ValueError, match="seed"):
        fit_class(data, "power_law", seed=-1)


@pytest.mark.slow
def test_diffract_fit_is_scale_invariant(ram_container) -> None:
    """Fits run on internally normalized data, so the absolute parameter
    bounds (MIN_LAMBDA in particular) must not depend on the data scale."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import (
        exponential_fit,
        power_law_fit,
    )

    esd = _pareto_sample(alpha=2.5, size=3000, seed=14)
    base_alpha, base_xmin, base_ks = power_law_fit(esd, fit_method="diffract")
    for scale in (1e-6, 1e6):
        alpha, xmin, ks = power_law_fit(esd * scale, fit_method="diffract")
        assert alpha == pytest.approx(base_alpha, rel=1e-2)
        assert xmin == pytest.approx(base_xmin * scale, rel=1e-2)
        assert ks == pytest.approx(base_ks, abs=1e-3)

    # lambda_true = 2e-5 sits far below the (normalized-units) 1e-4 floor,
    # so the fit must still return finite parameters.
    rng = np.random.default_rng(15)
    exp_esd = (1.0 + rng.exponential(0.05, 4000)) * 1e6
    expon_lambda, expon_xmin, _ = exponential_fit(exp_esd, fit_method="diffract")
    assert expon_lambda == pytest.approx(20.0 / 1e6, rel=0.1)
    assert expon_xmin == pytest.approx(1e6, rel=0.05)


@pytest.mark.slow
def test_diffract_truncated_power_law_fit_survives_large_scale(ram_container) -> None:
    """At data scale 1e4 the true lambda is 1e-5, far below the parameter
    floor; the fit must still recover alpha rather than settle on the
    bounds-penalty plateau with alpha pinned at ~1."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import (
        truncated_power_law_fit_diffract_implementation,
    )

    scale = 1e4
    tpl_alpha, tpl_lambda, tpl_esd_xmin, _ = (
        truncated_power_law_fit_diffract_implementation(_tpl_sample() * scale)
    )

    assert tpl_alpha == pytest.approx(2.5, abs=0.5)
    assert tpl_lambda == pytest.approx(0.1 / scale, rel=0.6)
    assert tpl_esd_xmin == pytest.approx(scale, rel=0.2)


@pytest.mark.slow
def test_diffract_tpl_p_value_survives_alpha_near_one(ram_container) -> None:
    """At alpha ~ 1 the f32 power-law proposal would overflow into inf
    draws, so the exponential proposal must take over (acceptance-ratio
    rule) and keep the bootstrap finite."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    rng = np.random.default_rng(16)
    esd = 1.0 + rng.exponential(1.0, 300)
    fit = _get_fit_class()(esd, "truncated_power_law")
    try:
        fit.set_params(
            xmin=1.0,
            pl_alpha=1.001,
            expon_lambda=0.99,
            ks_distance=0.05,
            tail_size=300,
        )
        _, p_value = fit.p_value_test()
    finally:
        fit.close()

    assert np.isfinite(p_value)


@pytest.mark.slow
def test_diffract_set_params_is_atomic(ram_container) -> None:
    """A rejected set_params call must leave the previous parameters intact."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    fit = _get_fit_class()(_pareto_sample(alpha=2.5, size=200, seed=17), "power_law")
    try:
        fit.set_params(xmin=1.5, pl_alpha=2.0, ks_distance=0.1, tail_size=100)
        with pytest.raises(ValueError, match="alpha"):
            fit.set_params(alpha=2.0)
        assert fit.params == {
            "xmin": 1.5,
            "pl_alpha": 2.0,
            "ks_distance": 0.1,
            "tail_size": 100,
        }
    finally:
        fit.close()


@pytest.mark.slow
def test_diffract_fit_pools_are_reused_and_history_independent(ram_container) -> None:
    """Fields and compiled kernels are pooled by padded data size:
    repeated fits must reuse the pool entry, results must not depend on
    what was fitted before, and a concurrent same-bucket instance must
    fall back to private fields without corrupting either fit."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.extensions import power_law as accelerated

    data_a = _pareto_sample(alpha=2.5, size=3000, seed=18)
    data_b = _pareto_sample(alpha=2.0, size=2500, seed=19)  # same 4096 bucket

    def fit_of(data):
        fit = accelerated.Fit(data, "power_law")
        try:
            params = dict(fit.fit_params())
            _, p_value = fit.p_value_test()
        finally:
            fit.close()
        return params, p_value

    first = fit_of(data_a)
    fit_of(data_b)
    second = fit_of(data_a)

    assert repr(first) == repr(second)

    open_fits = [
        accelerated.Fit(data_a, "power_law")
        for _ in range(accelerated.MAX_MAIN_POOLS_PER_SIZE + 1)
    ]
    try:
        pooled_flags = [fit._main_pooled for fit in open_fits]
        assert pooled_flags == [True] * accelerated.MAX_MAIN_POOLS_PER_SIZE + [False]
        results = {repr(dict(fit.fit_params())) for fit in open_fits}
        assert len(results) == 1
    finally:
        for fit in open_fits:
            fit.close()


@pytest.mark.slow
def test_diffract_fit_does_not_leak_memory_across_instances(ram_container) -> None:
    """Kernels compile once per process and fields are pooled, so repeated
    Fit instances must not accumulate compiled artifacts."""
    _skip_without_taichi(ram_container)
    resource = pytest.importorskip("resource")

    from diffract.core.compute.extensions import power_law as accelerated

    data = _pareto_sample(alpha=2.5, size=2000, seed=20)
    for _ in range(3):
        fit = accelerated.Fit(data, "power_law")
        fit.fit_params()
        fit.close()

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(40):
        fit = accelerated.Fit(data, "power_law")
        fit.fit_params()
        fit.close()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is bytes on macOS and KiB on Linux; either way the growth
    # over 40 cycles must stay far below the ~2 MB of compiled artifacts a
    # single per-instance compilation would retain.
    growth_mb = (after - before) / (1024 * 1024 if sys.platform == "darwin" else 1024)
    assert growth_mb < 30


@pytest.mark.slow
def test_diffract_set_params_validates_values(ram_container) -> None:
    """Inconsistent injected parameters (tail_size > n) must be rejected
    rather than yield a meaningless (True, 1.0) plausibility verdict."""
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    fit = _get_fit_class()(_pareto_sample(alpha=2.5, size=200, seed=21), "power_law")
    try:
        with pytest.raises(ValueError, match="tail_size"):
            fit.set_params(xmin=1.0, pl_alpha=2.0, ks_distance=0.1, tail_size=100_000)
        with pytest.raises(ValueError, match="real number"):
            fit.set_params(xmin="garbage")
        with pytest.raises(ValueError, match="xmin"):
            fit.set_params(xmin=-1.0, pl_alpha=2.0, ks_distance=0.1, tail_size=100)
    finally:
        fit.close()


@pytest.mark.slow
def test_diffract_p_value_refuses_oversized_bootstrap(ram_container) -> None:
    _skip_without_taichi(ram_container)
    from diffract.core.compute.kernels.heavy_tailed import _get_fit_class

    fit = _get_fit_class()(np.linspace(1.0, 2.0, 200_001), "power_law")
    try:
        fit.set_params(xmin=1.0, pl_alpha=2.5, ks_distance=0.05, tail_size=200_001)
        with pytest.raises(ValueError, match="subsample"):
            fit.p_value_test()
    finally:
        fit.close()


@pytest.mark.slow
def test_diffract_fit_is_faster_than_powerlaw(ram_container) -> None:
    _skip_without_taichi(ram_container)
    import time

    from diffract.core.compute.kernels.heavy_tailed import (
        power_law_fit_diffract_implementation,
        power_law_fit_powerlaw_implementation,
    )

    # Warm up the taichi JIT outside the measured region.
    power_law_fit_diffract_implementation(_pareto_sample(alpha=2.5, size=500, seed=4))

    esd = _pareto_sample(alpha=2.5, size=10_000, seed=5)

    start = time.perf_counter()
    power_law_fit_powerlaw_implementation(esd)
    reference_time = time.perf_counter() - start

    start = time.perf_counter()
    power_law_fit_diffract_implementation(esd)
    accelerated_time = time.perf_counter() - start

    # Measured ~9x at n=10k; require a conservative 2x to stay robust.
    assert accelerated_time < reference_time / 2


def test_fit_kernels_reject_unknown_method(ram_container) -> None:
    from diffract.core.compute.kernels.heavy_tailed import (
        exponential_fit,
        power_law_fit,
        truncated_power_law_fit,
    )

    esd = _pareto_sample(alpha=3.0, size=100, seed=2)

    for fit in (power_law_fit, truncated_power_law_fit, exponential_fit):
        with pytest.raises(ValueError, match="not implemented"):
            fit(esd, fit_method="taichi")  # type: ignore[arg-type]


def test_explicit_diffract_without_taichi_fails_loudly(ram_container) -> None:
    """Explicitly requesting the accelerated path when the taichi extra is
    absent must fail loudly and name the extra to install, never silently
    fall back (the fit kernels and the p-value functions share one gate)."""
    import diffract.core.utils.imports as import_utils
    from diffract.core.compute.kernels import heavy_tailed as ht

    if import_utils.is_available("taichi"):
        pytest.skip("taichi extra installed; the missing-extra path is not exercised")

    esd = _pareto_sample(alpha=2.5, size=200, seed=2)

    # The error must name the exact extra to install, per the
    # optional-dependency contract (not merely mention taichi somewhere).
    for fit in (ht.power_law_fit, ht.truncated_power_law_fit, ht.exponential_fit):
        with pytest.raises(ModuleNotFoundError, match=r"diffract-core\[taichi\]"):
            fit(esd, fit_method="diffract")

    for p_value, args in (
        (ht.pl_p_value, (esd, 2.5, 1.0, 0.05)),
        (ht.tpl_p_value, (esd, 2.5, 0.1, 1.0, 0.05)),
        (ht.expon_p_value, (esd, 0.1, 1.0, 0.05)),
    ):
        with pytest.raises(ModuleNotFoundError, match=r"diffract-core\[taichi\]"):
            p_value(*args)


def test_session_resolves_heavy_tailed_chain(ram_container) -> None:
    repository = ram_container.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="p0",
        name="layer.0.weight",
        ptype=ParameterType.DENSE,
        model_id="m1",
    )
    proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
    rng = np.random.default_rng(0)
    proxy.set_field(WEIGHTS_FIELD, rng.standard_normal((64, 48)).astype(np.float32))

    session = Session(container=ram_container)
    session.compute.apply("pl_ks")
    session.compute.apply("expon_concentration")

    scalars = session.results.export_metrics(
        "pl_ks", "expon_concentration", export_format="dict"
    )
    fields = scalars[meta.uid]["fields"]
    assert np.isfinite(fields["pl_ks"])
    assert 0.0 <= fields["expon_concentration"] <= 1.0


def test_ht_presence_is_tail_width_fraction() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_presence

    assert ht_presence(esd_min=1.0, esd_max=10.0, ht_esd_xmin=4.0) == pytest.approx(
        6.0 / 9.0
    )


def test_ht_presence_propagates_nan_xmin() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_presence

    assert np.isnan(ht_presence(esd_min=1.0, esd_max=10.0, ht_esd_xmin=float("nan")))


def test_ht_presence_is_nan_on_zero_width() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_presence

    assert np.isnan(ht_presence(esd_min=5.0, esd_max=5.0, ht_esd_xmin=5.0))


def test_ht_scale_is_max_times_lambda() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_scale

    assert ht_scale(esd_max=8.0, ht_lambda=0.1) == pytest.approx(0.8)


def test_ht_scale_propagates_nan() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_scale

    # A degenerate fit returns lambda = nan (BAD_RESULT); an inf-corrupted
    # checkpoint gives esd_max = nan. Either operand carries the nan through
    # instead of swallowing it into a plausible number.
    assert np.isnan(ht_scale(esd_max=8.0, ht_lambda=float("nan")))
    assert np.isnan(ht_scale(esd_max=float("nan"), ht_lambda=0.1))


def test_ht_presence_propagates_nan_esd_bounds() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_presence

    # An inf-corrupted checkpoint gives an all-nan spectrum (esd_min = esd_max
    # = nan); the width guard returns nan rather than dividing into a warning.
    nan = float("nan")
    assert np.isnan(ht_presence(esd_min=nan, esd_max=nan, ht_esd_xmin=3.0))


def test_ht_concentration_is_tail_size_fraction() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_concentration

    esd = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    # xmin = 2.0 -> {2.0, 3.0, 4.0} = 3 of 5 entries sit in the tail.
    assert ht_concentration(esd, 2.0) == pytest.approx(3.0 / 5.0)


def test_ht_concentration_propagates_nan_xmin() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_concentration

    # A failed / no-valid fit returns xmin = nan; the tail fraction must carry
    # the nan through instead of counting zero tail entries and reporting a
    # plausible 0.0.
    assert np.isnan(ht_concentration(np.array([1.0, 2.0, 3.0]), float("nan")))


def test_ht_concentration_propagates_nan_spectrum() -> None:
    from diffract.core.compute.kernels.heavy_tailed import ht_concentration

    # An inf-corrupted checkpoint yields an all-nan esd; the fraction propagates
    # nan rather than silently reporting no tail.
    assert np.isnan(ht_concentration(np.full(3, np.nan), 2.0))
