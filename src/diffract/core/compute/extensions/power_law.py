"""Accelerated heavy-tailed distribution fitting with taichi.

Implements the fitting procedure of Clauset, Shalizi & Newman (2009) for
power-law, truncated power-law and exponential tails: maximum-likelihood
parameter estimates at every xmin candidate, KS-optimal xmin selection,
and the semi-parametric bootstrap p-value of section 4.1. The scan over
xmin candidates and the bootstrap replicas are parallelized with taichi.

Input data is normalized internally by a power-of-two scale close to its
median (exact in floating point), so the parameter bounds below are
relative to the data scale; reported parameters are always in the
original units. The supported domain is pl_alpha in (1, 8] and
expon_lambda * scale >= 1e-4; tails outside it (e.g. a power law steeper
than alpha = 8) yield NaN parameters rather than a clamped estimate.

One documented simplification: bootstrap replicas are re-fitted at the
original xmin (per-replica xmin re-selection is skipped for tractability).
This biases p-values upward: the test has low power against misspecified
alternatives and should be used to confirm a plausible fit, not to select
between candidate distributions.

All randomness derives from a counter-based splitmix64 hash keyed by the
``seed`` constructor argument, so results are deterministic per call and
identical across machines and thread schedules.

The module owns the process-wide taichi runtime, initialized on import
with the CPU backend: Metal lacks the f64 arithmetic and the field
lifecycle this module needs, and the workload is too branchy and too
small for CUDA to pay off.

Taichi fields are pooled process-wide by data size padded to a power of
two, and the kernels are module-level functions templated on those pooled
fields. A process that runs thousands of fits therefore compiles each
kernel a bounded number of times (per pool entry) instead of once per
fit, and repeated fits skip the per-instance compilation cost entirely.
:meth:`Fit.close` returns the borrowed fields to the pool; a closed
instance raises RuntimeError. Up to MAX_MAIN_POOLS_PER_SIZE concurrent
same-bucket fits share a bucket's entries; the large bootstrap pools are
capped at MAX_BOOTSTRAP_POOLS (least recently used ones are released),
and sizes above MAX_POOLED_SIZE / MAX_POOLED_BOOTSTRAP_SIZE are never
pooled. Outside those limits a fit still works but uses private fields:
it recompiles its kernels (~0.2 s) and taichi retains the compiled
artifacts (~2 MB) for the process lifetime.
"""

import math
import threading
from typing import Any, Literal

import numpy as np

import diffract.core.utils.imports as import_utils

ti = import_utils.require("taichi")

ti.init(arch=ti.cpu)

# params bounds
MIN_ALPHA = 1.0
MAX_ALPHA = 8.0  # empirically determined
MIN_LAMBDA = 1e-4  # empirically determined
MIN_KS_DISTANCE = 1e-4  # empirically determined

# optimizer
OPTIM_N_ITERS = 1000
OBJ_FUNC_THRESHOLD = 1e-6
BOUNDS_PENALTY = 1e30
SIMPLEX_SIZE = 3  # 2 parameters + 1

# other constants
P_VALUE_EPS = 1e-2
P_VALUE_THRESHOLD = 0.1  # from Clauset et al.
MIN_TAIL_SIZE = 50  # from Clauset et al.
MIN_BOOTSTRAP_TAIL_SIZE = 2
MAX_REJECTION_ATTEMPTS = 4096
F32_SAMPLE_CAP = 3.0e38  # just under f32 max
MAX_P_VALUE_ELEMENTS = 500_000_000  # ~2 GB of f32 bootstrap samples

# field pooling
MIN_POOL_SIZE = 64
MAX_POOLED_SIZE = 65_536
MAX_MAIN_POOLS_PER_SIZE = 4
MAX_BOOTSTRAP_POOLS = 2
MAX_POOLED_BOOTSTRAP_SIZE = 16_384

BAD_RESULT = -1

POWER_LAW = 0
TRUNCATED_POWER_LAW = 1
EXPONENTIAL = 2
_DISTRIBUTION_CODES = {
    "power_law": POWER_LAW,
    "truncated_power_law": TRUNCATED_POWER_LAW,
    "exponential": EXPONENTIAL,
}

DistributionParams = ti.types.struct(xmin=ti.f32, pl_alpha=ti.f32, expon_lambda=ti.f32)


@ti.func
def _uniform01(seed: ti.u32, a: ti.u32, b: ti.u32, c: ti.u32) -> ti.f32:
    """Deterministic counter-based uniform in [0, 1) (splitmix64 hash)."""
    x = (
        ti.cast(seed, ti.u64)
        + ti.cast(a, ti.u64) * ti.u64(0x9E3779B97F4A7C15)
        + ti.cast(b, ti.u64) * ti.u64(0xBF58476D1CE4E5B9)
        + ti.cast(c, ti.u64) * ti.u64(0x94D049BB133111EB)
    )
    x ^= x >> 30
    x *= ti.u64(0xBF58476D1CE4E5B9)
    x ^= x >> 27
    x *= ti.u64(0x94D049BB133111EB)
    x ^= x >> 31
    # Top 24 bits fill the f32 mantissa exactly, so the result is < 1.0.
    return ti.cast(x >> 40, ti.f32) * ti.f32(1.0 / 16777216.0)


@ti.func
def _rvs(  # noqa: ANN202
    distribution: ti.template(),
    distribution_params: DistributionParams,
    seed: ti.u32,
    iteration: ti.u32,
    item: ti.u32,
):
    """Draw one tail sample via inverse-CDF / rejection sampling.

    Power-law draws are computed in f64 and capped just below the f32
    maximum, so alpha ~ 1 tails stay finite. The TPL rejection sampler
    uses whichever proposal accepts more often and falls back to xmin
    on attempt exhaustion (reachable only for alpha ~ 1 with
    lambda*xmin ~ alpha-1).
    """
    xmin = distribution_params.xmin
    pl_alpha = distribution_params.pl_alpha
    expon_lambda = distribution_params.expon_lambda

    probability = _uniform01(seed, iteration, item, 2)
    result = ti.cast(BAD_RESULT, ti.f32)

    if ti.static(distribution == POWER_LAW):
        value = ti.cast(xmin, ti.f64) * ti.cast(1 - probability, ti.f64) ** ti.cast(
            -1 / (pl_alpha - 1), ti.f64
        )
        result = ti.cast(ti.min(value, ti.f64(F32_SAMPLE_CAP)), ti.f32)
    elif ti.static(distribution == TRUNCATED_POWER_LAW):
        # Rejection sampling. Acceptance rates are s**alpha * G for
        # the exponential proposal and (alpha-1) * s**(alpha-1) * G
        # for the power-law one (s = lambda*xmin, common factor G),
        # so the better proposal flips at s = alpha - 1.
        use_pl_proposal = expon_lambda * xmin < pl_alpha - 1
        attempt = ti.u32(3)
        accepted = False
        result = xmin
        for _ in range(MAX_REJECTION_ATTEMPTS):
            candidate = xmin - (1 / expon_lambda) * ti.log(1 - probability)
            acceptance = ti.f32(0.0)
            if use_pl_proposal:
                value = ti.cast(xmin, ti.f64) * ti.cast(
                    1 - probability, ti.f64
                ) ** ti.cast(-1 / (pl_alpha - 1), ti.f64)
                candidate = ti.cast(ti.min(value, ti.f64(F32_SAMPLE_CAP)), ti.f32)
                acceptance = ti.exp(-expon_lambda * (candidate - xmin))
            else:
                acceptance = (candidate / xmin) ** (-pl_alpha)
            if _uniform01(seed, iteration, item, attempt) < acceptance:
                result = candidate
                accepted = True
            probability = _uniform01(seed, iteration, item, attempt + 1)
            attempt += 2
            if accepted:
                break
    else:
        result = xmin - (1 / expon_lambda) * ti.log(1 - probability)

    return result


@ti.func
def _cdf(  # noqa: ANN202
    distribution: ti.template(),
    value: float,
    distribution_params: DistributionParams,
):
    """Conditional model CDF on the tail (x >= xmin), in shifted form."""
    xmin = distribution_params.xmin
    pl_alpha = distribution_params.pl_alpha
    expon_lambda = distribution_params.expon_lambda

    result = ti.cast(BAD_RESULT, ti.f32)
    if ti.static(distribution == POWER_LAW):
        if pl_alpha < MIN_ALPHA or pl_alpha > MAX_ALPHA:
            result = BAD_RESULT
        else:
            result = 1 - (value / xmin) ** (1 - pl_alpha)
    elif ti.static(distribution == TRUNCATED_POWER_LAW):
        if pl_alpha < MIN_ALPHA or pl_alpha > MAX_ALPHA or expon_lambda < MIN_LAMBDA:
            result = BAD_RESULT
        else:
            shift = expon_lambda * xmin
            survival_xmin = incomplete_gamma(1 - pl_alpha, shift, shift)
            survival_value = incomplete_gamma(1 - pl_alpha, expon_lambda * value, shift)
            result = ti.cast(1.0 - survival_value / survival_xmin, ti.f32)
    elif expon_lambda < MIN_LAMBDA:
        result = BAD_RESULT
    else:
        result = 1 - ti.exp(-expon_lambda * (value - xmin))

    return result


@ti.func
def _ks_distance(  # noqa: ANN202
    distribution: ti.template(),
    src: ti.template(),
    row: ti.i32,
    data_size: ti.i32,
    distribution_params: DistributionParams,
    min_tail_size: ti.i32,
):
    """One-sided KS statistic on the tail (empirical grid i/n)."""
    xmin = distribution_params.xmin

    tail_size = 0
    for index in range(data_size):
        if src[row, index] < xmin:
            continue
        tail_size += 1

    bulk_size = data_size - tail_size

    ks_distance = ti.cast(0.0, ti.f32)
    if tail_size >= min_tail_size:
        for index in range(tail_size):
            value = src[row, bulk_size + index]

            empirical_cdf = index / tail_size
            model_cdf = _cdf(distribution, value, distribution_params)
            diff = abs(empirical_cdf - model_cdf)
            ks_distance = max(ks_distance, diff)

    return ks_distance


@ti.func
def _tpl_negative_log_likelihood(  # noqa: ANN202
    xmin,  # noqa: ANN001
    mean_log: ti.f32,
    mean_value: ti.f32,
    params_array,  # noqa: ANN001
):
    """Per-point negative log-likelihood of the truncated power law."""
    pl_alpha = params_array[0]
    expon_lambda = params_array[1]

    result = ti.cast(BOUNDS_PENALTY, ti.f64)
    if MIN_ALPHA < pl_alpha <= MAX_ALPHA and expon_lambda >= MIN_LAMBDA:
        shift = ti.cast(expon_lambda * xmin, ti.f64)
        normalization = incomplete_gamma(
            1 - pl_alpha, expon_lambda * xmin, expon_lambda * xmin
        )
        result = (
            (ti.cast(pl_alpha, ti.f64) - 1) * ti.log(ti.cast(expon_lambda, ti.f64))
            + ti.log(normalization)
            - shift
            + ti.cast(pl_alpha * mean_log, ti.f64)
            + ti.cast(expon_lambda * mean_value, ti.f64)
        )

    return result


@ti.func
def _nelder_mead_optimizer(  # noqa: ANN202
    guess_idx,  # noqa: ANN001
    xmin,  # noqa: ANN001
    mean_log: ti.f32,
    mean_value: ti.f32,
    initial_params,  # noqa: ANN001
    simplex: ti.template(),
    values: ti.template(),
):
    """Minimize the TPL negative log-likelihood over (alpha, lambda)."""
    # alias
    idx = guess_idx
    dimension = 2

    for dim in range(dimension + 1):
        simplex[idx, dim] = initial_params
        if dim > 0:
            simplex[idx, dim][dim - 1] *= 1.5

    for _iteration in range(OPTIM_N_ITERS):
        for dim in range(dimension + 1):
            values[idx, dim] = _tpl_negative_log_likelihood(
                xmin, mean_log, mean_value, simplex[idx, dim]
            )

        best_idx = 0
        worst_idx = 0
        for dim in range(dimension + 1):
            if values[idx, dim] < values[idx, best_idx]:
                best_idx = dim
            if values[idx, dim] > values[idx, worst_idx]:
                worst_idx = dim
        second_worst_idx = best_idx
        for dim in range(dimension + 1):
            if dim != worst_idx and values[idx, dim] > values[idx, second_worst_idx]:
                second_worst_idx = dim

        centroid = ti.Vector([0.0, 0.0])
        for dim in range(dimension + 1):
            if dim != worst_idx:
                centroid += simplex[idx, dim]
        centroid /= dimension

        # reflection
        reflected_point = centroid + (centroid - simplex[idx, worst_idx])
        reflected_value = _tpl_negative_log_likelihood(
            xmin, mean_log, mean_value, reflected_point
        )

        if reflected_value < values[idx, best_idx]:
            # expansion
            expanded_point = centroid + 2.0 * (centroid - simplex[idx, worst_idx])
            expanded_value = _tpl_negative_log_likelihood(
                xmin, mean_log, mean_value, expanded_point
            )
            if expanded_value < reflected_value:
                simplex[idx, worst_idx] = expanded_point
                values[idx, worst_idx] = expanded_value
            else:
                simplex[idx, worst_idx] = reflected_point
                values[idx, worst_idx] = reflected_value
        elif reflected_value < values[idx, second_worst_idx]:
            simplex[idx, worst_idx] = reflected_point
            values[idx, worst_idx] = reflected_value
        else:
            # contraction
            if reflected_value < values[idx, worst_idx]:
                simplex[idx, worst_idx] = reflected_point
                values[idx, worst_idx] = reflected_value
            contracted_point = centroid + 0.5 * (simplex[idx, worst_idx] - centroid)
            contracted_value = _tpl_negative_log_likelihood(
                xmin, mean_log, mean_value, contracted_point
            )
            if contracted_value < values[idx, worst_idx]:
                simplex[idx, worst_idx] = contracted_point
                values[idx, worst_idx] = contracted_value
            else:
                # reduction
                for dim in range(dimension + 1):
                    if dim != best_idx:
                        simplex[idx, dim] = 0.5 * (
                            simplex[idx, dim] + simplex[idx, best_idx]
                        )
                        values[idx, dim] = _tpl_negative_log_likelihood(
                            xmin, mean_log, mean_value, simplex[idx, dim]
                        )

        max_value = ti.cast(-np.inf, ti.f64)
        min_value = ti.cast(np.inf, ti.f64)
        for dim in range(dimension + 1):
            max_value = ti.max(max_value, values[idx, dim])
            min_value = ti.min(min_value, values[idx, dim])

        if max_value - min_value < OBJ_FUNC_THRESHOLD:
            break

    best_idx = 0
    for dim in range(dimension + 1):
        if values[idx, dim] < values[idx, best_idx]:
            best_idx = dim

    return simplex[idx, best_idx]


@ti.func
def _estimate_for_specific_xmin(  # noqa: ANN202
    distribution: ti.template(),
    guess_index,  # noqa: ANN001
    src: ti.template(),
    row: ti.i32,
    data_size: ti.i32,
    xmin: float,
    simplex: ti.template(),
    values: ti.template(),
):
    """Maximum-likelihood parameter estimates for a fixed xmin."""
    tail_size = 0
    tail_log_sum = ti.cast(0.0, ti.f32)
    tail_mean = ti.cast(0.0, ti.f32)

    for index in range(data_size):
        value = src[row, index]

        if value < xmin:
            continue

        tail_size += 1
        tail_log_sum += ti.log(value / xmin)
        tail_mean += value
    tail_mean /= ti.max(tail_size, 1)

    # Closed-form conditional MLEs (Clauset et al. 2009).
    mle_pl_alpha = 1 + tail_size / ti.max(tail_log_sum, 1e-12)
    mle_expon_lambda = 1 / ti.max(tail_mean - xmin, 1e-9)

    distribution_params = DistributionParams()
    if ti.static(distribution == POWER_LAW):
        distribution_params = DistributionParams(
            xmin=xmin,
            pl_alpha=mle_pl_alpha,
            expon_lambda=np.nan,
        )
    elif ti.static(distribution == EXPONENTIAL):
        distribution_params = DistributionParams(
            xmin=xmin,
            pl_alpha=np.nan,
            expon_lambda=mle_expon_lambda,
        )
    else:
        # TPL has no closed-form MLE; minimize the negative
        # log-likelihood over (alpha, lambda) with Nelder-Mead. The
        # start point is clamped so the whole initial simplex (x1.5
        # vertex perturbations) lies inside the parameter bounds;
        # otherwise the optimizer would start on the constant
        # BOUNDS_PENALTY plateau and stop immediately.
        mean_log = tail_log_sum / ti.max(tail_size, 1) + ti.log(xmin)
        initial_alpha = ti.min(ti.max(mle_pl_alpha, 1.0 + 1e-4), MAX_ALPHA / 1.5)
        initial_lambda = ti.max(mle_expon_lambda, 2.0 * MIN_LAMBDA)
        solution = _nelder_mead_optimizer(
            guess_index,
            xmin,
            mean_log,
            tail_mean,
            ti.Vector([initial_alpha, initial_lambda], dt=ti.f32),
            simplex,
            values,
        )

        distribution_params = DistributionParams(
            xmin=xmin,
            pl_alpha=solution[0],
            expon_lambda=solution[1],
        )

    return distribution_params


@ti.kernel
def _iter_through_xmins(  # noqa: ANN202 - taichi kernels reject the None annotation
    distribution: ti.template(),
    samples: ti.template(),
    pl_alphas: ti.template(),
    expon_lambdas: ti.template(),
    ks_dists: ti.template(),
    optim_simplex: ti.template(),
    optim_values: ti.template(),
    data_size: ti.i32,
):
    """Estimate parameters and KS distance for every xmin candidate."""
    for index in range(data_size):
        xmin = samples[0, index]

        distribution_params = _estimate_for_specific_xmin(
            distribution,
            index,
            samples,
            0,
            data_size,
            xmin,
            optim_simplex,
            optim_values,
        )
        pl_alphas[index] = distribution_params.pl_alpha
        expon_lambdas[index] = distribution_params.expon_lambda
        ks_dists[index] = _ks_distance(
            distribution, samples, 0, data_size, distribution_params, MIN_TAIL_SIZE
        )


@ti.kernel
def _generate_synth_datasets(  # noqa: ANN202 - taichi kernels reject the None annotation
    distribution: ti.template(),
    samples: ti.template(),
    synth_datasets: ti.template(),
    distribution_params: DistributionParams,
    seed: ti.u32,
    data_size: ti.i32,
    iters: ti.i32,
    tail_size: ti.i32,
):
    """Draw bootstrap datasets (body resample + fitted-tail draws)."""
    # Clauset et al. semi-parametric bootstrap: each point comes from
    # the fitted tail with probability tail_size/n, otherwise it is a
    # bootstrap resample of the empirical body (values below xmin).
    tail_probability = tail_size / data_size
    bulk_size = data_size - tail_size

    for iteration, item in ti.ndrange(iters, data_size):
        coin = _uniform01(seed, iteration, item, 0)
        if coin < tail_probability or bulk_size == 0:
            synth_datasets[iteration, item] = _rvs(
                distribution, distribution_params, seed, iteration, item
            )
        else:
            pick_probability = _uniform01(seed, iteration, item, 1)
            pick = ti.min(ti.cast(pick_probability * bulk_size, ti.i32), bulk_size - 1)
            synth_datasets[iteration, item] = samples[0, pick]


@ti.kernel
def _compute_p_value_dists(  # noqa: ANN202 - taichi kernels reject the None annotation
    distribution: ti.template(),
    synth_datasets: ti.template(),
    p_value_test_dists: ti.template(),
    synth_optim_simplex: ti.template(),
    synth_optim_values: ti.template(),
    distribution_params: DistributionParams,
    data_size: ti.i32,
    iters: ti.i32,
):
    """Re-fit each bootstrap dataset and record its KS distance."""
    xmin = distribution_params.xmin

    for index in range(iters):
        other_params = _estimate_for_specific_xmin(
            distribution,
            index,
            synth_datasets,
            index,
            data_size,
            xmin,
            synth_optim_simplex,
            synth_optim_values,
        )
        other_ks_distance = _ks_distance(
            distribution,
            synth_datasets,
            index,
            data_size,
            other_params,
            MIN_BOOTSTRAP_TAIL_SIZE,
        )

        p_value_test_dists[index] = other_ks_distance


# region field pools


class _MainFieldPool:
    """Fit-scan taichi fields for one padded data size.

    Pool entries are kept for the lifetime of the process, so the kernels
    templated on them are compiled once per entry. Retention is ~64 bytes
    per padded element (a few hundred KB at typical bucket sizes); at
    most MAX_MAIN_POOLS_PER_SIZE entries exist per bucket, and sizes
    above MAX_POOLED_SIZE are never pooled.
    """

    def __init__(self, size: int):
        self.size = size
        self.in_use = False
        builder = ti.FieldsBuilder()
        self.samples = ti.field(dtype=ti.f32)
        self.pl_alphas = ti.field(dtype=ti.f32)
        self.expon_lambdas = ti.field(dtype=ti.f32)
        self.ks_dists = ti.field(dtype=ti.f32)
        self.optim_simplex = ti.Vector.field(2, dtype=ti.f32)
        self.optim_values = ti.field(dtype=ti.f64)
        builder.dense(ti.ij, (1, size)).place(self.samples)
        builder.dense(ti.i, size).place(
            self.pl_alphas, self.expon_lambdas, self.ks_dists
        )
        builder.dense(ti.ij, (size, SIMPLEX_SIZE)).place(
            self.optim_simplex, self.optim_values
        )
        self._tree = builder.finalize()

    def destroy(self) -> None:
        self._tree.destroy()


class _BootstrapFieldPool:
    """Bootstrap taichi fields for one padded data size.

    These hold iters x size f32 samples (tens of MB for large fits), so
    only MAX_BOOTSTRAP_POOLS of them are kept; least recently used ones
    are destroyed.
    """

    def __init__(self, size: int, iters: int):
        self.size = size
        self.iters = iters
        self.in_use = False
        builder = ti.FieldsBuilder()
        self.p_value_test_dists = ti.field(dtype=ti.f32)
        self.synth_optim_simplex = ti.Vector.field(2, dtype=ti.f32)
        self.synth_optim_values = ti.field(dtype=ti.f64)
        self.synth_datasets = ti.field(dtype=ti.f32)
        builder.dense(ti.i, iters).place(self.p_value_test_dists)
        builder.dense(ti.ij, (iters, SIMPLEX_SIZE)).place(
            self.synth_optim_simplex, self.synth_optim_values
        )
        builder.dense(ti.ij, (iters, size)).place(self.synth_datasets)
        self._tree = builder.finalize()

    def destroy(self) -> None:
        self._tree.destroy()


_MAIN_POOLS: dict[int, list[_MainFieldPool]] = {}
_BOOTSTRAP_POOLS: dict[int, _BootstrapFieldPool] = {}
_POOLS_LOCK = threading.Lock()


def _padded_size(data_size: int) -> int:
    return max(MIN_POOL_SIZE, 2 ** math.ceil(math.log2(data_size)))


def _acquire_main_pool(size: int) -> tuple[_MainFieldPool, bool]:
    # A non-pooled instance (oversized bucket, or every entry busy) gets
    # private fields; they are destroyed on close, but the kernels
    # compiled for them stay cached by taichi (~2 MB per instance).
    if size > MAX_POOLED_SIZE:
        return _MainFieldPool(size), False
    with _POOLS_LOCK:
        entries = _MAIN_POOLS.setdefault(size, [])
        for entry in entries:
            if not entry.in_use:
                entry.in_use = True
                return entry, True
        if len(entries) < MAX_MAIN_POOLS_PER_SIZE:
            entry = _MainFieldPool(size)
            entries.append(entry)
            entry.in_use = True
            return entry, True
        return _MainFieldPool(size), False


def _acquire_bootstrap_pool(size: int, iters: int) -> tuple[_BootstrapFieldPool, bool]:
    if size > MAX_POOLED_BOOTSTRAP_SIZE:
        return _BootstrapFieldPool(size, iters), False
    with _POOLS_LOCK:
        pool = _BOOTSTRAP_POOLS.get(size)
        if pool is not None and pool.in_use:
            return _BootstrapFieldPool(size, iters), False
        if pool is not None and pool.iters < iters:
            _BOOTSTRAP_POOLS.pop(size).destroy()
            pool = None
        if pool is None:
            while len(_BOOTSTRAP_POOLS) >= MAX_BOOTSTRAP_POOLS:
                evictable = next(
                    (key for key, p in _BOOTSTRAP_POOLS.items() if not p.in_use),
                    None,
                )
                if evictable is None:
                    return _BootstrapFieldPool(size, iters), False
                _BOOTSTRAP_POOLS.pop(evictable).destroy()
            pool = _BootstrapFieldPool(size, iters)
        else:
            _BOOTSTRAP_POOLS.pop(size)
        _BOOTSTRAP_POOLS[size] = pool  # (re-)insert as most recently used
        pool.in_use = True
        return pool, True


def _release_pool(pool, pooled: bool) -> None:  # noqa: ANN001
    with _POOLS_LOCK:
        if pooled:
            pool.in_use = False
        else:
            pool.destroy()


# endregion field pools


class Fit:
    """Taichi-based heavy-tailed distribution fitter.

    Data is rescaled internally by a power of two near its median (exact
    in floating point), which makes the parameter bounds scale-invariant;
    fitted and injected parameters are always in the original data units.
    """

    def __init__(
        self,
        data,  # noqa: ANN001
        distribution: Literal["power_law", "truncated_power_law", "exponential"],
        *,
        seed: int = 42,
    ):
        data = np.asarray(data)
        if data.ndim != 1:
            msg = f"Expected 1-D data, got shape {data.shape}"
            raise ValueError(msg)

        if data.size == 0:
            msg = "Cannot fit an empty dataset"
            raise ValueError(msg)

        if not np.all(np.isfinite(data)):
            msg = "Data must contain only finite values"
            raise ValueError(msg)

        if distribution not in _DISTRIBUTION_CODES:
            msg = f"Unknown distribution {distribution!r}"
            raise ValueError(msg)

        if not 0 <= seed < 2**32:
            msg = f"seed must fit in an unsigned 32-bit integer, got {seed}"
            raise ValueError(msg)

        median = float(np.median(data))
        self._scale = 2.0 ** round(math.log2(median)) if median > 0 else 1.0

        self.data_size = data.size
        self.seed = seed
        self.distribution = distribution
        self._distribution_code = _DISTRIBUTION_CODES[distribution]

        self.p_value_test_iters = round(0.25 * (P_VALUE_EPS ** (-2)))

        # The pool region beyond data_size is padding; every kernel loop
        # is bounded by data_size, so its contents never matter.
        buffer = np.full((1, _padded_size(data.size)), np.inf, dtype=np.float32)
        buffer[0, : self.data_size] = np.sort((data / self._scale).astype(np.float32))

        self._main_pool, self._main_pooled = _acquire_main_pool(buffer.shape[1])
        self._bootstrap_pool = None
        self._bootstrap_pooled = False
        self._main_pool.samples.from_numpy(buffer)

        # must be available during p_value test
        self.params = None
        self.tail_size = None

    def close(self) -> None:
        """Return the taichi fields borrowed by this instance to the pool.

        Idempotent; a closed instance raises RuntimeError from
        :meth:`fit_params` and :meth:`p_value_test`.
        """
        if self._bootstrap_pool is not None:
            _release_pool(self._bootstrap_pool, self._bootstrap_pooled)
            self._bootstrap_pool = None
        if self._main_pool is not None:
            _release_pool(self._main_pool, self._main_pooled)
            self._main_pool = None

    def __del__(self) -> None:  # noqa: D105
        # A bare try/except: module globals (contextlib.suppress included)
        # may already be torn down when the destructor runs at shutdown.
        try:  # noqa: SIM105
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass

    def fit_params(self):  # noqa: ANN201
        """Fit the distribution, selecting xmin by global KS minimum.

        Returned parameters are in the original data units.
        """
        if self._main_pool is None:
            msg = "Fit is closed"
            raise RuntimeError(msg)

        if self.params is None:
            pool = self._main_pool
            _iter_through_xmins(
                self._distribution_code,
                pool.samples,
                pool.pl_alphas,
                pool.expon_lambdas,
                pool.ks_dists,
                pool.optim_simplex,
                pool.optim_values,
                self.data_size,
            )

            ks_dists = pool.ks_dists.to_numpy()[: self.data_size]
            # A genuine KS statistic lies in (0, 1]; values outside come
            # from degenerate candidates (out-of-bounds parameters).
            mask = (ks_dists > MIN_KS_DISTANCE) & (ks_dists <= 1.0)

            argmin = int(np.argmin(ks_dists[mask])) if mask.any() else BAD_RESULT

            if argmin == BAD_RESULT:
                xmin = np.nan
                pl_alpha = np.nan
                expon_lambda = np.nan
                ks_distance = np.nan
                self.tail_size = 0
            else:
                data = pool.samples.to_numpy()[0, : self.data_size]
                xmin = data[mask][argmin]
                pl_alpha = pool.pl_alphas.to_numpy()[: self.data_size][mask][argmin]
                expon_lambda = pool.expon_lambdas.to_numpy()[: self.data_size][mask][
                    argmin
                ]
                ks_distance = ks_dists[mask][argmin]
                self.tail_size = int(np.sum(data >= xmin))

            self.params = {
                "xmin": float(xmin * self._scale),
                "pl_alpha": float(pl_alpha),
                "expon_lambda": float(expon_lambda / self._scale),
                "ks_distance": float(ks_distance),
                "tail_size": self.tail_size,
            }

        return self.params

    def set_params(self, **kwargs: Any) -> None:
        """Inject externally estimated parameters (e.g. from powerlaw).

        Values are in the original data units; the previous parameter set
        is replaced atomically (and kept intact on invalid input).
        """
        allowed = ("xmin", "pl_alpha", "expon_lambda", "ks_distance", "tail_size")
        validated: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in allowed:
                msg = f"Unknown fit parameter {key!r}"
                raise ValueError(msg)
            try:
                validated[key] = float(value)
            except (TypeError, ValueError) as error:
                msg = f"Fit parameter {key!r} must be a real number, got {value!r}"
                raise ValueError(msg) from error

        tail_size = validated.get("tail_size")
        if tail_size is not None:
            if not tail_size.is_integer() or not 0 <= tail_size <= self.data_size:
                msg = (
                    f"tail_size must be an integer in [0, {self.data_size}], "
                    f"got {kwargs['tail_size']!r}"
                )
                raise ValueError(msg)
            validated["tail_size"] = int(tail_size)

        xmin = validated.get("xmin")
        if xmin is not None and not (math.isnan(xmin) or xmin > 0):
            msg = f"xmin must be positive, got {kwargs['xmin']!r}"
            raise ValueError(msg)

        self.params = validated

    def _required_params_finite(self) -> bool:
        xmin = self.params.get("xmin", np.nan)
        pl_alpha = self.params.get("pl_alpha", np.nan)
        expon_lambda = self.params.get("expon_lambda", np.nan)

        if not np.isfinite(xmin):
            return False
        if self.distribution in (
            "power_law",
            "truncated_power_law",
        ) and not np.isfinite(pl_alpha):
            return False
        return not (
            self.distribution in ("truncated_power_law", "exponential")
            and not np.isfinite(expon_lambda)
        )

    def p_value_test(self):  # noqa: ANN201
        """Semi-parametric bootstrap plausibility test (Clauset et al. 4.1).

        Returns (plausible, p_value); p_value is NaN when the tail is too
        small (< MIN_TAIL_SIZE) or the fit produced no valid parameters.
        """
        if self._main_pool is None:
            msg = "Fit is closed"
            raise RuntimeError(msg)

        if not self.params:
            return False, float("nan")

        self.tail_size = int(self.params.get("tail_size", 0))
        ks_distance = self.params.get("ks_distance", np.nan)

        if (
            not self._required_params_finite()
            or not np.isfinite(ks_distance)
            or self.tail_size < MIN_TAIL_SIZE
        ):
            return False, float("nan")

        if self.p_value_test_iters * self._main_pool.size > MAX_P_VALUE_ELEMENTS:
            msg = (
                f"p_value_test would allocate {self.p_value_test_iters} x "
                f"{self._main_pool.size} float32 bootstrap samples; subsample "
                "the data first"
            )
            raise ValueError(msg)

        distribution_params = DistributionParams(
            xmin=self.params.get("xmin", np.nan) / self._scale,
            pl_alpha=self.params.get("pl_alpha", np.nan),
            expon_lambda=self.params.get("expon_lambda", np.nan) * self._scale,
        )

        if self._bootstrap_pool is None:
            self._bootstrap_pool, self._bootstrap_pooled = _acquire_bootstrap_pool(
                self._main_pool.size, self.p_value_test_iters
            )
        boot = self._bootstrap_pool

        _generate_synth_datasets(
            self._distribution_code,
            self._main_pool.samples,
            boot.synth_datasets,
            distribution_params,
            self.seed,
            self.data_size,
            self.p_value_test_iters,
            self.tail_size,
        )

        # Bootstrap resampling perturbs both body and tail; taichi has no
        # library sort, numpy row-wise sort is fast and exact. Only the
        # first data_size columns hold samples; the padding is untouched.
        synth = boot.synth_datasets.to_numpy()
        synth[: self.p_value_test_iters, : self.data_size].sort(axis=1)
        boot.synth_datasets.from_numpy(synth)

        _compute_p_value_dists(
            self._distribution_code,
            boot.synth_datasets,
            boot.p_value_test_dists,
            boot.synth_optim_simplex,
            boot.synth_optim_values,
            distribution_params,
            self.data_size,
            self.p_value_test_iters,
        )

        dists = boot.p_value_test_dists.to_numpy()[: self.p_value_test_iters]
        p_value = np.sum(dists > ks_distance) / dists.size
        result = p_value > P_VALUE_THRESHOLD

        return result, float(p_value)


# region utils

NUM_GAUSS_NODES = 16

gauss_nodes = ti.field(ti.f64, shape=NUM_GAUSS_NODES)
gauss_nodes.from_numpy(
    np.array(
        [
            0.0052995325041750337,
            0.027712488463383700,
            0.067184398806084122,
            0.12229779582249845,
            0.19106187779867811,
            0.27099161117138635,
            0.35919822461037054,
            0.45249374508118123,
            0.54750625491881877,
            0.64080177538962946,
            0.72900838882861365,
            0.80893812220132189,
            0.87770220417750155,
            0.93281560119391588,
            0.97228751153661630,
            0.99470046749582497,
        ]
    ).astype(np.float64)
)

gauss_weights = ti.field(ti.f64, shape=NUM_GAUSS_NODES)
gauss_weights.from_numpy(
    np.array(
        [
            0.013576229705877048,
            0.031126761969323946,
            0.047579255841246393,
            0.062314485627766938,
            0.074797994408288368,
            0.084578259697501267,
            0.091301707522461792,
            0.094725305227534251,
            0.094725305227534251,
            0.091301707522461792,
            0.084578259697501267,
            0.074797994408288368,
            0.062314485627766938,
            0.047579255841246393,
            0.031126761969323946,
            0.013576229705877048,
        ]
    ).astype(np.float64)
)

# Below this lower limit the integrand's t**(alpha-1) singularity defeats
# the mapped quadrature; the [lower, SPLIT) part is integrated analytically
# via the exp(-t) Taylor series instead.
GAMMA_SERIES_SPLIT = 0.5
GAMMA_SERIES_TERMS = 8
GAMMA_EXPONENT_EPS = 1e-6


@ti.func
def incomplete_gamma(alpha: ti.f32, lower_limit: ti.f32, shift: ti.f32) -> ti.f64:
    """Shifted upper incomplete gamma: integral of t**(alpha-1)*exp(shift-t).

    Equals exp(shift) * Gamma(alpha, lower_limit); the shift keeps the
    result representable when lower_limit is large (conditional CDFs only
    ever need the ratio of two values with a common shift). Computed in
    f64 via mapped Gauss-Legendre quadrature plus an analytic series for
    the near-singular region.
    """
    a = ti.cast(alpha, ti.f64)
    lower = ti.cast(lower_limit, ti.f64)
    shift64 = ti.cast(shift, ti.f64)
    start = ti.max(lower, ti.f64(GAMMA_SERIES_SPLIT))

    result = ti.cast(0.0, ti.f64)
    for node_index in range(NUM_GAUSS_NODES):
        transformed_node = start + (
            gauss_nodes[node_index] / (1.0 - gauss_nodes[node_index])
        )
        weight = gauss_weights[node_index] / ((1.0 - gauss_nodes[node_index]) ** 2)
        result += (
            weight * transformed_node ** (a - 1) * ti.exp(shift64 - transformed_node)
        )

    if lower < GAMMA_SERIES_SPLIT:
        # exp(shift) is bounded here: shift <= lower < SPLIT by construction.
        scale = ti.exp(shift64)
        coefficient = ti.cast(1.0, ti.f64)
        for k in range(GAMMA_SERIES_TERMS):
            if k > 0:
                coefficient *= -1.0 / k
            exponent = a + k
            term = ti.cast(0.0, ti.f64)
            if ti.abs(exponent) < GAMMA_EXPONENT_EPS:
                term = ti.log(start / lower)
            else:
                term = (start**exponent - lower**exponent) / exponent
            result += scale * coefficient * term

    return result


# endregion utils
