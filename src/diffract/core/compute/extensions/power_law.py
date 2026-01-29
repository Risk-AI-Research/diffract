from typing import Any, Literal

import numpy as np

import diffract.core.utils.imports as import_utils

ti = import_utils.require("taichi")
njit = import_utils.require("numba").njit

ti.init(arch=ti.cpu)

# params bounds
MIN_ALPHA = 1.0
MAX_ALPHA = 8.0  # empirically determined
MIN_LAMBDA = 1e-4  # empirically determined
MIN_KS_DISTANCE = 1e-4  # empirically determined

# optimizer
OPTIM_N_ITERS = 1000
OBJ_FUNC_THRESHOLD = 1e-6

# other constants
P_VALUE_EPS = 1e-2
P_VALUE_THRESHOLD = 0.1  # from Clauset et al.
MIN_TAIL_SIZE = 50  # from Clauset et al.

BAD_RESULT = -1

DistributionParams = ti.types.struct(xmin=ti.f32, pl_alpha=ti.f32, expon_lambda=ti.f32)


@ti.data_oriented
class Fit:
    """Taichi-based power law distribution fitter."""

    def __init__(
        self,
        data,  # noqa: ANN001
        distribution: Literal["power_law", "truncated_power_law", "exponential"],
    ):
        self.data_size = data.size

        self.data = ti.field(dtype=ti.f32, shape=(data.size))
        self.data.from_numpy(np.sort(data.astype(np.float32)))

        self.pl_alphas = ti.field(dtype=ti.f32, shape=(data.size))
        self.expon_lambdas = ti.field(dtype=ti.f32, shape=(data.size))

        self.ks_dists = ti.field(dtype=ti.f32, shape=(data.size))

        self.distribution = distribution

        match distribution:
            case "power_law":
                self.nelder_mead_optim_dimension = 0
            case "truncated_power_law":
                self.nelder_mead_optim_dimension = 2
            case "exponential":
                self.nelder_mead_optim_dimension = 1

        self.max_dim = 2
        self.optim_simplex = ti.Vector.field(
            self.max_dim, dtype=ti.f32, shape=(self.data_size, self.max_dim + 1)
        )
        self.optim_values = ti.field(
            dtype=ti.f32, shape=(self.data_size, self.max_dim + 1)
        )

        self.p_value_test_iters = round(0.25 * (P_VALUE_EPS ** (-2)))
        self.p_value_test_dists = ti.field(
            dtype=ti.f32, shape=(self.p_value_test_iters)
        )
        self.synth_optim_simplex = ti.Vector.field(
            self.max_dim,
            dtype=ti.f32,
            shape=(self.p_value_test_iters, self.max_dim + 1),
        )
        self.synth_optim_values = ti.field(
            dtype=ti.f32, shape=(self.p_value_test_iters, self.max_dim + 1)
        )
        self.synth_datasets = ti.field(
            dtype=ti.f32, shape=(self.p_value_test_iters, data.size)
        )

        # must be available during p_value test
        self.params = None
        self.tail_size = None
        self.synth_datasets_probas = None

    def fit_params(self):  # noqa: ANN201, D102
        if self.params is None:
            self.iter_through_xmins()

            ks_dists = self.ks_dists.to_numpy()
            mask = ks_dists > MIN_KS_DISTANCE
            argmin = find_left_min_in_cycle(ks_dists[mask], 1.0)

            if argmin == BAD_RESULT:
                xmin = np.nan
                pl_alpha = np.nan
                expon_lambda = np.nan
                ks_distance = np.nan
                self.tail_size = np.nan
            else:
                data = self.data.to_numpy()
                xmin = data[mask][argmin]
                pl_alpha = self.pl_alphas.to_numpy()[mask][argmin]
                expon_lambda = self.expon_lambdas.to_numpy()[mask][argmin]
                ks_distance = self.ks_dists.to_numpy()[mask][argmin]
                self.tail_size = np.sum(data >= xmin)

            self.params = {
                "xmin": xmin,
                "pl_alpha": pl_alpha,
                "expon_lambda": expon_lambda,
                "ks_distance": ks_distance,
                "tail_size": self.tail_size,
            }

        return self.params

    def set_params(self, **kwargs: Any) -> None:  # noqa: D102
        if self.params is None:
            self.params = {}

        for key, value in kwargs.items():
            if key not in (
                "xmin",
                "pl_alpha",
                "expon_lambda",
                "ks_distance",
                "tail_size",
            ):
                self.params = None
                raise ValueError

            self.params[key] = value

    def p_value_test(self):  # noqa: ANN201, D102
        self.tail_size = self.params.get("tail_size", np.nan)

        ks_distance = self.params.get("ks_distance", np.nan)
        distribution_params = DistributionParams(
            xmin=self.params.get("xmin", np.nan),
            pl_alpha=self.params.get("pl_alpha", np.nan),
            expon_lambda=self.params.get("expon_lambda", np.nan),
        )

        if np.isnan(distribution_params.xmin):
            return False, BAD_RESULT

        self.generate_synth_datasets(distribution_params)
        self.sort_synth_tails()
        self.compute_p_value_dists(distribution_params)

        dists = self.p_value_test_dists.to_numpy()
        p_value = np.sum(dists > ks_distance) / dists.size
        result = p_value > P_VALUE_THRESHOLD

        return result, p_value

    @ti.func
    def rvs(self, distribution_params: DistributionParams):  # noqa: ANN201, D102
        xmin = distribution_params.xmin
        pl_alpha = distribution_params.pl_alpha
        expon_lambda = distribution_params.expon_lambda

        result = ti.cast(BAD_RESULT, ti.f32)

        probability = ti.random()
        if self.distribution == "power_law":
            result = xmin * (1 - probability) ** (-1 / (pl_alpha - 1))
        elif self.distribution == "truncated_power_law":
            result = xmin - (1 / expon_lambda) * ti.log(1 - probability)
            while True:
                candidate = xmin - (1 / expon_lambda) * ti.log(1 - probability)
                validation_proba = (candidate / xmin) ** (-pl_alpha)
                if ti.random() < validation_proba:
                    result = candidate
                    break
                probability = ti.random()
        elif self.distribution == "exponential":
            result = xmin - (1 / expon_lambda) * ti.log(1 - probability)

        return result

    @ti.func
    def cdf(self, value: float, distribution_params: DistributionParams) -> float:  # noqa: D102
        xmin = distribution_params.xmin
        pl_alpha = distribution_params.pl_alpha
        expon_lambda = distribution_params.expon_lambda

        result = ti.cast(BAD_RESULT, ti.f32)
        if self.distribution == "power_law":
            if pl_alpha < MIN_ALPHA or pl_alpha > MAX_ALPHA:
                result = BAD_RESULT
            else:
                result = 1 - (value / xmin) ** (1 - pl_alpha)
        elif self.distribution == "truncated_power_law":
            if (
                pl_alpha < MIN_ALPHA
                or pl_alpha > MAX_ALPHA
                or expon_lambda < MIN_LAMBDA
            ):
                result = BAD_RESULT
            else:
                cdf_xmin = ti.cast(
                    incomplete_gamma(1 - pl_alpha, expon_lambda * xmin), ti.f32
                )
                cdf_xmin /= expon_lambda ** (1 - pl_alpha)
                cdf_xmin = 1 - cdf_xmin

                denom = ti.cast(
                    incomplete_gamma(1 - pl_alpha, expon_lambda * value), ti.f32
                )
                denom /= expon_lambda ** (1 - pl_alpha)
                denom = 1 - denom

                result = (denom - cdf_xmin) / (1 - cdf_xmin)
        elif expon_lambda < MIN_LAMBDA:
            result = BAD_RESULT
        else:
            cdf_xmin = 1 - ti.exp(-expon_lambda * xmin)
            denom = 1 - ti.exp(-expon_lambda * value)

            result = (denom - cdf_xmin) / (1 - cdf_xmin)

        return result

    @ti.func
    def ks_distance(  # noqa: ANN201, D102
        self,
        synth_data: bool,
        synth_data_idx: int,
        distribution_params: DistributionParams,
    ):
        xmin = distribution_params.xmin

        tail_size = 0
        for index in range(self.data_size):
            value = (
                self.synth_datasets[synth_data_idx, index]
                if synth_data
                else self.data[index]
            )

            if value < xmin:
                continue
            tail_size += 1

        bulk_size = self.data_size - tail_size

        ks_distance = ti.cast(0.0, ti.f32)
        if tail_size > MIN_TAIL_SIZE:
            for index in range(tail_size):
                idx = bulk_size + index
                value = (
                    self.synth_datasets[synth_data_idx, idx]
                    if synth_data
                    else self.data[idx]
                )

                empirical_cdf = index / tail_size
                model_cdf = self.cdf(value, distribution_params)
                diff = abs(empirical_cdf - model_cdf)
                ks_distance = max(ks_distance, diff)

        return ks_distance

    @ti.kernel
    def generate_synth_datasets(self, distribution_params: DistributionParams) -> None:  # noqa: D102
        xmin = distribution_params.xmin

        for p_value_test_iteration, item_idx in self.synth_datasets:
            if self.data[item_idx] < xmin:
                self.synth_datasets[p_value_test_iteration, item_idx] = self.data[
                    item_idx
                ]
            else:
                self.synth_datasets[p_value_test_iteration, item_idx] = self.rvs(
                    distribution_params
                )

    @ti.kernel
    def sort_synth_tails(self) -> None:  # noqa: D102
        bulk_size = self.data_size - self.tail_size

        for itr in range(self.p_value_test_iters):
            for outer in range(self.tail_size):
                for inner in range(self.tail_size - 1 - outer):
                    left, right = bulk_size + inner, bulk_size + inner + 1
                    if self.synth_datasets[itr, left] > self.synth_datasets[itr, right]:
                        (
                            self.synth_datasets[itr, left],
                            self.synth_datasets[itr, right],
                        ) = (
                            self.synth_datasets[itr, right],
                            self.synth_datasets[itr, left],
                        )

    @ti.kernel
    def compute_p_value_dists(self, distribution_params: DistributionParams) -> None:  # noqa: D102
        xmin = distribution_params.xmin

        for index in range(self.p_value_test_iters):
            other_params = self.estimate_for_specific_xmin(index, True, index, xmin)
            other_ks_distance = self.ks_distance(True, index, other_params)

            self.p_value_test_dists[index] = other_ks_distance

    @ti.kernel
    def iter_through_xmins(self) -> None:  # noqa: D102
        for index in self.data:
            xmin = self.data[index]

            distribution_params = self.estimate_for_specific_xmin(index, False, 0, xmin)
            self.pl_alphas[index] = distribution_params.pl_alpha
            self.expon_lambdas[index] = distribution_params.expon_lambda
            self.ks_dists[index] = self.ks_distance(False, 0, distribution_params)

    @ti.func
    def estimate_for_specific_xmin(  # noqa: ANN201, D102
        self,
        guess_index,  # noqa: ANN001
        synth_data: bool,
        synth_data_idx: int,
        xmin: float,
    ):
        tail_size = 0
        tail_log_sum = ti.cast(0.0, ti.f32)
        tail_mean = ti.cast(0.0, ti.f32)

        for index in range(self.data_size):
            value = (
                self.synth_datasets[synth_data_idx, index]
                if synth_data
                else self.data[index]
            )

            if value < xmin:
                continue

            tail_size += 1
            tail_log_sum += ti.log(value / xmin)
            tail_mean += value
        tail_mean /= tail_size

        initial_pl_alpha = 1 + tail_size / tail_log_sum
        initial_expon_lambda = 1 / tail_mean

        distribution_params = DistributionParams()
        if self.distribution == "power_law":
            distribution_params = DistributionParams(
                xmin=xmin,
                pl_alpha=initial_pl_alpha,
                expon_lambda=np.nan,
            )
        else:
            solution = self.nelder_mead_optimizer(
                synth_data,
                synth_data_idx,
                guess_index,
                xmin,
                ti.Vector([initial_pl_alpha, initial_expon_lambda], dt=ti.f32),
            )

            if self.distribution == "truncated_power_law":
                distribution_params = DistributionParams(
                    xmin=xmin,
                    pl_alpha=solution[0],
                    expon_lambda=solution[1],
                )
            else:
                distribution_params = DistributionParams(
                    xmin=xmin,
                    pl_alpha=np.nan,
                    expon_lambda=solution[1],
                )

        return distribution_params

    @ti.func
    def objective_function(  # noqa: ANN201, D102
        self,
        synth_data: bool,
        synth_data_idx: int,
        xmin,  # noqa: ANN001
        params_array,  # noqa: ANN001
    ):
        params = DistributionParams()
        if self.distribution == "truncated_power_law":
            params = DistributionParams(
                xmin=xmin,
                pl_alpha=params_array[0],
                expon_lambda=params_array[1],
            )
        else:
            params = DistributionParams(
                xmin=xmin,
                pl_alpha=np.nan,
                expon_lambda=params_array[1],
            )
        return self.ks_distance(synth_data, synth_data_idx, params)

    @ti.func
    def nelder_mead_optimizer(  # noqa: ANN201, D102
        self,
        synth_data: bool,
        synth_data_idx: int,
        guess_idx,  # noqa: ANN001
        xmin,  # noqa: ANN001
        initial_params,  # noqa: ANN001
    ):
        # alias
        idx = guess_idx

        simplex, values = ti.static(self.optim_simplex, self.optim_values)
        if synth_data:
            simplex, values = ti.static(
                self.synth_optim_simplex, self.synth_optim_values
            )

        for dim in range(self.nelder_mead_optim_dimension + 1):
            simplex[idx, dim] = initial_params
            if dim > 0:
                simplex[idx, dim][dim - 1] += 1.0

        for _iteration in range(OPTIM_N_ITERS):
            for dim in range(self.nelder_mead_optim_dimension + 1):
                values[idx, dim] = self.objective_function(
                    synth_data, synth_data_idx, xmin, simplex[idx, dim]
                )

            best_idx = 0
            worst_idx = 0
            second_worst_idx = 0
            for dim in range(self.nelder_mead_optim_dimension + 1):
                if values[idx, dim] < values[idx, best_idx]:
                    best_idx = dim
                if values[idx, dim] > values[idx, worst_idx]:
                    second_worst_idx = worst_idx
                    worst_idx = dim
                elif (
                    values[idx, dim] > values[idx, second_worst_idx]
                    and dim != worst_idx
                ):
                    second_worst_idx = dim

            centroid = ti.Vector([0.0] * self.max_dim)
            for dim in range(self.nelder_mead_optim_dimension + 1):
                if dim != worst_idx:
                    centroid += simplex[idx, dim]
            centroid /= self.nelder_mead_optim_dimension

            # reflection
            reflected_point = centroid + (centroid - simplex[idx, worst_idx])
            reflected_value = self.objective_function(
                synth_data, synth_data_idx, xmin, reflected_point
            )

            if reflected_value < values[idx, best_idx]:
                # expansion
                expanded_point = centroid + 2.0 * (centroid - simplex[idx, worst_idx])
                expanded_value = self.objective_function(
                    synth_data, synth_data_idx, xmin, expanded_point
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
                contracted_value = self.objective_function(
                    synth_data, synth_data_idx, xmin, contracted_point
                )
                if contracted_value < values[idx, worst_idx]:
                    simplex[idx, worst_idx] = contracted_point
                    values[idx, worst_idx] = contracted_value
                else:
                    # reduction
                    for dim in range(self.nelder_mead_optim_dimension + 1):
                        if dim != best_idx:
                            simplex[idx, dim] = 0.5 * (
                                simplex[idx, dim] + simplex[idx, best_idx]
                            )
                            values[idx, dim] = self.objective_function(
                                synth_data, synth_data_idx, xmin, simplex[idx, dim]
                            )

            max_value = -np.inf
            min_value = np.inf
            for dim in range(self.nelder_mead_optim_dimension + 1):
                max_value = max(max_value, values[idx, dim])
                min_value = min(min_value, values[idx, dim])

            if max_value - min_value < OBJ_FUNC_THRESHOLD:
                break

        best_idx = 0
        for dim in range(self.nelder_mead_optim_dimension + 1):
            if values[idx, dim] > values[idx, best_idx]:
                best_idx = dim

        return simplex[idx, best_idx]


# region utils

NUM_GAUSS_NODES = 16

gauss_nodes = ti.field(ti.f32, shape=NUM_GAUSS_NODES)
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
    ).astype(np.float32)
)

gauss_weights = ti.field(ti.f32, shape=NUM_GAUSS_NODES)
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
    ).astype(np.float32)
)


@ti.func
def gamma_integrand(alpha: ti.f32, t: ti.f32) -> ti.f32:
    """Gamma function integrand for Taichi computation."""
    return t ** (alpha - 1) * ti.exp(-t)


@ti.func
def incomplete_gamma(alpha: ti.f32, lower_limit: ti.f32) -> ti.f32:
    """Incomplete gamma function for Taichi computation."""
    result = ti.cast(0.0, ti.f32)
    for node_index in range(NUM_GAUSS_NODES):
        transformed_node = lower_limit + (
            gauss_nodes[node_index] / (1.0 - gauss_nodes[node_index])
        )
        weight = gauss_weights[node_index] / ((1.0 - gauss_nodes[node_index]) ** 2)
        result += weight * gamma_integrand(alpha, transformed_node)
    return result


@njit
def find_left_min(values, min_depth):  # noqa: ANN001, ANN201
    """Find left minimum in array using numba JIT."""
    left_ptr = 0
    right_ptr = 0

    if values.size == 1:
        return left_ptr

    while right_ptr != (values.size - 1) and values[right_ptr] < values[right_ptr + 1]:
        right_ptr += 1
    if right_ptr == (values.size - 1):
        return left_ptr

    left_ptr = right_ptr

    while right_ptr < (values.size - 1):
        if values[right_ptr] >= values[right_ptr + 1]:
            right_ptr += 1

        else:
            left_depth = values[left_ptr] - values[right_ptr]
            if left_depth < min_depth:
                right_ptr += 1
                continue
            left_ptr = right_ptr
            while right_ptr < (values.size - 1):
                if values[right_ptr] < values[left_ptr]:
                    left_ptr = right_ptr
                if values[right_ptr] < values[right_ptr + 1]:
                    right_ptr += 1
                else:
                    right_depth = values[right_ptr] - values[left_ptr]
                    if right_depth < min_depth:
                        right_ptr += 1
                        continue
                    break

            if right_ptr == (values.size - 1):
                right_depth = values[right_ptr] - values[left_ptr]
            if min(left_depth, right_depth) >= min_depth:
                return left_ptr
            left_ptr = right_ptr

    return BAD_RESULT


@njit
def find_left_min_in_cycle(values, min_depth):  # noqa: ANN001, ANN201
    """Find left minimum with iterative depth reduction using numba JIT."""
    min_ptr = BAD_RESULT
    while min_ptr == BAD_RESULT:
        min_ptr = find_left_min(values, min_depth)
        min_depth *= 0.5
        if min_depth < OBJ_FUNC_THRESHOLD:
            break

    return min_ptr


# endregion utils
