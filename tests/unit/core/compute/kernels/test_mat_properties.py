"""Analytic and property tests for matrix-property kernels.

greater_dim/lower_dim/aspect_ratio are shape accessors and a ratio: analytic
points covering both orientations pin them fully. weights_std carries a
homogeneity property (it scales with the matrix). weights_rand is a
permutation null model pinned by its true invariant: shuffling preserves the
exact multiset of entries and is deterministic given a fixed seed.
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

finite_matrices = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8),
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)


def test_dims_pick_shape_extremes() -> None:
    from diffract.core.compute.kernels.mat_properties import greater_dim, lower_dim

    wide = np.zeros((3, 5))
    tall = np.zeros((5, 3))
    # Both orientations agree only if the extreme (not a fixed axis) is taken.
    assert (greater_dim(wide), lower_dim(wide)) == (5, 3)
    assert (greater_dim(tall), lower_dim(tall)) == (5, 3)


def test_aspect_ratio_matches_ratio() -> None:
    from diffract.core.compute.kernels.mat_properties import aspect_ratio

    assert aspect_ratio(6, 4) == pytest.approx(1.5)
    assert aspect_ratio(5, 5) == pytest.approx(1.0)


def test_weights_std_matches_population_std() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_std

    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert weights_std(weights) == pytest.approx(np.sqrt(1.25))
    # population (ddof=0), not the sample deviation
    assert weights_std(weights) != pytest.approx(float(np.std(weights, ddof=1)))


@given(mat=finite_matrices, scale=st.floats(min_value=0.1, max_value=10.0))
def test_weights_std_is_scale_equivariant(
    mat: NDArray[np.float64], scale: float
) -> None:
    from diffract.core.compute.kernels.mat_properties import weights_std

    assert weights_std(scale * mat) == pytest.approx(
        scale * weights_std(mat), rel=1e-6, abs=1e-9
    )


@given(mat=finite_matrices)
def test_weights_rand_preserves_multiset_and_shape(mat: NDArray[np.float64]) -> None:
    from diffract.core.compute.kernels.mat_properties import weights_rand

    shuffled = weights_rand(mat)
    assert shuffled.shape == mat.shape
    assert np.array_equal(np.sort(shuffled, axis=None), np.sort(mat, axis=None))


def test_weights_rand_is_deterministic_for_fixed_seed() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_rand

    weights = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert np.array_equal(weights_rand(weights), weights_rand(weights))
    assert np.array_equal(weights_rand(weights, seed=7), weights_rand(weights, seed=7))


def test_weights_rand_does_not_mutate_input() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_rand

    weights = np.arange(12, dtype=np.float64).reshape(3, 4)
    original = weights.copy()
    weights_rand(weights)
    assert np.array_equal(weights, original)


def test_weights_rand_unseeded_is_nondeterministic() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_rand

    weights = np.arange(100, dtype=np.float64).reshape(10, 10)
    first = weights_rand(weights, seed=-1)
    second = weights_rand(weights, seed=-1)
    assert not np.array_equal(first, second)
    assert np.array_equal(np.sort(first, axis=None), np.sort(weights, axis=None))


def test_weights_std_propagates_nan() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_std

    # A diverged checkpoint (nan weights) passes extraction unvalidated;
    # weights_std carries the nan through, warning-free.
    assert np.isnan(weights_std(np.array([[1.0, np.nan], [2.0, 3.0]])))


def test_weights_rand_preserves_nan_entries() -> None:
    from diffract.core.compute.kernels.mat_properties import weights_rand

    # The multiset invariant must hold for a nan-bearing spectrum too; the
    # finite multiset test cannot cover it (nan != nan under array_equal).
    weights = np.array([[1.0, np.nan], [np.nan, 4.0]])
    expected_nan_count = 2
    shuffled = weights_rand(weights, seed=7)

    assert shuffled.shape == weights.shape
    assert int(np.isnan(shuffled).sum()) == expected_nan_count
    finite = np.sort(shuffled[~np.isnan(shuffled)])
    np.testing.assert_array_equal(finite, np.array([1.0, 4.0]))


def test_weights_rand_has_no_iteration_config() -> None:
    import inspect

    from diffract.core.compute.kernels.mat_properties import weights_rand

    # k shuffles == one shuffle, so no shuffle-count config is exposed.
    assert "n_randomise_iterations" not in inspect.signature(weights_rand).parameters


def test_weights_std_is_warning_free_on_inf_weights() -> None:
    import warnings

    from diffract.core.compute.kernels.mat_properties import weights_std

    # An inf-corrupted checkpoint must return nan without a numpy RuntimeWarning
    # (which filterwarnings=error would turn into a crash).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = weights_std(np.array([[np.inf, 1.0], [2.0, 3.0]]))
    assert np.isnan(result)
