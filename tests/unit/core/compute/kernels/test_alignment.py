"""Analytic and property tests for singular-vector alignment kernels.

Each kernel body is fed a hand-built overlap matrix / agreement vector
directly, so no other kernel is exercised. max_l_agreement/max_r_agreement
share one body (max_vector_agreement); avg_max_l/r_agreement share another
(avg_vector_agreement), so each body is tested once through the shared
function.
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

_overlap_shape = st.shared(
    hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8),
    key="overlap_shape",
)

overlaps = hnp.arrays(
    dtype=np.float64,
    shape=_overlap_shape,
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)

sign_flips = hnp.arrays(dtype=np.bool_, shape=_overlap_shape)


def test_max_agreement_is_per_row_max_abs() -> None:
    from diffract.core.compute.kernels.alignment import max_vector_agreement

    overlap = np.array(
        [
            [0.2, -0.9, 0.1],
            [0.5, 0.4, -0.3],
            [-1.0, 0.0, 0.6],
        ]
    )
    # The largest-magnitude entry per row is negative in rows 0 and 2, so a
    # missing abs would return the wrong (signed) value.
    np.testing.assert_array_equal(
        max_vector_agreement(overlap), np.array([0.9, 0.5, 1.0])
    )


def test_avg_agreement_is_the_mean() -> None:
    from diffract.core.compute.kernels.alignment import avg_vector_agreement

    agreement = np.array([0.9, 0.5, 1.0, 0.4])
    assert avg_vector_agreement(agreement) == pytest.approx(0.7)


def test_vector_agreement_is_the_overlap_diagonal() -> None:
    from diffract.core.compute.kernels.alignment import vector_agreement

    overlap = np.array(
        [
            [0.3, 0.9],
            [0.2, 0.8],
        ]
    )
    # Agreement is the on-diagonal overlap of matched components. This matrix
    # separates the diagonal (0.3, 0.8) from the per-row max (0.9, 0.8) and the
    # per-column max (0.3, 0.9), so a max/argmax reduction cannot pass it.
    np.testing.assert_array_equal(vector_agreement(overlap), np.array([0.3, 0.8]))


def test_avg_agreement_is_gauge_invariant_to_svd_sign() -> None:
    from diffract.core.compute.kernels.alignment import (
        avg_vector_agreement,
        overlap,
        vector_agreement,
    )
    from diffract.core.compute.kernels.mat_decomposition import svd

    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((64, 48))
    lsvs, _, _ = svd(matrix, allow_cuda=False)
    l_dim = min(matrix.shape)

    # A second checkpoint's SVD can return the same subspace under a flipped
    # sign gauge (the LAPACK sign is arbitrary); -lsvs is a legitimate basis of
    # the same matrix. Perfectly aligned components must read as agreement ~1,
    # not the artefactual -1 a signed overlap would yield.
    aligned = avg_vector_agreement(
        vector_agreement(overlap((lsvs, -lsvs), (l_dim, l_dim)))
    )
    assert aligned == pytest.approx(1.0)


@given(overlap=overlaps, flip=sign_flips)
def test_max_agreement_is_gauge_invariant(
    overlap: NDArray[np.float64], flip: NDArray[np.bool_]
) -> None:
    from diffract.core.compute.kernels.alignment import max_vector_agreement

    signs = np.where(flip, -1.0, 1.0)
    # Multiplying by exactly +/-1 only toggles sign bits, so max-abs is
    # bit-for-bit unchanged: the abs makes the metric gauge invariant.
    np.testing.assert_array_equal(
        max_vector_agreement(overlap * signs), max_vector_agreement(overlap)
    )


@given(
    n=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_signed_permutation_gives_unit_agreement(n: int, seed: int) -> None:
    from diffract.core.compute.kernels.alignment import max_vector_agreement

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    mat = np.zeros((n, n))
    mat[np.arange(n), perm] = signs

    # A (possibly sign-flipped) permutation is a perfect one-to-one match:
    # every row's best agreement is exactly 1.
    np.testing.assert_array_equal(max_vector_agreement(mat), np.ones(n))


def test_l_and_r_variants_share_the_body(ram_container) -> None:
    from diffract.core.compute.kernels.alignment import (
        avg_vector_agreement,
        max_vector_agreement,
    )

    registry = ram_container.compute_singleton.kernel_registry()

    # l/r siblings differ only in the overlap field they consume; both names
    # must resolve to one shared body (@wraps exposes it as __wrapped__).
    max_l = registry.get_kernel_implementation("max_l_agreement")
    max_r = registry.get_kernel_implementation("max_r_agreement")
    avg_l = registry.get_kernel_implementation("avg_max_l_agreement")
    avg_r = registry.get_kernel_implementation("avg_max_r_agreement")

    assert max_l.__wrapped__ is max_r.__wrapped__ is max_vector_agreement
    assert avg_l.__wrapped__ is avg_r.__wrapped__ is avg_vector_agreement


def test_agreement_kernels_propagate_nan() -> None:
    from diffract.core.compute.kernels.alignment import (
        avg_vector_agreement,
        max_vector_agreement,
    )

    # A diverged checkpoint has nan singular vectors, poisoning the whole
    # overlap to nan; both reductions must carry the nan through.
    assert np.all(np.isnan(max_vector_agreement(np.full((2, 3), np.nan))))
    assert np.isnan(avg_vector_agreement(np.full(3, np.nan)))


def test_overlap_is_abs_gram_truncated_to_lower_dim() -> None:
    from diffract.core.compute.kernels.alignment import overlap

    svs = np.eye(3)
    svs_other = np.array(
        [
            [0.6, -0.8, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # svs.T @ svs_other == svs_other here; abs neutralises the arbitrary -0.8
    # sign gauge, and lower_dim=(2, 2) truncates the unmatched third component.
    np.testing.assert_allclose(
        overlap((svs, svs_other), (2, 2)),
        np.array([[0.6, 0.8], [0.8, 0.6]]),
    )


def test_overlap_rejects_mismatched_lower_dim() -> None:
    from diffract.core.compute.kernels.alignment import overlap

    svs = np.eye(3)

    # The two checkpoints must share a lower dimension; a mismatch is a wiring
    # error, not a silently truncated overlap.
    with pytest.raises(ValueError, match="equal"):
        overlap((svs, svs), (2, 3))
