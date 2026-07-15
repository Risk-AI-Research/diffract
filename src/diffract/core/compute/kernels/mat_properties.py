from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel


@kernel
def greater_dim(weights: NDArray[int]) -> int:
    r"""Greater matrix dimension :math:`N = \max(m, n)`."""
    return max(weights.shape)


@kernel
def lower_dim(weights: NDArray[int]) -> int:
    r"""Lower matrix dimension :math:`M = \min(m, n)`."""
    return min(weights.shape)


@kernel
def aspect_ratio(greater_dim: int, lower_dim: int) -> float:
    r"""Aspect ratio :math:`Q = N / M \ge 1`."""
    return greater_dim / lower_dim


@kernel
def weights_std(weights: NDArray[np.floating[Any]]) -> float:
    r"""Standard deviation of the weight entries :math:`\operatorname{std}(W)`."""
    with np.errstate(invalid="ignore"):
        return cast("float", weights.std())


@kernel
def weights_rand(
    weights: NDArray[np.floating[Any]],
    *,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    r"""Permutation null model of the weight matrix.

    :math:`W_{\mathrm{rand}} = \operatorname{reshape}(P\,\operatorname{vec} W)`
    for a uniform permutation :math:`P` of the entries of :math:`W`: it preserves
    the multiset of weights and destroys correlation structure.
    """
    rng = np.random.default_rng(None if seed == -1 else seed)

    result = weights.copy()

    shape = result.shape
    index = np.arange(np.prod(shape))
    rng.shuffle(index)

    return result.reshape(-1)[index].reshape(shape)
