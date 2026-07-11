from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel


@kernel
def greater_dim(weights: NDArray[int]) -> int:
    """Get greater dimension from shape."""
    return max(weights.shape)


@kernel
def lower_dim(weights: NDArray[int]) -> int:
    """Get lower dimension from shape."""
    return min(weights.shape)


@kernel
def aspect_ratio(greater_dim: int, lower_dim: int) -> float:
    """Compute aspect ratio (greater_dim / lower_dim)."""
    return greater_dim / lower_dim


@kernel
def weights_std(weights: NDArray[np.floating[Any]]) -> float:
    """Compute standard deviation of weights."""
    return cast("float", weights.std())


@kernel
def weights_rand(
    weights: NDArray[np.floating[Any]],
    *,
    n_randomise_iterations: int = 1,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate randomized version of weights by shuffling."""
    rng = np.random.default_rng(None if seed == -1 else seed)

    result = weights.copy()

    shape = result.shape
    index = np.arange(np.prod(shape))

    result = result.reshape(-1)
    for _ in range(n_randomise_iterations):
        rng.shuffle(index)
    result = result[index]

    return result.reshape(shape)
