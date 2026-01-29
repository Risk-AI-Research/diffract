from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
    KernelRestrictions,
)


@kernel(
    name="l_overlap",
    require_fields=("weights_lsvs", "lower_dim"),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    restrictions=KernelRestrictions.BINARY,
)
@kernel(
    name="r_overlap",
    require_fields=("weights_rsvs", "lower_dim"),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    restrictions=KernelRestrictions.BINARY,
)
def overlap(
    weights_svs: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    lower_dim: tuple[int, int],
    *,
    absolute: bool = False,
) -> NDArray[np.floating[Any]]:
    """Compute overlap matrix between two sets of singular vectors."""
    svs, svs_other = weights_svs

    l_dim, l_dim_other = lower_dim
    if l_dim != l_dim_other:
        msg = "l_dim and l_dim_other should be equal"
        raise ValueError(msg)

    result = (svs.T @ svs_other)[:l_dim, :l_dim]

    if absolute:
        result = np.abs(result)

    return cast("NDArray[np.floating[Any]]", result)


@kernel(
    name="l_agreement",
    require_fields=("l_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
)
@kernel(
    name="r_agreement",
    require_fields=("r_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
)
def vector_agreement(overlap: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Return the diagonal of an overlap matrix (per-component agreement)."""
    result = np.diag(overlap)
    return cast("NDArray[np.floating[Any]]", result)


@kernel(
    name="max_l_agreement",
    require_fields=("l_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
)
@kernel(
    name="max_r_agreement",
    require_fields=("r_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
)
def max_vector_agreement(
    overlap: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Return max absolute overlap per row (best agreement per component)."""
    result = np.max(np.abs(overlap), axis=1)
    return cast("NDArray[np.floating[Any]]", result)


@kernel(execution_protocol=KernelExecutionProtocol.PARALLEL)
def svs_similarity(
    weights_lsvs: NDArray[np.floating[Any]], weights_rsvs: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Compute similarity between left and right singular vectors."""
    result = np.einsum("ij,ij->j", weights_lsvs, weights_rsvs)
    return cast("NDArray[np.floating[Any]]", result)
