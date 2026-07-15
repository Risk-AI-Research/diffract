from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelRestrictions,
)
from diffract.core.compute.metadata import KernelInfo


@kernel(
    name="l_overlap",
    require_fields=("weights_lsvs", "lower_dim"),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    restrictions=KernelRestrictions.BINARY,
    info=KernelInfo(formula=r"O^{L} = \lvert U_1^\top U_2\rvert"),
)
@kernel(
    name="r_overlap",
    require_fields=("weights_rsvs", "lower_dim"),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    restrictions=KernelRestrictions.BINARY,
    info=KernelInfo(formula=r"O^{R} = \lvert V_1^\top V_2\rvert"),
)
def overlap(
    weights_svs: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    lower_dim: tuple[int, int],
) -> NDArray[np.floating[Any]]:
    r"""Absolute overlap of two singular-vector bases.

    :math:`\lvert S_1^\top S_2\rvert` -- the SVD sign gauge is arbitrary, so only
    the absolute overlap is an invariant of the checkpoint pair.
    """
    svs, svs_other = weights_svs

    l_dim, l_dim_other = lower_dim
    if l_dim != l_dim_other:
        msg = "l_dim and l_dim_other should be equal"
        raise ValueError(msg)

    result = np.abs((svs.T @ svs_other)[:l_dim, :l_dim])

    return cast("NDArray[np.floating[Any]]", result)


@kernel(
    name="l_agreement",
    require_fields=("l_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"(O^{L})_{ii}"),
)
@kernel(
    name="r_agreement",
    require_fields=("r_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"(O^{R})_{ii}"),
)
def vector_agreement(overlap: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    r"""Per-component agreement :math:`a_i = O_{ii}` (diagonal of the overlap)."""
    result = np.diag(overlap)
    return cast("NDArray[np.floating[Any]]", result)


@kernel(
    name="max_l_agreement",
    require_fields=("l_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\max_j (O^{L})_{ij}"),
)
@kernel(
    name="max_r_agreement",
    require_fields=("r_overlap",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\max_j (O^{R})_{ij}"),
)
def max_vector_agreement(
    overlap: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Best-match agreement per component :math:`\max_j \lvert O_{ij}\rvert`."""
    result = np.max(np.abs(overlap), axis=1)
    return cast("NDArray[np.floating[Any]]", result)


@kernel(
    name="avg_l_agreement",
    require_fields=("l_agreement",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\big\langle (O^{L})_{ii}\big\rangle"),
)
@kernel(
    name="avg_r_agreement",
    require_fields=("r_agreement",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\big\langle (O^{R})_{ii}\big\rangle"),
)
@kernel(
    name="avg_max_l_agreement",
    require_fields=("max_l_agreement",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\big\langle \max_j (O^{L})_{ij}\big\rangle"),
)
@kernel(
    name="avg_max_r_agreement",
    require_fields=("max_r_agreement",),
    apply_level=KernelApplyLevel.CROSS_MODEL,
    info=KernelInfo(formula=r"\big\langle \max_j (O^{R})_{ij}\big\rangle"),
)
def avg_vector_agreement(
    agreement: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Mean per-component agreement :math:`\langle O_{ii}\rangle`."""
    result = np.mean(agreement)
    return cast("NDArray[np.floating[Any]]", result)
