import functools
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import diffract.core.utils.imports as import_utils
from diffract.core.compute.decorator import kernel
from diffract.core.compute.metadata import KernelInfo

torch = import_utils.optional_import("torch")
TORCH_AVAILABLE = torch is not None
if TORCH_AVAILABLE:
    from diffract.core.compute.extensions.utils import torch_cuda_wrapper

    IS_CUDA_AVAILABLE = bool(torch.cuda.is_available())
else:
    IS_CUDA_AVAILABLE = False


@kernel(
    name="weights_svd",
    require_fields=("weights",),
    produce_fields=(
        "weights_lsvs",
        "weights_svals",
        "weights_rsvs",
    ),
)
@kernel(
    name="weights_rand_svd",
    require_fields=("weights_rand",),
    produce_fields=(
        "weights_rand_lsvs",
        "weights_rand_svals",
        "weights_rand_rsvs",
    ),
    info=KernelInfo(formula=r"W_{\mathrm{rand}} = U\Sigma V^\top"),
)
def svd(
    mat: NDArray[np.floating[Any]], *, allow_cuda: bool = True
) -> tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    r"""Economy SVD :math:`W = U\Sigma V^\top` with ascending :math:`\sigma_i`."""
    if IS_CUDA_AVAILABLE and allow_cuda:
        svd_fn = functools.partial(torch.linalg.svd, full_matrices=False)
        u, svals, vt = torch_cuda_wrapper(mat, svd_fn)
    else:
        try:
            u, svals, vt = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            # nan weights make gesdd fail to converge; mirror the inf path's
            # all-nan spectrum so the nan contract holds downstream.
            k = min(mat.shape)
            u = np.full((mat.shape[0], k), np.nan)
            svals = np.full(k, np.nan)
            vt = np.full((k, mat.shape[1]), np.nan)

    svals = np.abs(svals.real).flatten()
    svals_argsort_index = np.argsort(svals)

    lsvs_sorted = u[:, svals_argsort_index]
    svals_sorted = svals[svals_argsort_index]
    rsvs_sorted = vt.T[:, svals_argsort_index]

    return lsvs_sorted, svals_sorted, rsvs_sorted


@kernel(name="esd", require_fields=("weights_svals", "greater_dim"))
@kernel(
    name="esd_rand",
    require_fields=("weights_rand_svals", "greater_dim"),
    info=KernelInfo(
        formula=r"\lambda_i^{\mathrm{rand}} = (\sigma_i^{\mathrm{rand}})^2 / N"
    ),
)
def esd(
    svals: NDArray[np.floating[Any]], greater_dim: int
) -> NDArray[np.floating[Any]]:
    r"""Empirical spectral distribution :math:`\lambda_i = \sigma_i^2 / N`."""
    result = np.square(svals) / greater_dim
    return cast("NDArray[np.floating[Any]]", result)


# region utils


@kernel(name="max_weights_sval", require_fields=("weights_svals",))
@kernel(
    name="max_weights_rand_sval",
    require_fields=("weights_rand_svals",),
    info=KernelInfo(
        formula=r"\sigma_{\max}^{\mathrm{rand}} = \max_i \sigma_i^{\mathrm{rand}}"
    ),
)
def max_sval(svals: NDArray[np.floating[Any]]) -> float:
    r"""Maximum singular value :math:`\sigma_{\max} = \max_i \sigma_i`."""
    return cast("float", svals[-1].item())


@kernel(name="min_weights_sval", require_fields=("weights_svals",))
@kernel(
    name="min_weights_rand_sval",
    require_fields=("weights_rand_svals",),
    info=KernelInfo(
        formula=r"\sigma_{\min}^{\mathrm{rand}} = \min_i \sigma_i^{\mathrm{rand}}"
    ),
)
def min_sval(svals: NDArray[np.floating[Any]]) -> float:
    r"""Minimum singular value :math:`\sigma_{\min} = \min_i \sigma_i`."""
    return cast("float", svals[0].item())


@kernel(name="esd_max", require_fields=("esd",))
@kernel(
    name="esd_rand_max",
    require_fields=("esd_rand",),
    info=KernelInfo(
        formula=r"\lambda_{\max}^{\mathrm{rand}} = \max_i \lambda_i^{\mathrm{rand}}"
    ),
)
def esd_max(esd: NDArray[np.floating[Any]]) -> float:
    r"""Maximum ESD eigenvalue :math:`\lambda_{\max} = \max_i \lambda_i`."""
    return cast("float", esd[-1].item())


@kernel(name="esd_min", require_fields=("esd",))
@kernel(
    name="esd_rand_min",
    require_fields=("esd_rand",),
    info=KernelInfo(
        formula=r"\lambda_{\min}^{\mathrm{rand}} = \min_i \lambda_i^{\mathrm{rand}}"
    ),
)
def esd_min(esd: NDArray[np.floating[Any]]) -> float:
    r"""Minimum ESD eigenvalue :math:`\lambda_{\min} = \min_i \lambda_i`."""
    return cast("float", esd[0].item())


# endregion utils
