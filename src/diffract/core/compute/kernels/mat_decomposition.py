from functools import partial
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.extensions.spectral import bind_svd
from diffract.core.compute.metadata import KernelInfo


@kernel(
    name="weights_svd",
    require_fields=("weights",),
    produce_fields=(
        "weights_lsvs",
        "weights_svals",
        "weights_rsvs",
    ),
    bind_execution=partial(bind_svd, output="factors"),
    version="2",
    stream_inputs=True,
)
def svd(
    mat: NDArray[np.floating[Any]],
    *,
    allow_cuda: bool = True,
    precision: str = "preserve",
) -> tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    r"""Economy SVD :math:`W = U\Sigma V^\top` with ascending :math:`\sigma_i`."""
    _, solve = bind_svd(
        {"precision": precision, "allow_cuda": allow_cuda}, output="factors"
    )
    return solve(mat).value


@kernel(
    name="weights_rand_svd",
    require_fields=("weights_rand",),
    produce_fields=("weights_rand_svals",),
    info=KernelInfo(formula=r"\boldsymbol{\sigma}(W^{\mathrm{rand}})"),
    bind_execution=partial(bind_svd, output="values"),
    version="3",
    stream_inputs=True,
)
def rand_singular_values(
    mat: NDArray[np.floating[Any]],
    *,
    allow_cuda: bool = True,
    precision: str = "preserve",
) -> NDArray[np.floating[Any]]:
    """Compute the ascending singular values of the entry-permuted weight matrix."""
    _, solve = bind_svd(
        {"precision": precision, "allow_cuda": allow_cuda}, output="values"
    )
    return solve(mat).value


@kernel(
    name="weights_rand_vectors",
    require_fields=("weights_rand",),
    produce_fields=("weights_rand_lsvs", "weights_rand_rsvs"),
    info=KernelInfo(formula=r"(U^{\mathrm{rand}}, V^{\mathrm{rand}})"),
    bind_execution=partial(bind_svd, output="vectors"),
    version="2",
    stream_inputs=True,
)
def rand_singular_vectors(
    mat: NDArray[np.floating[Any]],
    *,
    allow_cuda: bool = True,
    precision: str = "preserve",
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return the left and right singular vectors of the entry-permuted matrix."""
    _, solve = bind_svd(
        {"precision": precision, "allow_cuda": allow_cuda}, output="vectors"
    )
    return solve(mat).value


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
