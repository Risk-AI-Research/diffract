import math
from sys import float_info
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.execution.enums import KernelApplyLevel


@kernel(name="pl_alpha_norm", require_fields=("esd", "pl_alpha"))
@kernel(name="tpl_alpha_norm", require_fields=("esd", "tpl_alpha"))
def alpha_norm(esd: NDArray[np.floating[Any]], alpha: float) -> float:
    """Compute alpha norm: sum(esd^alpha)."""
    return cast("float", np.sum(esd**alpha).item())


@kernel(
    name="model_pl_alpha_norm",
    require_fields=("pl_alpha_norm",),
    apply_level=KernelApplyLevel.IN_MODEL,
)
@kernel(
    name="model_tpl_alpha_norm",
    require_fields=("tpl_alpha_norm",),
    apply_level=KernelApplyLevel.IN_MODEL,
)
def model_alpha_norm(alpha_norm: tuple[float, ...]) -> float:
    """Compute mean log10 alpha norm across model parameters."""
    alpha_norm_arr: NDArray[np.floating[Any]] = np.array(alpha_norm)
    log_alpha_norm = np.where(
        alpha_norm_arr == 0, float_info.min_10_exp, np.log10(alpha_norm_arr)
    )
    return cast("float", np.mean(log_alpha_norm).item())


@kernel
def frob_norm(weights_svals: NDArray[np.floating[Any]]) -> float:
    """Compute Frobenius norm from singular values."""
    return math.sqrt(cast("float", np.sum(weights_svals**2).item()))


@kernel
def l1_norm(weights_svals: NDArray[np.floating[Any]]) -> float:
    """Compute L1 norm (sum of singular values)."""
    return cast("float", weights_svals.sum().item())


@kernel
def l2_norm(max_weights_sval: float) -> float:
    """Compute L2 (spectral) norm (maximum singular value)."""
    return max_weights_sval


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_norm(frob_norm: tuple[float, ...]) -> float:
    """Compute mean squared log Frobenius norm across model parameters."""
    frob_norm_arr: NDArray[np.floating[Any]] = np.array(frob_norm)
    return cast("float", np.mean(np.log(frob_norm_arr) ** 2).item())


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_spectral_norm(l2_norm: tuple[float, ...]) -> float:
    """Compute mean squared log spectral norm across model parameters."""
    l2_norm_arr: NDArray[np.floating[Any]] = np.array(l2_norm)
    return cast("float", np.mean(np.log(l2_norm_arr) ** 2).item())


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def param_norm(frob_norm: tuple[float, ...]) -> float:
    """Compute sum of squared Frobenius norms across model parameters."""
    frob_norm_arr: NDArray[np.floating[Any]] = np.array(frob_norm)
    return cast("float", np.sum(frob_norm_arr**2).item())


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def prod_frob_norm(frob_norm: tuple[float, ...]) -> float:
    """Compute product of Frobenius norms across model parameters."""
    frob_norm_arr: NDArray[np.floating[Any]] = np.array(frob_norm)
    return cast("float", np.prod(frob_norm_arr).item())


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def prod_spectral_norm(l2_norm: tuple[float, ...]) -> float:
    """Compute product of spectral norms across model parameters."""
    l2_norm_arr: NDArray[np.floating[Any]] = np.array(l2_norm)
    return cast("float", np.prod(l2_norm_arr).item())
