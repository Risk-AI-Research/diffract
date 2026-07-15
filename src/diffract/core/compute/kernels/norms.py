import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.compute.decorator import kernel
from diffract.core.compute.execution.enums import KernelApplyLevel
from diffract.core.compute.metadata import KernelInfo


def _log10_or_nan(
    values: tuple[float, ...] | NDArray[np.floating[Any]], *, square: bool
) -> NDArray[np.floating[Any]]:
    """log10 of positive norms; a degenerate norm (<= 0 or nan) maps to nan.

    A dead layer (norm 0) or a diverged one (nan) is not measurable and must be
    skipped by the aggregation rather than collapsing it to -inf.
    """
    arr = np.asarray(values, dtype=float)
    base = arr**2 if square else arr
    with np.errstate(divide="ignore", invalid="ignore"):
        return cast(
            "NDArray[np.floating[Any]]",
            np.where(arr > 0, np.log10(base), np.nan),
        )


def _reduce_measurable(
    logs: NDArray[np.floating[Any]], *, reduce: Literal["mean", "sum"]
) -> float:
    """Reduce over the measurable (non-nan) layers; all-degenerate -> nan."""
    if np.all(np.isnan(logs)):
        return float("nan")
    value = np.nanmean(logs) if reduce == "mean" else np.nansum(logs)
    return cast("float", float(value))


@kernel(
    name="pl_alpha_norm",
    require_fields=("esd", "pl_alpha"),
    info=KernelInfo(formula=r"\sum_i \lambda_i^{\alpha_{\mathrm{PL}}}"),
)
@kernel(
    name="tpl_alpha_norm",
    require_fields=("esd", "tpl_alpha"),
    info=KernelInfo(formula=r"\sum_i \lambda_i^{\alpha_{\mathrm{TPL}}}"),
)
def alpha_norm(esd: NDArray[np.floating[Any]], alpha: float) -> float:
    r"""Weighted alpha norm :math:`\sum_i \lambda_i^{\alpha}`."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = cast("float", np.sum(esd**alpha).item())
    # A near-isometric (orthogonal-init) layer produces a garbage-huge alpha
    # whose weighted sum under/overflows to a non-finite or exactly-zero value
    # on a nonzero spectrum: unfittable, not a real zero -> not measurable here.
    if float(np.max(esd)) > 0 and not (math.isfinite(result) and result > 0):
        return float("nan")
    return result


@kernel(
    name="model_pl_alpha_norm",
    require_fields=("pl_alpha_norm",),
    apply_level=KernelApplyLevel.IN_MODEL,
    info=KernelInfo(
        formula=r"\langle \log_{10}\sum_i \lambda_i^{\alpha_{\mathrm{PL}}}\rangle"
    ),
)
@kernel(
    name="model_tpl_alpha_norm",
    require_fields=("tpl_alpha_norm",),
    apply_level=KernelApplyLevel.IN_MODEL,
    info=KernelInfo(
        formula=r"\langle \log_{10}\sum_i \lambda_i^{\alpha_{\mathrm{TPL}}}\rangle"
    ),
)
def model_alpha_norm(alpha_norm: tuple[float, ...]) -> float:
    r"""Mean log10 alpha norm over the measurable model parameters.

    :math:`\langle \log_{10}\sum_i \lambda_i^{\alpha}\rangle`
    """
    return _reduce_measurable(_log10_or_nan(alpha_norm, square=False), reduce="mean")


@kernel
def frob_norm(weights_svals: NDArray[np.floating[Any]]) -> float:
    r"""Frobenius norm :math:`\lVert W\rVert_F = \sqrt{\sum_i \sigma_i^2}`."""
    return math.sqrt(cast("float", np.sum(weights_svals**2).item()))


@kernel
def nuclear_norm(weights_svals: NDArray[np.floating[Any]]) -> float:
    r"""Nuclear (Schatten-1) norm :math:`\lVert W\rVert_* = \sum_i \sigma_i`."""
    return cast("float", weights_svals.sum().item())


@kernel
def l2_norm(max_weights_sval: float) -> float:
    r"""Spectral (L2) norm :math:`\lVert W\rVert_2 = \sigma_{\max}`."""
    return max_weights_sval


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_norm(frob_norm: tuple[float, ...]) -> float:
    r"""Mean log10 squared Frobenius norm over the measurable parameters.

    :math:`\langle \log_{10}\lVert W\rVert_F^2\rangle`
    """
    return _reduce_measurable(_log10_or_nan(frob_norm, square=True), reduce="mean")


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_spectral_norm(l2_norm: tuple[float, ...]) -> float:
    r"""Mean log10 squared spectral norm over the measurable parameters.

    :math:`\langle \log_{10}\lVert W\rVert_2^2\rangle`
    """
    return _reduce_measurable(_log10_or_nan(l2_norm, square=True), reduce="mean")


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def param_norm(frob_norm: tuple[float, ...]) -> float:
    r"""Sum of squared Frobenius norms :math:`\sum_\ell \lVert W_\ell\rVert_F^2`."""
    frob_norm_arr: NDArray[np.floating[Any]] = np.array(frob_norm)
    return cast("float", np.sum(frob_norm_arr**2).item())


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_prod_frob_norm(frob_norm: tuple[float, ...]) -> float:
    r"""Log10 of the product of Frobenius norms (sum of their log10).

    :math:`\sum_\ell \log_{10}\lVert W_\ell\rVert_F`
    """
    return _reduce_measurable(_log10_or_nan(frob_norm, square=False), reduce="sum")


@kernel(apply_level=KernelApplyLevel.IN_MODEL)
def log_prod_spectral_norm(l2_norm: tuple[float, ...]) -> float:
    r"""Log10 of the product of spectral norms (sum of their log10).

    :math:`\sum_\ell \log_{10}\lVert W_\ell\rVert_2`
    """
    return _reduce_measurable(_log10_or_nan(l2_norm, square=False), reduce="sum")
