import io
import warnings
from contextlib import redirect_stderr, redirect_stdout
from operator import itemgetter
from typing import Any, Literal, cast

import numpy as np
import powerlaw as pl
from numpy.typing import NDArray

import diffract.core.utils.imports as import_utils
from diffract.core.compute.decorator import kernel

_power_law_ext = import_utils.get_module("diffract.core.compute.extensions.power_law")
Fit = None if _power_law_ext is None else getattr(_power_law_ext, "Fit", None)
_DIFFRACT_FIT_AVAILABLE = Fit is not None


def _require_diffract_fit() -> None:
    if not _DIFFRACT_FIT_AVAILABLE or Fit is None:
        raise ModuleNotFoundError(
            "Taichi Fit implementation is not available. "
            "Install the 'taichi' extra or use fit_method='powerlaw'."
        )


# region general


@kernel(name="pl_concentration", require_fields=("esd", "pl_esd_xmin"))
@kernel(name="tpl_concentration", require_fields=("esd", "tpl_esd_xmin"))
@kernel(name="expon_concentration", require_fields=("esd", "expon_concentration"))
def ht_concentration(esd: NDArray[np.floating[Any]], ht_esd_xmin: float) -> float:
    """Compute heavy-tailed concentration (tail size / total size)."""
    tail_size = cast("int", np.sum(esd > ht_esd_xmin).item())

    return tail_size / esd.size


@kernel(name="pl_presence", require_fields=("esd_min", "esd_max", "pl_esd_xmin"))
@kernel(name="tpl_presence", require_fields=("esd_min", "esd_max", "tpl_esd_xmin"))
@kernel(name="expon_presence", require_fields=("esd_min", "esd_max", "expon_esd_xmin"))
def ht_presence(esd_min: float, esd_max: float, ht_esd_xmin: float) -> float:
    """Compute heavy-tailed presence (tail width / total width)."""
    esd_width = esd_max - esd_min
    ht_width = esd_max - ht_esd_xmin

    return ht_width / esd_width


@kernel(name="tpl_scale", require_fields=("esd_max", "tpl_lambda"))
@kernel(name="expon_scale", require_fields=("esd_max", "expon_lambda"))
def ht_scale(esd_max: float, ht_lambda: float) -> float:
    """Compute heavy-tailed scale (max * lambda)."""
    return esd_max * ht_lambda


# endregion general

# region power_law


def power_law_fit_powerlaw_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Fit power law using powerlaw library."""
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        distribution = "power_law"
        fit = pl.Fit(
            data,
            distribution=distribution,
            xmin_distribution=distribution,
            discrete=False,
        )

    fit_result = getattr(fit, distribution)
    pl_alpha = fit_result.alpha
    pl_esd_xmin = fit_result.xmin
    pl_ks = fit_result.KS()

    return pl_alpha, pl_esd_xmin, pl_ks


def power_law_fit_diffract_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Fit power law using diffract's Taichi implementation."""
    _require_diffract_fit()
    distribution = "power_law"
    fit = Fit(data, distribution)  # type: ignore[misc]
    params = fit.fit_params()

    pl_alpha, pl_esd_xmin, pl_ks = itemgetter("pl_alpha", "xmin", "ks_distance")(params)

    return pl_alpha, pl_esd_xmin, pl_ks


@kernel(produce_fields=("pl_alpha", "pl_esd_xmin", "pl_ks"))
def power_law_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: Literal["powerlaw", "diffract"] = "diffract",
) -> tuple[float, float, float]:
    """Fit power law distribution to ESD data."""
    match fit_method:
        case "powerlaw":
            fit_implementation = power_law_fit_powerlaw_implementation
        case "diffract":
            fit_implementation = power_law_fit_diffract_implementation
        case _:
            msg = (
                f"Fit method {fit_method} not implemented, choose method from "
                "('powerlaw', 'diffract')."
            )
            raise ValueError(msg)

    pl_alpha, pl_esd_xmin, pl_ks = fit_implementation(esd)

    return pl_alpha, pl_esd_xmin, pl_ks


@kernel
def pl_p_value(
    esd: NDArray[np.floating[Any]], pl_alpha: float, pl_esd_xmin: float, pl_ks: float
) -> float:
    """Compute p-value for power law fit."""
    _require_diffract_fit()
    fit = Fit(esd, "power_law")  # type: ignore[misc]
    fit.set_params(
        xmin=pl_esd_xmin,
        pl_alpha=pl_alpha,
        ks_distance=pl_ks,
        tail_size=cast("float", np.sum(esd >= pl_esd_xmin).item()),
    )
    _, p_value = fit.p_value_test()

    return p_value


# endregion power_law

# region truncated_power_law


def truncated_power_law_fit_powerlaw_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float, float]:
    """Fit truncated power law using powerlaw library."""
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        distribution = "truncated_power_law"
        fit = pl.Fit(
            data,
            distribution=distribution,
            xmin_distribution=distribution,
            discrete=False,
        )

    fit_result = getattr(fit, distribution)
    tpl_alpha = fit_result.alpha
    tpl_lambda = fit_result.Lambda
    tpl_esd_xmin = fit_result.xmin
    tpl_ks = fit_result.KS()

    return tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks


def truncated_power_law_fit_diffract_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float, float]:
    """Fit truncated power law using diffract's Taichi implementation."""
    _require_diffract_fit()
    distribution = "truncated_power_law"
    fit = Fit(data, distribution)  # type: ignore[misc]
    params = fit.fit_params()

    tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks = itemgetter(
        "pl_alpha", "expon_lambda", "xmin", "ks_distance"
    )(params)

    return tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks


@kernel(produce_fields=("tpl_alpha", "tpl_lambda", "tpl_esd_xmin", "tpl_ks"))
def truncated_power_law_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: Literal["powerlaw", "diffract"] = "diffract",
) -> tuple[float, float, float, float]:
    """Fit truncated power law distribution to ESD data."""
    match fit_method:
        case "powerlaw":
            fit_implementation = truncated_power_law_fit_powerlaw_implementation
        case "diffract":
            fit_implementation = truncated_power_law_fit_diffract_implementation
        case _:
            msg = (
                f"Fit method {fit_method} not implemented, choose method from "
                "('powerlaw', 'diffract')."
            )
            raise ValueError(msg)

    tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks = fit_implementation(esd)

    return tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks


@kernel
def tpl_p_value(
    esd: NDArray[np.floating[Any]],
    tpl_alpha: float,
    tpl_lambda: float,
    tpl_esd_xmin: float,
    tpl_ks: float,
) -> float:
    """Compute p-value for truncated power law fit."""
    _require_diffract_fit()
    fit = Fit(esd, "truncated_power_law")  # type: ignore[misc]
    fit.set_params(
        xmin=tpl_esd_xmin,
        pl_alpha=tpl_alpha,
        expon_lambda=tpl_lambda,
        ks_distance=tpl_ks,
        tail_size=cast("float", np.sum(esd >= tpl_esd_xmin).item()),
    )
    _, p_value = fit.p_value_test()

    return p_value


# endregion truncated_power_law

# region exponential


def exponential_fit_powerlaw_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Fit exponential distribution using powerlaw library."""
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        distribution = "exponential"
        fit = pl.Fit(
            data,
            distribution=distribution,
            xmin_distribution=distribution,
            discrete=False,
        )

    fit_result = getattr(fit, distribution)
    expon_lambda = fit_result.Lambda
    expon_esd_xmin = fit_result.xmin
    expon_ks = fit_result.KS()

    return expon_lambda, expon_esd_xmin, expon_ks


def exponential_fit_diffract_implementation(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Fit exponential distribution using diffract's Taichi implementation."""
    _require_diffract_fit()
    distribution = "exponential"
    fit = Fit(data, distribution)  # type: ignore[misc]
    params = fit.fit_params()

    expon_lambda, expon_esd_xmin, expon_ks = itemgetter(
        "expon_lambda", "xmin", "ks_distance"
    )(params)

    return expon_lambda, expon_esd_xmin, expon_ks


@kernel(produce_fields=("expon_lambda", "expon_esd_xmin", "expon_ks"))
def exponential_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: Literal["powerlaw", "diffract"] = "diffract",
) -> tuple[float, float, float]:
    """Fit exponential distribution to ESD data."""
    match fit_method:
        case "powerlaw":
            fit_implementation = exponential_fit_powerlaw_implementation
        case "diffract":
            fit_implementation = exponential_fit_diffract_implementation
        case _:
            msg = (
                f"Fit method {fit_method} not implemented, choose method from "
                "('powerlaw', 'diffract')."
            )
            raise ValueError(msg)

    expon_lambda, expon_esd_xmin, expon_ks = fit_implementation(esd)

    return expon_lambda, expon_esd_xmin, expon_ks


@kernel
def expon_p_value(
    esd: NDArray[np.floating[Any]],
    expon_lambda: float,
    expon_esd_xmin: float,
    expon_ks: float,
) -> float:
    """Compute p-value for exponential fit."""
    _require_diffract_fit()
    fit = Fit(esd, "exponential")  # type: ignore[misc]
    fit.set_params(
        xmin=expon_esd_xmin,
        expon_lambda=expon_lambda,
        ks_distance=expon_ks,
        tail_size=cast("float", np.sum(esd >= expon_esd_xmin).item()),
    )
    _, p_value = fit.p_value_test()

    return p_value
