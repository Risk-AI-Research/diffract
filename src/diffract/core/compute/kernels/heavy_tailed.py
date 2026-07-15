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
from diffract.core.compute.metadata import KernelInfo

# Availability is probed without importing the accelerated module itself:
# importing it initializes the taichi runtime, which must not happen (and
# must not be able to fail) during `import diffract`.
_DIFFRACT_FIT_AVAILABLE = import_utils.is_available("taichi")

_fit_class: Any = None
_DIFFRACT_FIT_BROKEN = False

# The statistical floor of the accelerated fitter: its xmin candidates
# must leave a tail of MIN_TAIL_SIZE (50) points, per Clauset et al., so
# below ~2x that it cannot fit at all. From this size up it is also
# faster than the `powerlaw` library at every tail size once warm.
AUTO_DIFFRACT_MIN_SIZE = 100

FitMethod = Literal["auto", "powerlaw", "diffract"]


def _accelerated_fit_ready() -> bool:
    """True when the accelerated implementation can actually initialize.

    taichi may be importable yet fail at runtime initialization; `auto`
    must degrade to the powerlaw library then, not raise.
    """
    global _DIFFRACT_FIT_BROKEN  # noqa: PLW0603
    if _DIFFRACT_FIT_BROKEN or not _DIFFRACT_FIT_AVAILABLE:
        return False
    try:
        _get_fit_class()
    except ModuleNotFoundError:
        _DIFFRACT_FIT_BROKEN = True
        return False
    return True


def _resolve_fit_method(fit_method: FitMethod, data_size: int) -> str:
    """Resolve `auto` to an implementation by data size and availability.

    `auto` never raises over the accelerated path: explicit
    fit_method="diffract" surfaces initialization errors instead.
    """
    if fit_method != "auto":
        return fit_method
    if data_size < AUTO_DIFFRACT_MIN_SIZE:
        return "powerlaw"
    return "diffract" if _accelerated_fit_ready() else "powerlaw"


def _get_fit_class() -> Any:
    """Import the accelerated Fit implementation on first use."""
    global _fit_class  # noqa: PLW0603
    if _fit_class is None:
        try:
            module = import_utils.get_module(
                "diffract.core.compute.extensions.power_law"
            )
            _fit_class = None if module is None else module.Fit
        except Exception as e:  # optional accelerator must not crash callers
            raise ModuleNotFoundError(
                "The accelerated 'diffract' fit implementation failed to "
                f"initialize: {e}. Use fit_method='powerlaw' instead."
            ) from e
        if _fit_class is None:
            raise ModuleNotFoundError(
                "The accelerated 'diffract' fit implementation needs taichi. "
                'Install it with: pip install "diffract-core[taichi]", '
                "or use fit_method='powerlaw'."
            )
    return _fit_class


# region general

# Display-formula templates for the stacked heavy-tailed summaries; each variant
# substitutes its fit family (PL / TPL / E), which selects the cutoff and rate.
_HT_CONC = r"\#\{i : \lambda_i \ge x_{\min}^{%s}\} / M"
_HT_PRES = r"(\lambda_{\max} - x_{\min}^{%s}) / (\lambda_{\max} - \lambda_{\min})"
_HT_SCALE = r"\lambda_{\max}\,\Lambda_{%s}"


@kernel(
    name="pl_concentration",
    require_fields=("esd", "pl_esd_xmin"),
    info=KernelInfo(formula=_HT_CONC % r"\mathrm{PL}"),
)
@kernel(
    name="tpl_concentration",
    require_fields=("esd", "tpl_esd_xmin"),
    info=KernelInfo(formula=_HT_CONC % r"\mathrm{TPL}"),
)
@kernel(
    name="expon_concentration",
    require_fields=("esd", "expon_esd_xmin"),
    info=KernelInfo(formula=_HT_CONC % r"\mathrm{E}"),
)
def ht_concentration(esd: NDArray[np.floating[Any]], ht_esd_xmin: float) -> float:
    r"""Tail concentration :math:`\#\{i : \lambda_i \ge x_{\min}\} / M`."""
    if np.isnan(ht_esd_xmin) or np.isnan(esd).any():
        return float("nan")

    tail_size = cast("int", np.sum(esd >= ht_esd_xmin).item())

    return tail_size / esd.size


@kernel(
    name="pl_presence",
    require_fields=("esd_min", "esd_max", "pl_esd_xmin"),
    info=KernelInfo(formula=_HT_PRES % r"\mathrm{PL}"),
)
@kernel(
    name="tpl_presence",
    require_fields=("esd_min", "esd_max", "tpl_esd_xmin"),
    info=KernelInfo(formula=_HT_PRES % r"\mathrm{TPL}"),
)
@kernel(
    name="expon_presence",
    require_fields=("esd_min", "esd_max", "expon_esd_xmin"),
    info=KernelInfo(formula=_HT_PRES % r"\mathrm{E}"),
)
def ht_presence(esd_min: float, esd_max: float, ht_esd_xmin: float) -> float:
    r"""Tail presence (tail width as a fraction of the spectrum width).

    :math:`(\lambda_{\max} - x_{\min}) / (\lambda_{\max} - \lambda_{\min})`
    """
    esd_width = esd_max - esd_min
    ht_width = esd_max - ht_esd_xmin

    return ht_width / esd_width if esd_width > 0 else float("nan")


@kernel(
    name="tpl_scale",
    require_fields=("esd_max", "tpl_lambda"),
    info=KernelInfo(formula=_HT_SCALE % r"\mathrm{TPL}"),
)
@kernel(
    name="expon_scale",
    require_fields=("esd_max", "expon_lambda"),
    info=KernelInfo(formula=_HT_SCALE % r"\mathrm{E}"),
)
def ht_scale(esd_max: float, ht_lambda: float) -> float:
    r"""Tail scale (fit rate acting over the observed range).

    :math:`\lambda_{\max}\,\Lambda`
    """
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
    """Fit power law using diffract's accelerated Taichi implementation."""
    fit = _get_fit_class()(data, "power_law")
    try:
        params = fit.fit_params()
    finally:
        fit.close()

    pl_alpha, pl_esd_xmin, pl_ks = itemgetter("pl_alpha", "xmin", "ks_distance")(params)

    return pl_alpha, pl_esd_xmin, pl_ks


@kernel(produce_fields=("pl_alpha", "pl_esd_xmin", "pl_ks"))
def power_law_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: FitMethod = "auto",
) -> tuple[float, float, float]:
    r"""Fit a power law to the ESD tail via the Clauset-Shalizi-Newman MLE.

    The exponent is the maximum-likelihood estimate
    :math:`\hat{\alpha} = 1 + n_{\mathrm{tail}}\,\big/\sum_i \ln(\lambda_i / x_{\min})`,
    with :math:`x_{\min}` chosen to minimise the Kolmogorov-Smirnov distance.
    `auto` uses the accelerated implementation when the taichi extra is
    installed and the ESD is large enough for it to fit at all.
    """
    match _resolve_fit_method(fit_method, esd.size):
        case "powerlaw":
            return power_law_fit_powerlaw_implementation(esd)
        case "diffract":
            return power_law_fit_diffract_implementation(esd)
        case _:
            msg = (
                f"Fit method {fit_method!r} not implemented, choose method "
                "from ('auto', 'powerlaw', 'diffract')."
            )
            raise ValueError(msg)


def pl_p_value(
    esd: NDArray[np.floating[Any]], pl_alpha: float, pl_esd_xmin: float, pl_ks: float
) -> float:
    r"""Bootstrap p-value for the power law fit (requires the taichi extra).

    :math:`p = \Pr(D^* > D_{\mathrm{PL}})` over synthetic resamples.
    """
    fit = _get_fit_class()(esd, "power_law")
    try:
        fit.set_params(
            xmin=pl_esd_xmin,
            pl_alpha=pl_alpha,
            ks_distance=pl_ks,
            tail_size=cast("int", np.sum(esd >= pl_esd_xmin).item()),
        )
        _, p_value = fit.p_value_test()
    finally:
        fit.close()

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
    """Fit truncated power law using diffract's accelerated Taichi implementation."""
    fit = _get_fit_class()(data, "truncated_power_law")
    try:
        params = fit.fit_params()
    finally:
        fit.close()

    tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks = itemgetter(
        "pl_alpha", "expon_lambda", "xmin", "ks_distance"
    )(params)

    return tpl_alpha, tpl_lambda, tpl_esd_xmin, tpl_ks


@kernel(produce_fields=("tpl_alpha", "tpl_lambda", "tpl_esd_xmin", "tpl_ks"))
def truncated_power_law_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: FitMethod = "auto",
) -> tuple[float, float, float, float]:
    r"""Fit a truncated power law to the ESD tail by maximum likelihood.

    Model :math:`p(\lambda) \propto \lambda^{-\hat{\alpha}}\,e^{-\hat{\Lambda}\lambda}`
    for :math:`\lambda \ge x_{\min}` (power law with an exponential cutoff).
    `auto` uses the accelerated implementation when the taichi extra is
    installed and the ESD is large enough for it to fit at all.
    """
    match _resolve_fit_method(fit_method, esd.size):
        case "powerlaw":
            return truncated_power_law_fit_powerlaw_implementation(esd)
        case "diffract":
            return truncated_power_law_fit_diffract_implementation(esd)
        case _:
            msg = (
                f"Fit method {fit_method!r} not implemented, choose method "
                "from ('auto', 'powerlaw', 'diffract')."
            )
            raise ValueError(msg)


def tpl_p_value(
    esd: NDArray[np.floating[Any]],
    tpl_alpha: float,
    tpl_lambda: float,
    tpl_esd_xmin: float,
    tpl_ks: float,
) -> float:
    r"""Bootstrap p-value for the truncated power law fit (requires the taichi extra).

    :math:`p = \Pr(D^* > D_{\mathrm{TPL}})` over synthetic resamples.
    """
    fit = _get_fit_class()(esd, "truncated_power_law")
    try:
        fit.set_params(
            xmin=tpl_esd_xmin,
            pl_alpha=tpl_alpha,
            expon_lambda=tpl_lambda,
            ks_distance=tpl_ks,
            tail_size=cast("int", np.sum(esd >= tpl_esd_xmin).item()),
        )
        _, p_value = fit.p_value_test()
    finally:
        fit.close()

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
    """Fit exponential distribution using the accelerated implementation."""
    fit = _get_fit_class()(data, "exponential")
    try:
        params = fit.fit_params()
    finally:
        fit.close()

    expon_lambda, expon_esd_xmin, expon_ks = itemgetter(
        "expon_lambda", "xmin", "ks_distance"
    )(params)

    return expon_lambda, expon_esd_xmin, expon_ks


@kernel(produce_fields=("expon_lambda", "expon_esd_xmin", "expon_ks"))
def exponential_fit(
    esd: NDArray[np.floating[Any]],
    *,
    fit_method: FitMethod = "auto",
) -> tuple[float, float, float]:
    r"""Fit an exponential tail to the ESD by maximum likelihood.

    MLE :math:`\hat{\Lambda} = 1 / (\langle\lambda\rangle_{\ge x_{\min}} - x_{\min})`
    (the light-tailed contrast family for the power-law fits).
    `auto` uses the accelerated implementation when the taichi extra is
    installed and the ESD is large enough for it to fit at all.
    """
    match _resolve_fit_method(fit_method, esd.size):
        case "powerlaw":
            return exponential_fit_powerlaw_implementation(esd)
        case "diffract":
            return exponential_fit_diffract_implementation(esd)
        case _:
            msg = (
                f"Fit method {fit_method!r} not implemented, choose method "
                "from ('auto', 'powerlaw', 'diffract')."
            )
            raise ValueError(msg)


def expon_p_value(
    esd: NDArray[np.floating[Any]],
    expon_lambda: float,
    expon_esd_xmin: float,
    expon_ks: float,
) -> float:
    r"""Bootstrap p-value for the exponential fit (requires the taichi extra).

    :math:`p = \Pr(D^* > D_{\mathrm{E}})` over synthetic resamples.
    """
    fit = _get_fit_class()(esd, "exponential")
    try:
        fit.set_params(
            xmin=expon_esd_xmin,
            expon_lambda=expon_lambda,
            ks_distance=expon_ks,
            tail_size=cast("int", np.sum(esd >= expon_esd_xmin).item()),
        )
        _, p_value = fit.p_value_test()
    finally:
        fit.close()

    return p_value


# The p-value kernels are only executable with the accelerated implementation,
# so they join the registry only when it is importable.
if _DIFFRACT_FIT_AVAILABLE:
    kernel(pl_p_value)
    kernel(tpl_p_value)
    kernel(expon_p_value)
