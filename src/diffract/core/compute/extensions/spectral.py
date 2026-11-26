"""Real-valued weight decompositions and spectral comparisons.

CPU decompositions use LAPACK GESDD and CUDA decompositions use cuSOLVER GESVD.
Both return the economy factors in ascending singular-value order. The preserve
policy promotes float16 inputs to float32 and retains higher input precision.
See :mod:`diffract.core.compute.extensions.svd_workspace` for the CPU memory budget.
"""

from __future__ import annotations

from functools import partial
from math import hypot, isfinite
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import linalg

from diffract.core.compute.config import KernelResult
from diffract.core.compute.exceptions import InvalidConfiguration
from diffract.core.utils.imports import optional_import

from .svd_workspace import require_svd_workspace
from .utils import torch_cuda_wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type SVDOutput = Literal["factors", "values", "vectors"]


def bind_svd(
    requested: dict[str, Any], *, output: SVDOutput
) -> tuple[dict[str, str], Callable[..., KernelResult]]:
    """Bind the cache identity and the calculation to the same solver settings."""
    precision = requested["precision"]
    if precision not in ("preserve", "float32", "float64"):
        raise InvalidConfiguration(
            "SVD precision must be 'preserve', 'float32' or 'float64'."
        )
    if not isinstance(requested["allow_cuda"], bool):
        raise InvalidConfiguration("SVD allow_cuda must be True or False.")
    torch = optional_import("torch") if requested["allow_cuda"] else None
    if (
        torch is not None
        and torch.version.cuda is not None
        and torch.cuda.is_available()
    ):
        device = f"cuda:{torch.cuda.current_device()}"
        backend, driver = "torch", "gesvd"
    else:
        device, backend, driver = "cpu", "scipy", "gesdd"
    settings = {"device": device, "backend": backend, "driver": driver}
    return settings, partial(_decompose, precision=precision, output=output, **settings)


def _decompose(
    matrix: np.ndarray,
    *,
    output: SVDOutput,
    precision: str,
    device: str,
    backend: str,
    driver: str,
) -> KernelResult:
    input_dtype = matrix.dtype
    if input_dtype.kind != "f" or input_dtype.itemsize not in (2, 4, 8):
        raise ValueError(
            f"SVD requires a float16, float32 or float64 matrix, got {input_dtype}."
        )
    dtype = (
        np.promote_types(input_dtype, np.float32)
        if precision == "preserve"
        else np.dtype(precision)
    )
    vectors = output != "values"
    if device == "cpu":
        require_svd_workspace(matrix, vectors=vectors)

    try:
        with np.errstate(over="raise"):
            working = np.asarray(matrix, dtype=dtype)
    except FloatingPointError as exc:
        raise ValueError(
            f"SVD input exceeds the range of {dtype}. Select float64."
        ) from exc
    solver = "cuSOLVER" if backend == "torch" else "LAPACK"
    observed_driver: str | None = driver
    if device == "cpu":
        try:
            result = linalg.svd(
                working, full_matrices=False, compute_uv=vectors, lapack_driver=driver
            )
        except (ValueError, np.linalg.LinAlgError) as err:
            if isinstance(err, ValueError):
                if np.isfinite(working).all():
                    raise
                solver = observed_driver = None
            k = min(working.shape)
            spectrum = np.full(k, np.nan, dtype=dtype)
            result = (
                (
                    np.full((working.shape[0], k), np.nan, dtype=dtype),
                    spectrum,
                    np.full((k, working.shape[1]), np.nan, dtype=dtype),
                )
                if vectors
                else spectrum
            )
    else:
        torch = optional_import("torch")
        solve = (
            partial(torch.linalg.svd, full_matrices=False, driver=driver)
            if vectors
            else partial(torch.linalg.svdvals, driver=driver)
        )
        result = torch_cuda_wrapper(working, solve, device=device)
    if vectors:
        u, spectrum, vt = result
        order = np.argsort(spectrum)
        left, right = u[:, order], vt.T[:, order]
        values = (
            (left, spectrum[order], right) if output == "factors" else (left, right)
        )
    else:
        spectrum = result
        values = np.sort(spectrum)
    return KernelResult(
        values,
        input_dtype=str(input_dtype),
        working_dtype=str(working.dtype),
        output_dtype=str(spectrum.dtype),
        backend=backend,
        device=device,
        solver=solver,
        driver=observed_driver,
    )


def relative_norm_change(
    source: NDArray[np.floating[Any]], target: NDArray[np.floating[Any]]
) -> float:
    """Compute the norm of the difference relative to the source in bounded memory."""
    if source.shape != target.shape:
        raise ValueError(
            "Relative change requires matching shapes, "
            f"got {source.shape} and {target.shape}."
        )
    if not source.size:
        return float("nan")
    scale = max(
        abs(float(source.min())),
        abs(float(source.max())),
        abs(float(target.min())),
        abs(float(target.max())),
    )
    if scale == 0 or not isfinite(scale):
        return float("nan")

    reference_norm = change_norm = 0.0
    # Scale and subtract in blocks to avoid full-sized float64 copies of weights.
    for source_block, target_block in np.nditer(
        (source, target),
        flags=["external_loop", "buffered"],
        op_dtypes=[np.float64, np.float64],
        buffersize=65_536,
    ):
        reference = source_block / scale
        delta = target_block / scale - reference
        reference_norm = hypot(
            reference_norm, linalg.norm(reference, check_finite=False)
        )
        change_norm = hypot(change_norm, linalg.norm(delta, check_finite=False))
    return change_norm / reference_norm if reference_norm else float("nan")
