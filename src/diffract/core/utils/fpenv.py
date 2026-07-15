"""Preservation of the floating-point environment across native initializers.

Some native runtimes enable flush-to-zero (FTZ) and denormals-are-zero (DAZ)
on the calling thread when they initialize; taichi's LLVM CPU backend is one of
them. These are CPU control-register flags, not library state: once set, every
subsequent floating-point operation on that thread rounds subnormal results to
zero, including plain numpy and scipy work that has nothing to do with the
runtime that set them.

For spectral analysis the effect is not cosmetic. Squared singular values of a
small-magnitude layer land in the subnormal range -- below ~1.18e-38 in float32,
below ~2.2e-308 in float64 -- so under FTZ a layer with genuinely tiny but
nonzero weights reports a Frobenius norm of exactly zero and is indistinguishable
from a dead layer. The flags are also per-thread, so leaving them set makes a
result depend on whether a kernel happened to run on the initializing thread or
on a worker.

``fegetenv``/``fesetenv`` are the C99 interface to that state. glibc exports them
from libm; macOS exports them from libc. Capture and restore must happen on the
same thread that runs the initializer.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Iterator
from contextlib import contextmanager

# Opaque fenv_t: 8 bytes on arm64, larger elsewhere. Deliberately oversized.
_FENV_STORAGE_BYTES = 64


def _fenv_library() -> ctypes.CDLL | None:
    """Load the C library exposing the C99 floating-point environment calls.

    glibc keeps ``fegetenv``/``fesetenv`` in libm; macOS exports them from libc.
    The first loadable library that carries both symbols is returned.

    Returns:
        The loaded library, or None when neither exposes the calls.
    """
    for lib_name in ("m", "c"):
        name = ctypes.util.find_library(lib_name)
        if name is None:
            continue
        try:
            lib = ctypes.CDLL(name)
        except OSError:  # noqa: S112 -- fall through to the next candidate library
            continue
        if hasattr(lib, "fegetenv") and hasattr(lib, "fesetenv"):
            return lib
    return None


@contextmanager
def preserved_fp_environment() -> Iterator[None]:
    """Restore the calling thread's floating-point control state on exit.

    Wrap any native runtime initializer that may enable flush-to-zero. Where
    libc does not expose the C99 calls the body still runs and the environment
    is left untouched.

    Yields:
        None.
    """
    lib = _fenv_library()
    if lib is None:
        yield
        return

    saved = (ctypes.c_byte * _FENV_STORAGE_BYTES)()
    if lib.fegetenv(ctypes.byref(saved)) != 0:
        yield
        return

    try:
        yield
    finally:
        lib.fesetenv(ctypes.byref(saved))
