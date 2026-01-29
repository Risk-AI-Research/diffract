"""Multiprocessing utilities for kernel execution.

Provides functions for marshaling, unmarshaling, and executing kernels
in multiprocessing contexts with proper serialization handling.

The marshal/unmarshal functions enable kernel functions to be transmitted
across process boundaries for parallel execution, while execute_kernel
provides uniform error handling across all execution modes.
"""

from __future__ import annotations

import marshal
import pickle
import types
from typing import TYPE_CHECKING, Any

from diffract.core.compute.exceptions import KernelExecutionError
from diffract.core.utils.exceptions import format_exception_message

if TYPE_CHECKING:
    from collections.abc import Callable


def marshal_kernel(kernel_implementation: Callable[..., Any]) -> bytes:
    """Serialize a Python function to bytes for cross-process transmission.

    Converts a kernel function into a serialized format suitable for
    transmission across process boundaries in multiprocessing scenarios.
    Captures the function's code, defaults, and closure state.

    Args:
        kernel_implementation: Function to serialize.

    Returns:
        Serialized function as bytes.

    Note:
        The function must be picklable. Closures with non-serializable
        state will fail to serialize.
    """
    kernel_code = kernel_implementation.__code__
    kernel_defaults = kernel_implementation.__defaults__

    kernel_closure_bytes: bytes | None = None
    if kernel_implementation.__closure__:
        kernel_closure = tuple(
            c.cell_contents for c in kernel_implementation.__closure__
        )
        kernel_closure_bytes = pickle.dumps(kernel_closure)

    return marshal.dumps((kernel_code, kernel_defaults, kernel_closure_bytes))


def unmarshal_kernel(serialized: bytes) -> Callable[..., Any]:
    """Reconstruct a Python function from serialized bytes.

    Deserializes a kernel function from bytes back into an executable
    function object with proper code, defaults, and closure restoration.

    Args:
        serialized: Serialized function bytes from marshal_kernel.

    Returns:
        Reconstructed function object.
    """
    kernel_code, kernel_defaults, kernel_closure = marshal.loads(serialized)  # noqa: S302
    kernel_closure = (
        pickle.loads(kernel_closure)  # noqa: S301
        if kernel_closure
        else None
    )
    if kernel_closure:
        kernel_closure = tuple(types.CellType(obj) for obj in kernel_closure)
    return types.FunctionType(
        code=kernel_code,
        globals=globals(),
        name=None,
        argdefs=kernel_defaults,
        closure=kernel_closure,
    )


def execute_kernel(
    kernel_name: str,
    kernel_implementation: Callable[..., Any] | bytes,
    kernel_args: tuple[Any, ...],
) -> Any:
    """Execute a kernel implementation with arguments, handling errors uniformly.

    Executes a kernel function, automatically deserializing if needed,
    and wrapping any exceptions in KernelExecutionError for consistent
    error handling across the execution pipeline.

    Args:
        kernel_name: Name of the kernel for error reporting.
        kernel_implementation: Function or serialized function bytes.
        kernel_args: Arguments to pass to the kernel function.

    Returns:
        Result of kernel execution.

    Raises:
        KernelExecutionError: If kernel execution fails for any reason.
    """
    try:
        if isinstance(kernel_implementation, bytes):
            kernel_implementation = unmarshal_kernel(kernel_implementation)
        return kernel_implementation(*kernel_args)
    except Exception as e:
        msg = f"Kernel '{kernel_name}' execution failed: {format_exception_message(e)}"
        raise KernelExecutionError(msg) from e
