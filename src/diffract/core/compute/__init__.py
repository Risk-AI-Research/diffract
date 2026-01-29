"""Compute subsystem public API and shortcuts.

This package exposes the kernel registry and executor along with common
enumerations and decorators used across the compute pipeline.

Example:
    >>> from diffract.core.compute import KernelExecutor, kernel
    >>> with KernelExecutor(registry) as ex:
    ...     ex.execute("my_kernel", params)
"""

from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
    KernelRestrictions,
)
from diffract.core.compute.execution.executor import KernelExecutor

from .config import KernelConfig
from .decorator import kernel, register_default_kernels
from .exceptions import (
    CircularDependencyError,
    DependencyNotFoundError,
    InconsistentWiring,
    InvalidConfiguration,
    KernelError,
    KernelExecutionError,
)
from .registry import KernelRegistry

__all__ = [
    "CircularDependencyError",
    "DependencyNotFoundError",
    "InconsistentWiring",
    "InvalidConfiguration",
    "KernelApplyLevel",
    "KernelConfig",
    "KernelError",
    "KernelExecutionError",
    "KernelExecutionProtocol",
    "KernelExecutor",
    "KernelRegistry",
    "KernelRestrictions",
    "kernel",
    "register_default_kernels",
]
