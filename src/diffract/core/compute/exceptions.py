"""Kernel-related exception classes."""

from __future__ import annotations


class KernelError(Exception):
    """Base class for kernel-related errors."""


class DependencyNotFoundError(KernelError):
    """Raised when a dependency is not found in the registry."""


class CircularDependencyError(KernelError):
    """Raised when a circular dependency is detected among kernels."""


class InvalidConfigurationError(KernelError):
    """Raised when kernel configuration is invalid."""


class InconsistentWiringError(KernelError):
    """Raised when dependency injection wiring is inconsistent."""


class KernelExecutionError(KernelError):
    """Raised when a kernel implementation raises during execution."""


InvalidConfiguration = InvalidConfigurationError
InconsistentWiring = InconsistentWiringError
