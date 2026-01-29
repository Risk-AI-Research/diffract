"""Optional dependency utilities and lazy import helpers.

Provides consistent helpers to check availability, import optionally with
fallbacks, enforce requirements with helpful errors, and decorate functions
that depend on optional packages.

The module also exposes module-level lazy flags like _IS_TORCH_AVAILABLE via
__getattr__ for convenience without upfront import cost.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

__all__ = [
    "LazyImport",
    "OptionalDependencyError",
    "available",
    "get_module",
    "is_available",
    "lazy_import",
    "optional_import",
    "require",
    "requires_package",
]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""


class LazyImport:
    """Lazy import wrapper that defers module loading until first access.

    Allows importing modules that may not be available at startup time,
    with proper error handling when the module is actually needed.
    """

    def __init__(self, module_name: str) -> None:
        """Initialize lazy import wrapper.

        Args:
            module_name: Name of the module to import lazily.
        """
        self._module_name = module_name
        self._module: ModuleType | None = None
        self._checked = False

    def __getattr__(self, name: str) -> Any:
        """Import module on first attribute access.

        Args:
            name: Attribute name to access from the module.

        Returns:
            Requested attribute from the imported module.

        Raises:
            OptionalDependencyError: If the module cannot be imported.
        """
        if not self._checked:
            self._module = get_module(self._module_name)
            self._checked = True

        if self._module is None:
            msg = (
                f"Module '{self._module_name}' is required but not available. "
                f"Install it with: pip install {self._module_name}"
            )
            raise OptionalDependencyError(msg)

        return getattr(self._module, name)


class _AvailabilityFlags:
    """Container for lazy availability flags."""

    @property
    def _IS_TORCH_AVAILABLE(self) -> bool:  # noqa: N802
        """Check if PyTorch is available."""
        return is_available("torch")

    @property
    def _IS_PANDAS_AVAILABLE(self) -> bool:  # noqa: N802
        """Check if pandas is available."""
        return is_available("pandas")

    @property
    def _IS_REDIS_AVAILABLE(self) -> bool:  # noqa: N802
        """Check if redis is available."""
        return is_available("redis")


_flags = _AvailabilityFlags()


def __getattr__(name: str) -> Any:
    """Provide lazy availability flags as module attributes.

    Supports names like _IS_TORCH_AVAILABLE by delegating to _flags.

    Args:
        name: Attribute name to retrieve.

    Returns:
        Value of the requested attribute.

    Raises:
        AttributeError: If the attribute is not available.
    """
    if hasattr(_flags, name):
        return getattr(_flags, name)
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


@lru_cache(maxsize=128)
def is_available(package: str) -> bool:
    """Check if a package can be imported.

    Args:
        package: Package name to check (e.g., "torch", "pandas").

    Returns:
        True if package is available, False otherwise.

    Example:
        >>> is_available("os")  # Built-in module
        True
        >>> is_available("nonexistent_package")
        False
    """
    try:
        importlib.import_module(package)
    except ImportError:
        return False
    else:
        return True


def get_module(package: str) -> ModuleType | None:
    """Import and return module or None if unavailable.

    Args:
        package: Package name to import.

    Returns:
        Imported module or None if import fails.

    Example:
        >>> module = get_module("os")
        >>> module is not None
        True
    """
    if is_available(package):
        return importlib.import_module(package)
    return None


def optional_import(package: str, *, fallback: Any = None) -> Any:
    """Import a package with optional fallback.

    Returns module or provided fallback if import fails.

    Args:
        package: Package name to import.
        fallback: Value to return if import fails.

    Returns:
        Imported module or fallback value.

    Example:
        >>> result = optional_import("nonexistent", fallback="not found")
        >>> result
        'not found'
    """
    module = get_module(package)
    return module if module is not None else fallback


def require(package: str) -> ModuleType:
    """Import a package or raise OptionalDependencyError.

    Args:
        package: Package name to import.

    Returns:
        Imported module.

    Raises:
        OptionalDependencyError: If package is not available.

    Example:
        >>> os_module = require("os")
        >>> os_module.__name__
        'os'
    """
    if not is_available(package):
        msg = (
            f"Package '{package}' is required but not available. "
            f"Install it with: pip install {package}"
        )
        raise OptionalDependencyError(msg)
    return importlib.import_module(package)


def lazy_import(package: str) -> LazyImport:
    """Create a lazy import wrapper for a package.

    Args:
        package: Package name to import lazily.

    Returns:
        LazyImport wrapper that will import the module on first access.

    Example:
        >>> torch = lazy_import("torch")
        >>> # Module is not imported yet
        >>> tensor = torch.tensor([1, 2, 3])  # Import happens here
    """
    return LazyImport(package)


def available(package: str) -> bool:
    """Alias for is_available for backward compatibility.

    Args:
        package: Package name to check.

    Returns:
        True if package is available, False otherwise.
    """
    return is_available(package)


def requires_package[R](
    package: str, *, fallback_return: R | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator ensuring required package is available before function execution.

    If the package is missing and fallback_return is provided, logs a warning
    and returns fallback_return instead of raising.

    Args:
        package: Required package name.
        fallback_return: Optional fallback value to return if package is missing.

    Returns:
        Decorator function.

    Example:
        >>> @requires_package("torch")
        ... def create_tensor():
        ...     import torch
        ...
        ...     return torch.tensor([1, 2, 3])
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not is_available(package):
                if fallback_return is not None:
                    logger.warning(
                        "Function '%s' requires package '%s' which is not available. "
                        "Returning fallback value.",
                        func.__name__,
                        package,
                    )
                    return fallback_return
                msg = (
                    f"Function '{func.__name__}' requires package '{package}' "
                    f"which is not available. Install it with: pip install {package}"
                )
                raise OptionalDependencyError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
