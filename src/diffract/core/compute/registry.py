"""Core kernel registry implementation.

Provides in-memory registry for kernel management with dependency resolution,
configuration APIs, and metadata management. The registry serves as the central
component for kernel discovery, validation, and execution orchestration.
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any

from ordered_set import OrderedSet

from .config import KernelConfig
from .exceptions import (
    CircularDependencyError,
    DependencyNotFoundError,
    InvalidConfiguration,
)
from .execution import KernelApplyLevel, KernelExecutionProtocol, KernelRestrictions
from .metadata import KernelInfo, KernelMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class KernelRegistry:
    """In-memory registry of kernels with dependency resolution and config APIs.

    Central registry managing kernel registration, metadata storage, dependency
    resolution, and configuration management. Provides APIs for kernel discovery,
    validation, and execution preparation.

    The registry maintains kernel metadata including signatures, dependencies,
    execution protocols, and configuration parameters. It supports dependency
    resolution with circular dependency detection and caching for performance.
    """

    def __init__(self) -> None:
        """Initialize empty registry with metadata storage and resolution cache."""
        self._metadata: dict[str, KernelMetadata] = {}
        self._resolve_cache: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _split_signature(
        func: Callable[..., Any],
    ) -> tuple[tuple[str, ...], KernelConfig]:
        """Split function signature into required fields and default config.

        Analyzes function signature to extract parameter names without defaults
        as required fields, and parameters with defaults as configuration options.

        Args:
            func: Function to analyze.

        Returns:
            Tuple of (required field names, configuration with defaults).

        Raises:
            TypeError: If function signature contains *args or **kwargs.
        """
        sig = inspect.signature(func)

        req: list[str] = []
        defs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                msg = "Kernels may not use *args or **kwargs."
                raise TypeError(msg)

            if param.default is inspect.Parameter.empty:
                req.append(name)
            else:
                defs[name] = param.default

        return tuple(req), KernelConfig(**defs)

    def register_kernel(
        self,
        *,
        name: str,
        require_fields: tuple[str, ...],
        produce_fields: tuple[str, ...],
        implementation: Callable[..., Any],
        apply_level: KernelApplyLevel,
        execution_protocol: KernelExecutionProtocol | None,
        restrictions: KernelRestrictions | None,
        config: KernelConfig,
        info: KernelInfo | None = None,
    ) -> None:
        """Register a kernel and its metadata in the registry.

        Stores complete kernel metadata including implementation, dependencies,
        execution parameters, and configuration. Clears dependency resolution
        cache to ensure fresh dependency analysis.

        Args:
            name: Unique kernel identifier.
            require_fields: Field names required as kernel input.
            produce_fields: Field names produced by kernel.
            implementation: Callable implementing kernel logic.
            apply_level: Execution level (parameter/model/cross-model).
            execution_protocol: Execution strategy (sequential/parallel).
            restrictions: Optional argument restrictions.
            config: Configuration parameters with defaults.
            info: Optional documentation metadata.
        """
        if not isinstance(config, KernelConfig):
            config = KernelConfig(**(config or {}))
        cfg = deepcopy(config)
        info = KernelInfo() if info is None else info
        meta = KernelMetadata(
            name,
            require_fields,
            produce_fields,
            implementation,
            apply_level,
            execution_protocol,
            restrictions,
            cfg,
            info,
        )
        self._metadata[name] = meta
        self._resolve_cache.clear()
        logger.debug(
            "Registered kernel '%s' (requires=%s, produce=%s)",
            name,
            require_fields,
            produce_fields,
        )

    def configure_kernel(self, name: str, conf: KernelConfig) -> None:
        """Update kernel configuration values safely.

        Merges provided configuration with existing kernel configuration,
        validating all parameters against registered defaults.

        Args:
            name: Registered kernel name.
            conf: Configuration updates to apply.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
            InvalidConfiguration: If configuration contains invalid parameters.
        """
        meta = self._get(name)
        meta.config.update(conf)
        logger.debug("Configured kernel '%s' with %s", name, conf.as_dict())

    def _get(self, name: str) -> KernelMetadata:
        """Get kernel metadata by name with error handling.

        Args:
            name: Kernel name to retrieve.

        Returns:
            Kernel metadata.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        try:
            return self._metadata[name]
        except KeyError as e:
            msg = f"Kernel '{name}' not registered"
            raise DependencyNotFoundError(msg) from e

    def list_kernels(self, verbose: bool = False) -> list[str]:
        """List all registered kernels.

        Args:
            verbose: If True, return detailed string representations.

        Returns:
            List of kernel names or detailed representations.
        """
        return sorted(
            [str(m) for m in self._metadata.values()]
            if verbose
            else list(self._metadata.keys())
        )

    def list_fields_can_produce(self, verbose: bool = False) -> list[str]:
        """List all fields that registered kernels can produce.

        Args:
            verbose: If True, annotate each field with its producing kernel.

        Returns:
            Sorted list of field names, or "field <- kernel" strings.
        """
        can_produce_fields: list[str] = []
        for kernel_name, kernel_meta in self._metadata.items():
            can_produce_fields.extend(
                f"{field} <- {kernel_name}" if verbose else field
                for field in kernel_meta.produce_fields
            )
        return sorted(can_produce_fields)

    def has_kernel(self, name: str) -> bool:
        """Check if a kernel is registered.

        Args:
            name: Kernel name to check.

        Returns:
            True if kernel is registered, False otherwise.
        """
        return name in self._metadata

    def can_produce_field(self, name: str) -> bool:
        """Check if any registered kernel can produce a field.

        Args:
            name: Field name to check.

        Returns:
            True if any kernel can produce the field, False otherwise.
        """
        return name in self.list_fields_can_produce()

    def get_kernel_apply_level(self, name: str) -> KernelApplyLevel:
        """Get kernel application level.

        Args:
            name: Registered kernel name.

        Returns:
            Kernel application level.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).apply_level

    def get_kernel_execution_protocol(self, name: str) -> KernelExecutionProtocol:
        """Get kernel execution protocol.

        Args:
            name: Registered kernel name.

        Returns:
            Kernel execution protocol.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).execution_protocol

    def get_fields_kernel_require(self, name: str) -> tuple[str, ...]:
        """Get field names required by a kernel.

        Args:
            name: Registered kernel name.

        Returns:
            Tuple of required field names.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).require_fields

    def get_fields_kernel_produce(self, name: str) -> tuple[str, ...]:
        """Get field names produced by a kernel.

        Args:
            name: Registered kernel name.

        Returns:
            Tuple of produced field names.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).produce_fields

    def get_kernel_config(self, name: str) -> dict[str, Any]:
        """Get kernel configuration as dictionary.

        Args:
            name: Registered kernel name.

        Returns:
            Dictionary of configuration parameters and values.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).config.as_dict()

    def get_kernel_restrictions(self, name: str) -> KernelRestrictions:
        """Get kernel argument restrictions.

        Args:
            name: Registered kernel name.

        Returns:
            Kernel restrictions flags.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).restrictions

    def get_kernel_implementation(self, name: str) -> Callable[..., Any]:
        """Get kernel implementation wrapped with configuration injection.

        Returns a wrapped version of the kernel implementation that automatically
        injects configuration parameters as keyword arguments during execution.

        Args:
            name: Registered kernel name.

        Returns:
            Wrapped kernel implementation with config injection.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        impl = self._get(name).implementation
        cfg = self.get_kernel_config(name)

        @wraps(impl)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            merged = {**cfg, **kwargs}
            return impl(*args, **merged)

        return wrapped

    def get_kernel_info(self, name: str) -> KernelInfo:
        """Get kernel documentation metadata.

        Args:
            name: Registered kernel name.

        Returns:
            Kernel documentation information.

        Raises:
            DependencyNotFoundError: If kernel is not registered.
        """
        return self._get(name).info

    def resolve_dependencies(
        self, field_or_kernel_name: str, visited: set[str] | None = None
    ) -> tuple[str, ...]:
        """Resolve dependencies producing required fields for a kernel or field name.

        Recursively resolves all kernel dependencies required to produce the
        specified field or execute the specified kernel. Uses caching for
        performance and detects circular dependencies.

        Args:
            field_or_kernel_name: Kernel name or field name to resolve.
            visited: Set of visited kernels for circular dependency detection.

        Returns:
            Tuple of kernel names in dependency order.

        Raises:
            CircularDependencyError: If circular dependency is detected.
            ValueError: If no kernel can produce the specified field.
        """
        if field_or_kernel_name not in self._metadata:
            kernel_name = self.get_kernel_producing_field(
                field_name=field_or_kernel_name
            )
        else:
            kernel_name = field_or_kernel_name

        if kernel_name in self._resolve_cache:
            return self._resolve_cache[kernel_name]

        if visited is None:
            visited = set()

        if kernel_name in visited:
            msg = f"Circular dependency for '{kernel_name}'"
            raise CircularDependencyError(msg)

        visited.add(kernel_name)

        meta = self._get(kernel_name)
        dependencies: OrderedSet[str] = OrderedSet()
        for required in meta.require_fields:
            for other_kernel in self.list_kernels():
                if kernel_name == other_kernel:
                    continue
                other_kernel_meta = self._get(other_kernel)
                if required in other_kernel_meta.produce_fields:
                    dependencies.update(
                        self.resolve_dependencies(other_kernel, visited)
                    )
        dependencies.add(kernel_name)

        result = tuple(dependencies)
        self._resolve_cache[kernel_name] = result
        return result

    def get_kernel_producing_field(self, field_name: str) -> str:
        """Find kernel that produces a specific field.

        Args:
            field_name: Name of the field to find a producer for.

        Returns:
            Name of kernel that produces the field.

        Raises:
            ValueError: If no kernel produces the specified field.
        """
        for kernel in self.list_kernels():
            kernel_meta = self._get(kernel)
            if field_name in kernel_meta.produce_fields:
                return kernel
        raise ValueError

    def resolve_produced_fields(self, field_or_kernel_name: str) -> set[str]:
        """Get all fields that will be produced when computing a field or kernel.

        Resolves the dependency chain and collects all fields produced by
        all kernels in the chain (including intermediate dependencies).

        Args:
            field_or_kernel_name: Field name or kernel name to resolve.

        Returns:
            Set of all field names that will be produced.
        """
        kernel_chain = self.resolve_dependencies(field_or_kernel_name)
        produced: set[str] = set()
        for kernel_name in kernel_chain:
            produced.update(self._get(kernel_name).produce_fields)
        return produced

    def normalize_kernel_result(self, kernel_name: str, result: Any) -> dict[str, Any]:
        """Normalize kernel outputs to a mapping of produced fields to values.

        Converts kernel return values to a standardized dictionary format mapping
        field names to values, handling various return types (dict, tuple, scalar).

        Args:
            kernel_name: Name of kernel that produced the result.
            result: Raw kernel return value.

        Returns:
            Dictionary mapping produced field names to values.

        Raises:
            InvalidConfiguration: If result format doesn't match kernel specification.
            DependencyNotFoundError: If kernel is not registered.
        """
        meta = self._get(kernel_name)

        if isinstance(result, dict):
            return result

        if isinstance(result, tuple):
            if len(result) != len(meta.produce_fields):
                msg = (
                    f"Kernel '{kernel_name}' returned {len(result)} values but "
                    f"produce declares {len(meta.produce_fields)}"
                )
                raise InvalidConfiguration(msg)
            return dict(zip(meta.produce_fields, result, strict=False))

        if len(meta.produce_fields) == 1:
            return {meta.produce_fields[0]: result}

        msg = (
            f"Kernel '{kernel_name}' returned scalar but produce declares multiple "
            "fields: "
            f"{meta.produce_fields}"
        )
        raise InvalidConfiguration(msg)
