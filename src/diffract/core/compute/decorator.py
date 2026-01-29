"""Kernel decorator and registration utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from .exceptions import InconsistentWiring
from .execution import KernelApplyLevel, KernelExecutionProtocol, KernelRestrictions
from .metadata import KernelInfo
from .registry import KernelRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


@inject
def kernel(
    _func: Callable[..., Any] | None = None,
    *,
    registry: KernelRegistry = Provide["kernel_registry"],
    name: str | None = None,
    require_fields: tuple[str, ...] | None = None,
    produce_fields: tuple[str, ...] | None = None,
    apply_level: KernelApplyLevel | None = KernelApplyLevel.PARAMETER,
    execution_protocol: KernelExecutionProtocol
    | None = KernelExecutionProtocol.SEQUENTIAL,
    restrictions: KernelRestrictions | None = None,
    info: KernelInfo | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a compute kernel.

    Registers a function with the kernel registry, enabling it to be used
    in the compute pipeline. The decorator infers required fields and default
    configuration from the function signature unless explicitly provided.

    Args:
        _func: Function to decorate (used internally).
        registry: Kernel registry for registration (injected).
        name: Custom kernel name (defaults to function name).
        require_fields: Input field names (inferred from signature if None).
        produce_fields: Output field names (defaults to kernel name).
        apply_level: Level at which kernel operates.
        execution_protocol: Execution strategy for the kernel.
        restrictions: Optional argument restrictions.
        info: Optional documentation metadata.

    Returns:
        Decorator function or decorated function.

    Raises:
        InconsistentWiring: If registry injection is not properly configured.

    Example:
        >>> @kernel(name="custom_field", produce_fields=("result",))
        ... def my_kernel(input_field: float, param: float = 1.0) -> float:
        ...     return input_field * param
    """
    if not isinstance(registry, KernelRegistry):
        raise InconsistentWiring

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        req_auto, cfg = registry._split_signature(func)  # noqa: SLF001
        final_name = name or func.__name__
        final_require = require_fields or req_auto
        final_produce = produce_fields or (final_name,)

        registry.register_kernel(
            name=final_name,
            require_fields=final_require,
            produce_fields=final_produce,
            implementation=func,
            apply_level=apply_level,
            execution_protocol=execution_protocol,
            restrictions=restrictions,
            config=cfg,
            info=info or KernelInfo(),
        )

        return func

    if _func is None:
        return decorator
    return decorator(_func)


def register_default_kernels() -> None:
    """Import and register default kernels from the kernels subpackage.

    Dynamically imports the kernels submodule, which triggers registration
    of all kernel functions decorated with @kernel in that module.
    This is called during container initialization to populate the registry
    with built-in kernel implementations.
    """
    importlib.import_module(".kernels", package=__package__)
