"""Kernel decorator and registration utilities."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from .exceptions import InconsistentWiring
from .execution import KernelApplyLevel, KernelExecutionProtocol, KernelRestrictions
from .metadata import KernelInfo
from .registry import KernelRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

# Registration manifest for the built-in kernels. Import side effects only
# run once per process, but each fresh registry still needs the built-ins;
# register_default_kernels replays this manifest into the target registry.
_DEFAULT_KERNEL_SPECS: list[dict[str, Any]] = []
_COLLECTING_DEFAULTS = False


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

        spec: dict[str, Any] = {
            "name": final_name,
            "require_fields": final_require,
            "produce_fields": final_produce,
            "implementation": func,
            "apply_level": apply_level,
            "execution_protocol": execution_protocol,
            "restrictions": restrictions,
            "config": cfg,
            "info": info or KernelInfo(),
        }
        registry.register_kernel(**spec)
        if _COLLECTING_DEFAULTS:
            _DEFAULT_KERNEL_SPECS.append(spec)

        return func

    if _func is None:
        return decorator
    return decorator(_func)


def register_default_kernels(registry: KernelRegistry | None = None) -> None:
    """Register the built-in kernels, replaying them into ``registry``.

    The first call imports the kernels subpackage, which registers every
    ``@kernel``-decorated function into the currently wired registry and
    records a manifest of the built-ins. Because module imports are cached
    per process, later calls replay that manifest into ``registry`` so a
    freshly created registry also receives the built-in kernels.

    Args:
        registry: Target registry for the manifest replay. When None, only
            the import side effects apply (first call in the process).
    """
    global _COLLECTING_DEFAULTS  # noqa: PLW0603

    module_name = f"{__package__}.kernels"
    if module_name not in sys.modules:
        _COLLECTING_DEFAULTS = True
        try:
            importlib.import_module(".kernels", package=__package__)
        finally:
            _COLLECTING_DEFAULTS = False

    if registry is not None:
        for spec in _DEFAULT_KERNEL_SPECS:
            if not registry.has_kernel(spec["name"]):
                registry.register_kernel(**spec)
