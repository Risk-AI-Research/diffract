"""Compute namespace for Session."""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from diffract.core.compute import KernelConfig, KernelExecutor, KernelRegistry
from diffract.core.compute.execution import (
    KernelApplyLevel,
    KernelExecutionProtocol,
    KernelRestrictions,
)
from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.utils.exceptions import format_exception_message
from diffract.session.errors import KernelNotFoundError, SessionError
from diffract.session.session import Session, SessionContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffract.core.data.nn.params.schema import ParameterType

logger = logging.getLogger(__name__)


def _did_you_mean(name: str, candidates: list[str]) -> str:
    close = difflib.get_close_matches(name, sorted(set(candidates)), n=3, cutoff=0.6)
    return f" Did you mean: {', '.join(close)}?" if close else ""


class ComputeNamespace:
    """Kernel compute API for Session."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        kernel_registry: KernelRegistry = Provide["compute.kernel_registry"],
        kernel_executor_factory: Callable[[], KernelExecutor] = Provide[
            "compute.kernel_executor.provider"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__kernel_registry = kernel_registry
        self.__kernel_executor_factory = kernel_executor_factory

    def apply(
        self,
        *fields_to_produce: str,
    ) -> None:
        """Apply computational kernels to stored parameters.

        Executes the specified kernels on filtered parameters in dependency order.
        Results are automatically stored and can be retrieved using export_metrics().

        Args:
            *fields_to_produce: Names of fields to compute using registered kernels.

        Raises:
            KernelNotFoundError: If any specified field cannot be produced.
            SessionError: If kernel execution fails.
        """
        with self.__session_or_context:
            if not fields_to_produce:
                logger.warning("No fields specified for computation")
                return

            for field_name in fields_to_produce:
                if not self.__kernel_registry.can_produce_field(field_name):
                    available = self.__kernel_registry.list_fields_can_produce()
                    hint = _did_you_mean(field_name, available)
                    msg = (
                        f"Cannot produce '{field_name}': no registered kernel "
                        f"produces it.{hint} Use "
                        "session.compute.list_available_metrics() to see all fields."
                    )
                    raise KernelNotFoundError(msg)

            if isinstance(self.__session_or_context, Session):
                pending = self.__param_repo.create_view()
            else:
                pending = self.__session_or_context._param_filter_context

            try:
                with self.__kernel_executor_factory() as executor:
                    for field_name in fields_to_produce:
                        executor.execute(
                            field_or_kernel_name=field_name, parameters=pending
                        )

                logger.info(
                    "Successfully produced fields: %s", ", ".join(fields_to_produce)
                )

                self.__session_or_context._field_cache.invalidate()

            except SessionError:
                raise
            except Exception as e:
                msg = f"Failed to produce fields: {format_exception_message(e)}"
                raise SessionError(msg) from e

    def list_available_kernels(self, verbose: bool = False) -> list[str]:
        """List all available computational kernels.

        Args:
            verbose: If True, include detailed kernel information including
                dependencies and field requirements.

        Returns:
            List of kernel names or detailed kernel descriptions.
        """
        with self.__session_or_context:
            return self.__kernel_registry.list_kernels(verbose=verbose)

    def list_available_metrics(self, verbose: bool = False) -> list[str]:
        """List all fields that can be computed by registered kernels.

        Args:
            verbose: If True, include detailed field information including
                producing kernels and dependencies.

        Returns:
            List of computable field names or detailed field descriptions.
        """
        with self.__session_or_context:
            return self.__kernel_registry.list_fields_can_produce(verbose=verbose)

    def configure_kernel(self, kernel_name: str, **config: Any) -> None:
        """Configure parameters for a specific kernel.

        Updates the configuration for a registered kernel, affecting
        subsequent computations using that kernel.

        Args:
            kernel_name: Name of kernel to configure.
            **config: Kernel configuration parameters as keyword arguments.

        Raises:
            KernelNotFoundError: If kernel is not registered in the registry.
        """
        with self.__session_or_context:
            if not self.__kernel_registry.has_kernel(kernel_name):
                hint = _did_you_mean(kernel_name, self.__kernel_registry.list_kernels())
                msg = f"Kernel '{kernel_name}' not found in registry.{hint}"
                raise KernelNotFoundError(msg)

            kernel_config = KernelConfig(**config)
            self.__kernel_registry.configure_kernel(kernel_name, kernel_config)

            logger.info("Configured kernel '%s' with: %s", kernel_name, config)

    def kernel(
        self,
        _func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        require_fields: tuple[str, ...] | None = None,
        produce_fields: tuple[str, ...] | None = None,
        apply_level: KernelApplyLevel = KernelApplyLevel.PARAMETER,
        execution_protocol: KernelExecutionProtocol = (
            KernelExecutionProtocol.SEQUENTIAL
        ),
        restrictions: KernelRestrictions | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering custom kernels in the session registry.

        Creates a kernel that can compute metrics on neural network parameters.
        Dependencies are automatically inferred from function arguments.

        Args:
            name: Custom kernel name (defaults to function name).
            require_fields: Input field names (inferred from signature if None).
            produce_fields: Output field names (defaults to kernel name).
            apply_level: Granularity at which the kernel is applied.
            execution_protocol: Execution protocol for the kernel.
            restrictions: Optional restrictions applied to kernel execution.

        Returns:
            Decorator function for registering the kernel.

        Example:
            >>> session = Session()
            >>> with session:
            ...
            ...     @session.compute.kernel()
            ...     def my_metric(frob_norm: float, *, scale: float = 1.0) -> float:
            ...         return frob_norm * scale
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            req_auto, cfg = self.__kernel_registry._split_signature(func)
            final_name = name or func.__name__
            final_require = require_fields or req_auto
            final_produce = produce_fields or (final_name,)

            self.__kernel_registry.register_kernel(
                name=final_name,
                require_fields=final_require,
                produce_fields=final_produce,
                implementation=func,
                apply_level=apply_level,
                execution_protocol=execution_protocol,
                restrictions=restrictions,
                config=cfg,
                info=None,
            )
            return func

        if _func is None:
            return decorator

        return decorator(_func)
