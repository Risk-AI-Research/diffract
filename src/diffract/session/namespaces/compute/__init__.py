"""Compute namespace for Session."""

from __future__ import annotations

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
from diffract.session.errors import (
    KernelNotFoundError,
    ScopeValidationError,
    SessionError,
)
from diffract.session.session import Session, SessionContext
from diffract.session.summaries import ApplySummary
from diffract.session.utils import did_you_mean

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.schema import ParameterType

logger = logging.getLogger(__name__)

_BINARY_MODEL_COUNT = 2


def _binary_scope_message(
    field_name: str, kernel_name: str, scope_models: list[str]
) -> str:
    """Actionable message for a binary cross-model kernel on a non-pair scope."""
    if kernel_name == field_name:
        clause = "it is a binary cross-model kernel that"
    else:
        clause = f"its dependency '{kernel_name}' is a binary cross-model kernel that"

    example = scope_models[:_BINARY_MODEL_COUNT] or scope_models
    return (
        f"Cannot produce '{field_name}': {clause} requires exactly two models "
        f"in scope, but the current scope has {len(scope_models)}: {scope_models}. "
        f"Restrict the scope to a model pair, e.g. "
        f"session.filter(model_ids={example}).compute.apply('{field_name}')."
    )


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
    ) -> ApplySummary:
        """Apply computational kernels to stored parameters.

        Executes the specified kernels on filtered parameters in dependency order.
        Results are automatically stored and can be retrieved using export_metrics().

        Args:
            *fields_to_produce: Names of fields to compute using registered kernels.

        Returns:
            An ApplySummary grouping the written fields by apply level (which
            exporter now holds them) and listing any requested field that
            produced nothing.

        Raises:
            KernelNotFoundError: If any specified field cannot be produced.
            ScopeValidationError: If the active scope is incompatible with a
                required kernel's apply level (e.g. a binary cross-model kernel
                with other than two models in scope).
            SessionError: If kernel execution fails.
        """
        with self.__session_or_context:
            if not fields_to_produce:
                logger.warning("No fields specified for computation")
                return ApplySummary()

            for field_name in fields_to_produce:
                if not self.__kernel_registry.can_produce_field(field_name):
                    available = self.__kernel_registry.list_fields_can_produce()
                    hint = did_you_mean(field_name, available)
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

            self._validate_scope(fields_to_produce, pending)

            try:
                written: set[str] = set()
                with self.__kernel_executor_factory() as executor:
                    for field_name in fields_to_produce:
                        written |= executor.execute(
                            field_or_kernel_name=field_name, parameters=pending
                        )

                produced = [f for f in fields_to_produce if f in written]
                if produced:
                    logger.info("Successfully produced fields: %s", ", ".join(produced))
                else:
                    logger.info(
                        "No new fields produced for: %s (already computed or not "
                        "applicable to the current scope)",
                        ", ".join(fields_to_produce),
                    )

                self.__session_or_context._field_cache.invalidate()

                return self._summarize_apply(fields_to_produce, written)

            except SessionError:
                raise
            except Exception as e:
                msg = f"Failed to produce fields: {format_exception_message(e)}"
                raise SessionError(msg) from e

    def _summarize_apply(
        self, requested: tuple[str, ...], written: set[str]
    ) -> ApplySummary:
        """Group written fields by apply level; list requested fields skipped."""
        buckets: dict[KernelApplyLevel, list[str]] = {
            KernelApplyLevel.PARAMETER: [],
            KernelApplyLevel.IN_MODEL: [],
            KernelApplyLevel.CROSS_MODEL: [],
        }
        for name in sorted(written):
            kernel = self.__kernel_registry.get_kernel_producing_field(name)
            buckets[self.__kernel_registry.get_kernel_apply_level(kernel)].append(name)
        skipped = tuple(
            (name, "no new field written (already computed or out of scope)")
            for name in requested
            if name not in written
        )
        return ApplySummary(
            parameter_fields=tuple(buckets[KernelApplyLevel.PARAMETER]),
            in_model_fields=tuple(buckets[KernelApplyLevel.IN_MODEL]),
            cross_model_fields=tuple(buckets[KernelApplyLevel.CROSS_MODEL]),
            skipped=skipped,
        )

    def _validate_scope(
        self, fields_to_produce: tuple[str, ...], pending: IParameterView
    ) -> None:
        """Reject a scope incompatible with a required kernel's apply level.

        A binary cross-model kernel operates on a model pair. Producing a field
        whose dependency chain includes such a kernel with any other number of
        models in scope would silently write nothing, so it is rejected up front
        with an actionable error rather than allowed to no-op.
        """
        offenders = [
            (field_name, kernel_name)
            for field_name in fields_to_produce
            for kernel_name in self.__kernel_registry.resolve_dependencies(field_name)
            if self._is_binary_cross_model(kernel_name)
        ]
        if not offenders:
            return

        scope_models = sorted({p.meta.model_id for p in pending})
        if len(scope_models) == _BINARY_MODEL_COUNT:
            return

        field_name, kernel_name = offenders[0]
        raise ScopeValidationError(
            _binary_scope_message(field_name, kernel_name, scope_models)
        )

    def _is_binary_cross_model(self, kernel_name: str) -> bool:
        """True if the kernel is a binary cross-model kernel."""
        if (
            self.__kernel_registry.get_kernel_apply_level(kernel_name)
            != KernelApplyLevel.CROSS_MODEL
        ):
            return False
        restrictions = self.__kernel_registry.get_kernel_restrictions(kernel_name)
        return bool(restrictions and (restrictions & KernelRestrictions.BINARY))

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

    def configure_kernel(self, kernel_name: str, **config: Any) -> KernelConfig:
        """Configure parameters for a specific kernel.

        Updates the configuration for a registered kernel, affecting
        subsequent computations using that kernel.

        Args:
            kernel_name: Name of kernel to configure.
            **config: Kernel configuration parameters as keyword arguments.

        Returns:
            The effective KernelConfig applied to the kernel.

        Raises:
            KernelNotFoundError: If kernel is not registered in the registry.
        """
        with self.__session_or_context:
            if not self.__kernel_registry.has_kernel(kernel_name):
                hint = did_you_mean(kernel_name, self.__kernel_registry.list_kernels())
                msg = f"Kernel '{kernel_name}' not found in registry.{hint}"
                raise KernelNotFoundError(msg)

            kernel_config = KernelConfig(**config)
            self.__kernel_registry.configure_kernel(kernel_name, kernel_config)

            logger.info("Configured kernel '%s' with: %s", kernel_name, config)

            return kernel_config

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
