"""Kernel execution engine and orchestration utilities.

Provides the main KernelExecutor class that orchestrates kernel execution
with dependency resolution, batching, and parallel processing support.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Self

from .aggregation_runner import AggregationKernelRunner
from .enums import KernelApplyLevel
from .parameter_runner import ParameterKernelRunner

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates import AggregateRepository
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.parallel import ParallelContext

    from .aggregation import CoordinatePairs
    from .traces import FieldStatus

logger = logging.getLogger(__name__)


class KernelExecutor:
    """Kernel execution orchestrator with dependency resolution and batching.

    Main execution engine that coordinates kernel execution across parameter
    collections with support for dependency resolution, batching, and parallel
    processing. Handles both individual parameter processing and aggregated
    computations across models.

    The executor delegates actual kernel execution to specialized runners:
    - ParameterKernelRunner: handles PARAMETER-level kernels
    - AggregationKernelRunner: handles IN_MODEL and CROSS_MODEL kernels

    Memory constraints are handled via budget-based chunking in the runners.
    """

    def __init__(
        self,
        registry: KernelRegistry,
        process_pool: ProcessPoolExecutor | None = None,
        parallel: ParallelContext | None = None,
        aggregate_repository: AggregateRepository | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize kernel executor.

        Args:
            registry: Kernel registry containing registered kernels.
            process_pool: Optional process pool for parallel kernel execution.
            parallel: Optional parallel context for parameter view operations.
            aggregate_repository: Optional repository for storing aggregation results.
            **_kwargs: Additional config arguments (ignored, for DI compatibility).
        """
        self._registry = registry
        self._executed: set[str] = set()

        self._parameter_runner = ParameterKernelRunner(
            registry=registry,
            process_pool=process_pool,
            parallel=parallel,
        )
        self._aggregation_runner = AggregationKernelRunner(
            registry=registry,
            process_pool=process_pool,
            parallel=parallel,
            aggregate_repository=aggregate_repository,
        )

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, clearing execution state."""
        self.clear_execution_state()

    def clear_execution_state(self) -> None:
        """Release the grouping snapshot and reset the executed kernels."""
        self._executed.clear()
        self._aggregation_runner.clear_execution_state()

    def execute(
        self,
        field_or_kernel_name: str,
        parameters: IParameterView,
        *,
        forced_kernels: frozenset[str] = frozenset(),
        pairs: CoordinatePairs | None = None,
    ) -> set[str]:
        """Execute a kernel or produce a field with dependency resolution.

        Main entry point for kernel execution. Resolves all dependencies
        and executes required kernels in proper order.

        Args:
            field_or_kernel_name: Kernel name or field name to execute/produce.
            parameters: Parameter collection to process.
            forced_kernels: Kernels to execute even when their outputs are fresh.
            pairs: Ordered cross-model coordinates selected by the caller.

        Returns:
            The set of field names actually written across the resolved
            dependency chain (empty when everything was already computed).
        """
        if field_or_kernel_name in self._executed:
            return set()

        dependencies = self._registry.resolve_dependencies(field_or_kernel_name)

        written: set[str] = set()
        for dep in dependencies:
            if dep != field_or_kernel_name and dep not in self._executed:
                written |= self.execute(
                    dep, parameters, forced_kernels=forced_kernels, pairs=pairs
                )

        if self._registry.has_kernel(field_or_kernel_name):
            written |= self._dispatch_kernel(
                field_or_kernel_name,
                parameters,
                force=field_or_kernel_name in forced_kernels,
                pairs=pairs,
            )
            self._executed.add(field_or_kernel_name)
        return written

    def _dispatch_kernel(
        self,
        kernel_name: str,
        parameters: IParameterView,
        *,
        force: bool,
        pairs: CoordinatePairs | None,
    ) -> set[str]:
        """Dispatch kernel execution to the appropriate runner."""
        apply_level = self._registry.get_kernel_apply_level(kernel_name)

        if apply_level == KernelApplyLevel.PARAMETER:
            return self._parameter_runner.run(kernel_name, parameters, force=force)
        if apply_level in (KernelApplyLevel.IN_MODEL, KernelApplyLevel.CROSS_MODEL):
            return self._aggregation_runner.run(
                kernel_name, parameters, force=force, pairs=pairs
            )
        msg = f"Unknown kernel apply level: {apply_level}"
        raise RuntimeError(msg)

    def inspect(
        self, field: str, parameters: IParameterView
    ) -> tuple[FieldStatus, ...]:
        """Describe stored output freshness for one field without executing kernels."""
        kernel = self._registry.get_kernel_producing_field(field)
        level = self._registry.get_kernel_apply_level(kernel)
        if level == KernelApplyLevel.PARAMETER:
            return self._parameter_runner.inspect(kernel, field, parameters)
        return self._aggregation_runner.inspect(kernel, field, parameters)
