"""Parameter-level kernel execution logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .restrictions import apply_restrictions_filter
from .strategy import create_execution_strategy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import ProcessPoolExecutor

    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.parallel import ParallelContext

logger = logging.getLogger(__name__)


class ParameterKernelRunner:
    """Executes parameter-level kernels with batching and streaming writes.

    Handles execution of kernels that operate on individual parameters,
    including memory-aware chunking, prefetching, and result streaming.
    """

    def __init__(
        self,
        registry: KernelRegistry,
        process_pool: ProcessPoolExecutor | None,
        parallel: ParallelContext | None,
    ) -> None:
        """Initialize the parameter kernel runner.

        Args:
            registry: Kernel registry for metadata lookup.
            process_pool: Optional process pool for parallel execution.
            parallel: Optional parallel context for view operations.
        """
        self._registry = registry
        self._process_pool = process_pool
        self._parallel = parallel

    def run(self, kernel_name: str, parameters: IParameterView) -> None:
        """Execute kernel on individual parameters.

        Args:
            kernel_name: Name of the registered kernel.
            parameters: Parameter collection to process.
        """
        target_fields = self._registry.get_fields_kernel_produce(kernel_name)
        pending = parameters.filter_by_fields(
            *target_fields,
            inverse_mask=True,
            parallel=self._parallel,
        )
        if not pending:
            logger.debug(
                "Skip execution of kernel '%s': no pending parameters", kernel_name
            )
            return

        logger.info("Executing kernel '%s'", kernel_name)

        required_fields = list(self._registry.get_fields_kernel_require(kernel_name))
        required_by_uid = {p.meta.uid: required_fields for p in pending}

        chunks = list(
            pending.iter_chunks_by_read_budget(
                required_fields_by_uid=required_by_uid,
                parallel=self._parallel,
            )
        )
        if not chunks:
            return

        for chunk in chunks:
            chunk.prefetch_fields(fields=required_fields, parallel=self._parallel)
            self._execute_batch(kernel_name, chunk)

    def _execute_batch(self, kernel_name: str, batch: IParameterView) -> None:
        """Execute parameter-level kernel on a batch with streaming writes.

        Results are written immediately as they become available, reducing
        memory pressure for heavy computations (e.g., SVD matrices).
        """
        required_args = self._registry.get_fields_kernel_require(kernel_name)
        tasks: dict[tuple[str, str], tuple[Any, ...]] = {}

        for param in batch:
            args = tuple(param.get_field(arg) for arg in required_args)
            tasks[(param.meta.model_id, param.meta.uid)] = args

        apply_restrictions_filter(
            kernel_name,
            tasks,
            self._registry.get_kernel_restrictions(kernel_name),
        )
        if not tasks:
            return

        param_lookup = {(p.meta.model_id, p.meta.uid): p for p in batch}

        with batch:
            for key, result in self._stream_results(kernel_name, tasks):
                normalized = self._registry.normalize_kernel_result(kernel_name, result)
                param = param_lookup[key]
                for field_name, value in normalized.items():
                    param.set_field(field_name, value)

    def _stream_results(
        self,
        kernel_name: str,
        tasks: dict[Any, tuple[Any, ...]],
    ) -> Iterator[tuple[Any, Any]]:
        """Execute kernel tasks and yield results as they become available."""
        strategy = create_execution_strategy(
            kernel_name,
            self._registry,
            self._process_pool,
        )
        implementation = self._registry.get_kernel_implementation(kernel_name)
        yield from strategy.execute_tasks(kernel_name, tasks, implementation)
