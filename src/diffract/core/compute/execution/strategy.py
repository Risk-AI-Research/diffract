"""Execution strategy implementations for kernel processing.

Provides strategy pattern implementations for sequential and parallel
kernel execution, abstracting the execution mechanism from kernel runners.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from .enums import KernelExecutionProtocol
from .utils import execute_kernel, marshal_kernel

if TYPE_CHECKING:
    from diffract.core.compute.registry import KernelRegistry


class ExecutionStrategy(ABC):
    """Abstract execution strategy interface."""

    @abstractmethod
    def execute_tasks(
        self,
        kernel_name: str,
        tasks: dict[Any, tuple[Any, ...]],
        implementation: Callable[..., Any],
    ) -> Iterator[tuple[Any, Any]]:
        """Execute tasks and yield (task_id, result) pairs as they complete.

        Args:
            kernel_name: Name of the kernel for error reporting.
            tasks: Dictionary mapping task_id to argument tuples.
            implementation: Kernel function to execute.

        Yields:
            Tuples of (task_id, result) as execution completes.
        """
        raise NotImplementedError


class SequentialStrategy(ExecutionStrategy):
    """Sequential execution strategy.

    Executes tasks one by one in the current process, yielding
    results immediately after each execution completes.
    """

    def execute_tasks(
        self,
        kernel_name: str,
        tasks: dict[Any, tuple[Any, ...]],
        implementation: Callable[..., Any],
    ) -> Iterator[tuple[Any, Any]]:
        """Execute tasks sequentially, yielding results one by one."""
        for task_id, args in tasks.items():
            result = execute_kernel(kernel_name, implementation, args)
            yield task_id, result


class ParallelStrategy(ExecutionStrategy):
    """Parallel execution strategy using ProcessPoolExecutor.

    Distributes tasks across worker processes and yields results
    as they complete (not necessarily in submission order).
    """

    def __init__(self, pool: ProcessPoolExecutor) -> None:
        """Initialize with a process pool.

        Args:
            pool: ProcessPoolExecutor for parallel execution.
        """
        self._pool = pool

    def execute_tasks(
        self,
        kernel_name: str,
        tasks: dict[Any, tuple[Any, ...]],
        implementation: Callable[..., Any],
    ) -> Iterator[tuple[Any, Any]]:
        """Execute tasks in parallel, yielding results as they complete."""
        marshaled = marshal_kernel(implementation)
        future_to_task = {
            self._pool.submit(execute_kernel, kernel_name, marshaled, args): task_id
            for task_id, args in tasks.items()
        }

        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            yield task_id, future.result()


def create_execution_strategy(
    kernel_name: str,
    registry: KernelRegistry,
    process_pool: ProcessPoolExecutor | None,
) -> ExecutionStrategy:
    """Create appropriate execution strategy based on kernel config and pool.

    Args:
        kernel_name: Name of the kernel to execute.
        registry: Kernel registry for protocol lookup.
        process_pool: Optional process pool for parallel execution.

    Returns:
        Configured execution strategy instance.
    """
    if process_pool is None:
        return SequentialStrategy()

    protocol = registry.get_kernel_execution_protocol(kernel_name)
    if protocol == KernelExecutionProtocol.PARALLEL:
        return ParallelStrategy(process_pool)

    return SequentialStrategy()
