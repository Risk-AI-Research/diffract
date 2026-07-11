"""Runtime helpers for parallel execution decisions.

This module intentionally does not depend on dependency-injector. It provides
small, testable building blocks that can be wired via DI at the application
level.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelCalibration:
    """Thread-pool overhead calibration data."""

    workers: int
    submit_overhead_per_task_s: float


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelContext:
    """Per-method parallel execution context."""

    executor: Any
    calibration: ParallelCalibration
    workers: int


def calibrate_thread_pool_overhead(
    *, submit: Callable[[Callable[[int], int], int], object], workers: int
) -> ParallelCalibration:
    """Measure approximate per-task overhead for a running thread pool.

    Args:
        submit: Callable with signature like executor.submit(fn, arg).
        workers: Configured worker count for the pool.

    Returns:
        Calibration object. The overhead is measured by submitting a small batch
        of no-op tasks and waiting for completion.
    """

    def _noop(_: int) -> int:
        return 0

    n = max(1, workers * (workers + 1))
    t0 = time.perf_counter()
    futures = [submit(_noop, i) for i in range(n)]
    for fut in futures:
        # Executor API: future-like object with result()
        _ = fut.result()
    dt = max(0.0, time.perf_counter() - t0)
    overhead = dt / n if n > 0 else 0.0
    return ParallelCalibration(workers=workers, submit_overhead_per_task_s=overhead)


@functools.lru_cache(maxsize=32)
def get_thread_pool_calibration(workers: int) -> ParallelCalibration:
    """Get cached overhead calibration for a worker count."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return calibrate_thread_pool_overhead(
            submit=executor.submit,
            workers=workers,
        )


def should_parallelize(
    *,
    avg_task_s: float,
    n_tasks: int,
    workers: int,
    overhead_per_task_s: float,
) -> bool:
    """Return True if parallel execution is expected to be faster."""
    if n_tasks <= 1 or workers <= 1:
        return False
    avg = max(0.0, avg_task_s)
    overhead = max(0.0, overhead_per_task_s)
    sequential = avg * n_tasks
    parallel_compute = (avg * n_tasks) / workers
    parallel_total = parallel_compute + overhead * n_tasks
    return sequential > parallel_total
