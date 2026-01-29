"""Parallel execution utilities."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from typing import TypeVar

from diffract.core.utils.math import mean

from .runtime import ParallelContext, should_parallelize

T = TypeVar("T")
R = TypeVar("R")


def map_maybe_parallel(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    parallel: ParallelContext | None,
) -> Iterator[R]:
    """Map fn over items with automatic parallelization decision.

    Uses a small parallel pilot batch to estimate average task cost and
    decide whether remaining work should be parallelized.

    Args:
        items: Sequence of items to process.
        fn: Function to apply to each item.
        parallel: Optional parallel context for execution.

    Returns:
        List of results from applying fn to each item.
    """
    if not items:
        return iter([])

    if parallel is None:
        return map(fn, items)

    executor = parallel.executor
    workers = parallel.workers
    overhead = parallel.calibration.submit_overhead_per_task_s

    n = len(items)
    if n <= 1 or workers <= 1:
        return map(fn, items)

    # Fast path: for small batches, avoid pilot overhead and preserve order.
    if n <= workers:
        return executor.map(fn, items)

    # If overhead is already comparable to expected work, skip parallelism.
    # This is a cheap guard for very small/fast tasks.
    overhead = max(0.0, overhead)
    pilot_n = min(n, workers)

    def _timed(idx: int, item: T) -> tuple[int, float, R]:
        t0 = time.perf_counter()
        value = fn(item)
        dt = max(0.0, time.perf_counter() - t0)
        return idx, dt, value

    futures = [executor.submit(_timed, i, items[i]) for i in range(pilot_n)]

    pilot_results: list[R | None] = [None] * pilot_n
    dts: list[float] = []
    for fut in concurrent.futures.as_completed(futures):
        idx, dt, value = fut.result()
        pilot_results[idx] = value
        dts.append(dt)

    rest = items[pilot_n:]
    if not rest:
        return iter(pilot_results)

    avg = mean(dts)
    if should_parallelize(
        avg_task_s=avg,
        n_tasks=len(rest),
        workers=workers,
        overhead_per_task_s=overhead,
    ):
        rest_results = executor.map(fn, rest)
    else:
        rest_results = map(fn, rest)

    return chain(pilot_results, rest_results)
