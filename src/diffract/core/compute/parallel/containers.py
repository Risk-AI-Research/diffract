"""Dependency injection container for pool-per-method executors."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any

from dependency_injector import containers, providers

from .runtime import ParallelContext, get_thread_pool_calibration


def _normalize_workers(max_workers: int | None) -> int:
    if max_workers is None:
        return max(1, os.cpu_count() or 1)
    return max(1, int(max_workers))


def _thread_pool_context_resource(*, workers: int) -> Any:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        calibration = get_thread_pool_calibration(workers)
        yield ParallelContext(
            executor=executor, calibration=calibration, workers=workers
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _process_pool_context_resource(*, workers: int) -> Any:
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class ParallelSingletonContainer(containers.DeclarativeContainer):
    """Singleton container for project-wide parallelism resources."""

    config = providers.Configuration()

    max_workers = providers.Callable(_normalize_workers, config.thread_pool.max_workers)
    process_max_workers = providers.Callable(
        _normalize_workers, config.process_pool.max_workers
    )

    thread_pool_context = providers.Resource(
        _thread_pool_context_resource,
        workers=max_workers,
    )

    process_pool_context = providers.Resource(
        _process_pool_context_resource,
        workers=process_max_workers,
    )
