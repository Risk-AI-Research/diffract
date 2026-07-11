"""Parallel execution utilities for compute subsystem."""

from .containers import ParallelSingletonContainer
from .execution import map_maybe_parallel
from .runtime import (
    ParallelCalibration,
    ParallelContext,
    calibrate_thread_pool_overhead,
    get_thread_pool_calibration,
    should_parallelize,
)

__all__ = [
    "ParallelCalibration",
    "ParallelContext",
    "ParallelSingletonContainer",
    "calibrate_thread_pool_overhead",
    "get_thread_pool_calibration",
    "map_maybe_parallel",
    "should_parallelize",
]
