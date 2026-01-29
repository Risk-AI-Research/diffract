"""Helper utilities for integration and stress tests."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def create_test_model(framework: str = "torch") -> Any:
    """Create a test neural network model.

    Args:
        framework: Framework to use ("torch" or "tensorflow").

    Returns:
        Model instance.

    Raises:
        ImportError: If required framework is not available.
    """
    if framework == "torch":
        if not TORCH_AVAILABLE:
            msg = "PyTorch not available"
            raise ImportError(msg)
        return nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.ReLU(),
            nn.Linear(20, 5, bias=True),
        )

    if framework == "tensorflow":
        if not TF_AVAILABLE:
            msg = "TensorFlow not available"
            raise ImportError(msg)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(20, input_shape=(10,), use_bias=False),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(5, use_bias=True),
            ]
        )
        return model

    msg = f"Unknown framework: {framework}"
    raise ValueError(msg)


def create_large_array(size_mb: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create a large numpy array for testing.

    Args:
        size_mb: Size in megabytes.
        dtype: NumPy dtype.

    Returns:
        Large numpy array.
    """
    size_bytes = size_mb * 1024 * 1024
    size_elements = size_bytes // np.dtype(dtype).itemsize
    return np.random.randn(size_elements).astype(dtype)


def measure_memory_usage() -> dict[str, float]:
    """Measure current memory usage.

    Returns:
        Dictionary with memory metrics in MB.
    """
    if not PSUTIL_AVAILABLE:
        return {"rss_mb": 0.0, "vms_mb": 0.0}

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


def run_concurrent_operations(
    operation: Callable[[int], Any],
    num_threads: int,
    operations_per_thread: int,
    timeout: float = 300.0,
) -> tuple[list[Any], list[Exception]]:
    """Run concurrent operations and collect results.

    Args:
        operation: Function to run, takes thread index as argument.
        num_threads: Number of concurrent threads.
        operations_per_thread: Number of operations per thread.
        timeout: Maximum time to wait in seconds.

    Returns:
        Tuple of (results list, exceptions list).
    """
    results: list[Any] = []
    exceptions: list[Exception] = []
    results_lock = threading.Lock()
    exceptions_lock = threading.Lock()

    def worker(thread_idx: int) -> None:
        for _ in range(operations_per_thread):
            try:
                result = operation(thread_idx)
                with results_lock:
                    results.append(result)
            except Exception as e:  # noqa: BLE001
                with exceptions_lock:
                    exceptions.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

    start_time = time.time()
    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=max(1.0, timeout - (time.time() - start_time)))
        if t.is_alive():
            msg = f"Thread {t.name} did not complete within timeout"
            raise TimeoutError(msg)

    return results, exceptions

