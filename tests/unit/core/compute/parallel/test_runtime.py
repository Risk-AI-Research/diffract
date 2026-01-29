"""Unit tests for parallel runtime calibration cache."""

from __future__ import annotations

import pytest

from diffract.core.compute.parallel import runtime


pytestmark = pytest.mark.unit


def test_get_thread_pool_calibration_is_cached() -> None:
    runtime.get_thread_pool_calibration.cache_clear()

    c1 = runtime.get_thread_pool_calibration(2)
    c2 = runtime.get_thread_pool_calibration(2)

    assert c1 == c2
    info = runtime.get_thread_pool_calibration.cache_info()
    assert info.hits >= 1

