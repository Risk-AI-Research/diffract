"""Unit tests for parallel runtime calibration cache."""

from __future__ import annotations

import pytest

from diffract.core.parallel import get_thread_pool_calibration

pytestmark = pytest.mark.unit


def test_get_thread_pool_calibration_is_cached() -> None:
    get_thread_pool_calibration.cache_clear()

    c1 = get_thread_pool_calibration(2)
    c2 = get_thread_pool_calibration(2)

    assert c1 == c2
    info = get_thread_pool_calibration.cache_info()
    assert info.hits >= 1
