"""Fixtures and hypothesis configuration for compute-kernel tests.

Kernel functions are only importable once a container has wired the
`@kernel` decorator, and hypothesis re-runs each test body many times, so
wiring happens once per session (not per function, which would trip
hypothesis' function-scoped-fixture health check).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

settings.register_profile("diffract", max_examples=25, deadline=None, derandomize=True)
settings.load_profile("diffract")


@pytest.fixture(scope="session", autouse=True)
def _wire_kernels(ram_config_factory: Callable[[str], Path]) -> object:
    """Wire a container once so kernel functions import session-wide."""
    from diffract.containers import create_main_container

    return create_main_container(ram_config_factory("kernels_wiring"))
