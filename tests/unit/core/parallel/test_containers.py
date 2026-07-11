"""Tests for shared parallel resources."""

from __future__ import annotations

import pytest

from diffract.core.parallel import ParallelSingletonContainer


pytestmark = pytest.mark.unit


def test_thread_pool_context_is_closed_on_shutdown() -> None:
    container = ParallelSingletonContainer()
    container.config.from_dict({"thread_pool": {"max_workers": 2}})

    container.init_resources()
    ctx = container.thread_pool_context()
    container.shutdown_resources()

    with pytest.raises(RuntimeError):
        _ = ctx.executor.submit(lambda: 1)

