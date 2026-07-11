"""Unit tests for built-in kernel registration across registries."""

from __future__ import annotations

import pytest

from diffract.containers import MainContainer, WiringConfiguration

pytestmark = pytest.mark.unit


def test_register_default_kernels_populates_fresh_registry(ram_container) -> None:
    """A registry created after the kernels module was already imported must
    still receive the built-in kernels (manifest replay, not import side
    effects). Regression: a bare wired container used to end up with an
    empty registry and poison every later container in the process."""
    from diffract.core.compute.decorator import _DEFAULT_KERNEL_SPECS

    assert "frob_norm" in ram_container.compute_singleton.kernel_registry().list_kernels()
    builtin_names = {spec["name"] for spec in _DEFAULT_KERNEL_SPECS}
    assert "frob_norm" in builtin_names

    bare = MainContainer()
    WiringConfiguration.wire(bare)
    bare.compute_singleton.register_default_kernels()

    fresh = bare.compute_singleton.kernel_registry()
    assert builtin_names <= set(fresh.list_kernels())
