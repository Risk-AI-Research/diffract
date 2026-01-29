"""Unit tests for optional import helpers."""

from __future__ import annotations

import pytest

from diffract.core.utils import imports as import_utils

pytestmark = pytest.mark.unit


def test_is_available_and_get_module() -> None:
    assert import_utils.is_available("json") is True
    assert import_utils.get_module("json") is not None

    missing = "definitely_not_a_package_12345"
    assert import_utils.is_available(missing) is False
    assert import_utils.get_module(missing) is None


def test_require_missing_raises_helpful_error() -> None:
    with pytest.raises(import_utils.OptionalDependencyError, match="Install it with"):
        import_utils.require("definitely_not_a_package_12345")


def test_lazy_import_success_and_failure() -> None:
    lazy_math = import_utils.LazyImport("math")
    assert lazy_math.sqrt(9) == 3

    lazy_missing = import_utils.LazyImport("definitely_not_a_package_12345")
    with pytest.raises(import_utils.OptionalDependencyError):
        _ = lazy_missing.anything


def test_module_level_availability_flags() -> None:
    # Triggers module __getattr__().
    assert isinstance(import_utils._IS_TORCH_AVAILABLE, bool)


def test_requires_package_decorator() -> None:
    @import_utils.requires_package("json")
    def ok() -> int:
        return 1

    assert ok() == 1

    @import_utils.requires_package("definitely_not_a_package_12345", fallback_return="fallback")
    def returns_fallback() -> str:
        return "real"

    assert returns_fallback() == "fallback"

    @import_utils.requires_package("definitely_not_a_package_12345")
    def raises() -> int:
        return 123

    with pytest.raises(import_utils.OptionalDependencyError, match="requires package"):
        raises()

