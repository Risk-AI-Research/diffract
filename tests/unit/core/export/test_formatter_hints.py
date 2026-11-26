"""Unit tests for formatter lookup error messages."""

from __future__ import annotations

import pytest

from diffract.core.export.formatters import registry

pytestmark = pytest.mark.unit


def test_missing_optional_format_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(registry.FORMATTERS, "pandas", raising=False)

    with pytest.raises(ValueError, match=r"diffract-core\[pandas\]"):
        registry.get_formatter("pandas")


def test_unknown_format_lists_known_and_suggests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Did you mean: dict"):
        registry.get_formatter("dcit")


def test_unknown_format_without_close_match_lists_known() -> None:
    with pytest.raises(ValueError, match="Known: "):
        registry.get_formatter("parquet")


def test_custom_formatter_wrong_result_is_rejected_at_registration() -> None:
    class WrongFormatter:
        def format_results(self, *_args):
            return "results"

    with pytest.raises(TypeError, match="must return StructuredExportResult"):
        registry.register_formatter("wrong", WrongFormatter())
    assert "wrong" not in registry.FORMATTERS
