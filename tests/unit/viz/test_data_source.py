"""Tests for DataProvider and related data utilities (migrated to v0.2.0 viz API)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_session_with_export():
    """Create a mock session whose results.export() returns the unified shape.

    The v0.2.0 DataProvider always fetches via the unified
    ``session.results.export(...)`` API, which returns entries shaped as
    ``{uid: {"metadata": {...}, "fields": {...}}}``.
    """
    session = MagicMock()

    sample_data = {
        "p1": {
            "metadata": {"name": "layer.0.weight", "model_id": "m1"},
            "fields": {"stable_rank": 5.0, "frob_norm": 10.0},
        },
        "p2": {
            "metadata": {"name": "layer.1.weight", "model_id": "m1"},
            "fields": {"stable_rank": 3.0, "frob_norm": 8.0},
        },
    }

    session.results = MagicMock()
    session.results.export = MagicMock(return_value=sample_data)

    return session


class TestDataProvider:
    def test_fetch_basic(self, mock_session_with_export):
        from diffract.viz.data import DataProvider

        provider = DataProvider(mock_session_with_export)
        entries = provider.fetch(["stable_rank", "frob_norm"])

        assert len(entries) == 2
        assert "p1" in entries
        assert "p2" in entries

        # In v0.2.0 the field shape is reported as a DataShape enum via
        # get_field_data rather than a "field_shapes" string map.
        from diffract.viz.data import DataShape

        _, _, shape = provider.get_field_data("stable_rank")
        assert shape == DataShape.SCALAR

    def test_fetch_uses_unified_export(self, mock_session_with_export):
        from diffract.viz.data import DataProvider

        provider = DataProvider(mock_session_with_export)
        provider.fetch(["stable_rank"])

        # The unified export() path is always used (no include_aggregates toggle).
        mock_session_with_export.results.export.assert_called_once()

    def test_fetch_with_value_filter(self, mock_session_with_export):
        from diffract.viz.data import DataProvider

        provider = DataProvider(mock_session_with_export)
        entries = provider.fetch(
            ["stable_rank"],
            value_filter={"stable_rank": (">", 4.0)},
        )

        # Only p1 has stable_rank > 4.0
        assert len(entries) == 1
        assert "p1" in entries


class TestValueFilter:
    def test_filter_operators(self):
        from diffract.viz.data.filtering import _check_condition

        assert _check_condition(5, ">", 3) is True
        assert _check_condition(5, "<", 3) is False
        assert _check_condition(5, ">=", 5) is True
        assert _check_condition(5, "<=", 5) is True
        assert _check_condition(5, "==", 5) is True
        assert _check_condition(5, "!=", 5) is False

    def test_filter_with_array_uses_mean(self):
        from diffract.viz.data.filtering import _check_condition

        arr = np.array([1.0, 2.0, 3.0])
        assert _check_condition(arr, ">", 1.5) is True
        assert _check_condition(arr, "<", 1.5) is False

    def test_filter_with_none_returns_false(self):
        from diffract.viz.data.filtering import _check_condition

        assert _check_condition(None, ">", 0) is False

    def test_entry_passes_filter(self):
        from diffract.viz.data.filtering import _entry_passes_filter

        # In v0.2.0 metadata and fields are merged into a single "fields" map on
        # the Entry, so filterable metadata (e.g. layer_id) lives in "fields".
        entry = {
            "fields": {"metric": 10.0, "layer_id": 5},
        }

        assert _entry_passes_filter(entry, {"metric": (">", 5.0)}) is True
        assert _entry_passes_filter(entry, {"metric": ("<", 5.0)}) is False
        # Can also filter on former-metadata fields
        assert _entry_passes_filter(entry, {"layer_id": ("==", 5)}) is True
