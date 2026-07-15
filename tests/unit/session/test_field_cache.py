"""Unit tests for SessionFieldCache."""

from __future__ import annotations

import pytest

from diffract.session.field_cache import SessionFieldCache

pytestmark = pytest.mark.unit


class TestSessionFieldCacheBasic:
    """Basic functionality tests for SessionFieldCache."""

    def test_initial_state_is_invalid(self) -> None:
        """Newly created cache should be invalid."""
        cache = SessionFieldCache()
        assert not cache.is_valid
        assert cache.get() is None
        assert cache.generation == 0

    def test_set_makes_cache_valid(self) -> None:
        """Setting data should make cache valid."""
        cache = SessionFieldCache()
        fields_by_uid = {"uid1": ["field1", "field2"], "uid2": ["field3"]}

        cache.set(fields_by_uid)

        assert cache.is_valid
        assert cache.get() == fields_by_uid

    def test_get_returns_stored_data(self) -> None:
        """Get should return exactly what was set."""
        cache = SessionFieldCache()
        fields_by_uid = {"uid1": ["a", "b"], "uid2": ["c"]}

        cache.set(fields_by_uid)
        result = cache.get()

        assert result == fields_by_uid
        # Should be the same object (not a copy)
        assert result is fields_by_uid


class TestSessionFieldCacheInvalidation:
    """Tests for cache invalidation behavior."""

    def test_invalidate_clears_data(self) -> None:
        """Invalidate should clear cached data."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["field1"]})

        cache.invalidate()

        assert not cache.is_valid
        assert cache.get() is None

    def test_invalidate_increments_generation(self) -> None:
        """Each invalidation should increment generation counter."""
        cache = SessionFieldCache()
        assert cache.generation == 0

        cache.invalidate()
        assert cache.generation == 1

        cache.invalidate()
        assert cache.generation == 2

    def test_multiple_set_invalidate_cycles(self) -> None:
        """Cache should work correctly through multiple cycles."""
        cache = SessionFieldCache()

        # First cycle
        cache.set({"uid1": ["a"]})
        assert cache.is_valid
        cache.invalidate()
        assert not cache.is_valid

        # Second cycle
        cache.set({"uid2": ["b"]})
        assert cache.is_valid
        assert cache.get() == {"uid2": ["b"]}

    def test_clear_resets_generation(self) -> None:
        """Clear should reset generation to zero."""
        cache = SessionFieldCache()
        cache.invalidate()
        cache.invalidate()
        assert cache.generation == 2

        cache.clear()

        assert cache.generation == 0
        assert not cache.is_valid


class TestSessionFieldCacheIncrementalUpdate:
    """Tests for incremental cache updates."""

    def test_add_computed_fields_updates_existing_entries(self) -> None:
        """Adding fields should update existing cache entries."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["field1"], "uid2": ["field2"]})

        cache.add_computed_fields(["uid1", "uid2"], ["new_field"])

        result = cache.get()
        assert "new_field" in result["uid1"]
        assert "new_field" in result["uid2"]
        # Original fields should be preserved
        assert "field1" in result["uid1"]
        assert "field2" in result["uid2"]

    def test_add_computed_fields_skips_unknown_uids(self) -> None:
        """Adding fields for unknown UIDs should not create new entries."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["field1"]})

        cache.add_computed_fields(["uid1", "unknown_uid"], ["new_field"])

        result = cache.get()
        assert "new_field" in result["uid1"]
        assert "unknown_uid" not in result

    def test_add_computed_fields_noop_when_invalid(self) -> None:
        """Adding fields to invalid cache should be a no-op."""
        cache = SessionFieldCache()
        # Cache is not populated

        cache.add_computed_fields(["uid1"], ["new_field"])

        assert not cache.is_valid

    def test_add_computed_fields_multiple_fields(self) -> None:
        """Should handle adding multiple fields at once."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["a"]})

        cache.add_computed_fields(["uid1"], ["b", "c", "d"])

        result = cache.get()
        assert set(result["uid1"]) == {"a", "b", "c", "d"}

    def test_remove_fields_by_uids_removes_from_entries(self) -> None:
        """Removing fields should update cache entries."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["field1", "field2"], "uid2": ["field1", "field3"]})

        cache.remove_fields_by_uids(["uid1", "uid2"], ["field1"])

        result = cache.get()
        assert result["uid1"] == ["field2"]
        assert result["uid2"] == ["field3"]

    def test_remove_fields_by_uids_noop_when_invalid(self) -> None:
        """Removing fields from invalid cache should be a no-op."""
        cache = SessionFieldCache()

        cache.remove_fields_by_uids(["uid1"], ["field1"])

        assert not cache.is_valid


class TestSessionFieldCacheEdgeCases:
    """Edge case tests for SessionFieldCache."""

    def test_empty_fields_by_uid(self) -> None:
        """Setting empty dict should still make cache valid."""
        cache = SessionFieldCache()

        cache.set({})

        assert cache.is_valid
        assert cache.get() == {}

    def test_empty_field_list_for_uid(self) -> None:
        """UIDs with empty field lists should be handled."""
        cache = SessionFieldCache()

        cache.set({"uid1": [], "uid2": ["field1"]})

        result = cache.get()
        assert result["uid1"] == []
        assert result["uid2"] == ["field1"]

    def test_add_fields_to_empty_field_list(self) -> None:
        """Adding fields to UID with empty list should work."""
        cache = SessionFieldCache()
        cache.set({"uid1": []})

        cache.add_computed_fields(["uid1"], ["new_field"])

        result = cache.get()
        assert "new_field" in result["uid1"]

    def test_add_duplicate_fields(self) -> None:
        """Adding fields that already exist should not duplicate."""
        cache = SessionFieldCache()
        cache.set({"uid1": ["field1"]})

        cache.add_computed_fields(["uid1"], ["field1"])

        result = cache.get()
        assert result["uid1"].count("field1") == 1
