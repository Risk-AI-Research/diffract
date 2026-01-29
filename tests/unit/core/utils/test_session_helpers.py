"""Unit tests for session helper classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from diffract.core.utils.session import (
    FieldIngester,
    FieldIngestionError,
    MetadataPatcher,
    MetadataPatchError,
    ResultsEraser,
    ResultsEraserError,
)

if TYPE_CHECKING:
    from diffract.core.data.nn.params.metadata import ParameterMetadata

pytestmark = pytest.mark.unit


class MockParameterProxy:
    """Mock parameter proxy for testing."""

    def __init__(
        self,
        uid: str,
        fields: dict[str, Any] | None = None,
        other_meta: dict[str, Any] | None = None,
    ) -> None:
        self._uid = uid
        self._fields = fields or {}
        self._other_meta = other_meta or {}

        # Create a mock meta object
        self.meta = MagicMock()
        self.meta.uid = uid
        self.meta.other_meta = self._other_meta
        self.meta.to_dict.return_value = {
            "uid": uid,
            "name": f"param_{uid}",
            "ptype": "DENSE",
            "model_id": "test_model",
            "other_meta": self._other_meta,
        }

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def get_field(self, name: str, default: Any = None) -> Any:
        return self._fields.get(name, default)

    def set_field(self, name: str, value: Any) -> None:
        self._fields[name] = value


class MockParameterView:
    """Mock parameter view for testing."""

    def __init__(self, proxies: dict[str, MockParameterProxy] | None = None) -> None:
        self._proxies = proxies or {}
        self._context_entered = False

    def __getitem__(self, uid: str) -> MockParameterProxy:
        if uid not in self._proxies:
            raise KeyError(uid)
        return self._proxies[uid]

    def __enter__(self) -> "MockParameterView":
        self._context_entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self._context_entered = False

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Mock implementation that tracks called patterns."""
        self._erased_patterns = list(patterns)


class MockKernelRegistry:
    """Mock kernel registry for testing."""

    def __init__(self, producible_fields: set[str] | None = None) -> None:
        self._producible_fields = producible_fields or set()

    def can_produce_field(self, name: str) -> bool:
        return name in self._producible_fields

    def list_kernels(self) -> list[str]:
        return []

    def get_kernel_producing_field(self, field_name: str) -> str:
        return f"kernel_for_{field_name}"

    def resolve_dependencies(self, kernel_name: str) -> list[str]:
        return []

    def get_fields_kernel_require(self, kernel_name: str) -> list[str]:
        return []

    def get_fields_kernel_produce(self, kernel_name: str) -> list[str]:
        return []


class TestFieldIngester:
    """Tests for FieldIngester."""

    def test_ingest_empty_input_returns_early(self) -> None:
        """Empty fields_by_uid should return without error."""
        ingester = FieldIngester()
        view = MockParameterView()
        ingester.ingest(fields_by_uid={}, parameters=view, force=False)

    def test_ingest_unknown_uids_raises(self) -> None:
        """Unknown UIDs should raise FieldIngestionError."""
        ingester = FieldIngester()
        view = MockParameterView()

        with pytest.raises(FieldIngestionError, match="Unknown parameter UIDs"):
            ingester.ingest(
                fields_by_uid={"nonexistent_uid": {"field": 1}},
                parameters=view,
                force=False,
            )

    def test_ingest_conflict_without_force_raises(self) -> None:
        """Existing field without force=True should raise."""
        ingester = FieldIngester()
        proxy = MockParameterProxy("uid1", fields={"existing_field": "old_value"})
        view = MockParameterView({"uid1": proxy})

        with pytest.raises(FieldIngestionError, match="Field conflicts"):
            ingester.ingest(
                fields_by_uid={"uid1": {"existing_field": "new_value"}},
                parameters=view,
                force=False,
            )

    def test_ingest_conflict_with_force_overwrites(self) -> None:
        """Existing field with force=True should overwrite."""
        ingester = FieldIngester()
        proxy = MockParameterProxy("uid1", fields={"existing_field": "old_value"})
        view = MockParameterView({"uid1": proxy})

        ingester.ingest(
            fields_by_uid={"uid1": {"existing_field": "new_value"}},
            parameters=view,
            force=True,
        )

        assert proxy._fields["existing_field"] == "new_value"

    def test_ingest_success_writes_all_fields(self) -> None:
        """Successful ingestion should write all fields."""
        ingester = FieldIngester()
        proxy = MockParameterProxy("uid1")
        view = MockParameterView({"uid1": proxy})

        ingester.ingest(
            fields_by_uid={"uid1": {"field1": 1, "field2": "two"}},
            parameters=view,
            force=False,
        )

        assert proxy._fields["field1"] == 1
        assert proxy._fields["field2"] == "two"


class TestMetadataPatcher:
    """Tests for MetadataPatcher."""

    def test_patch_empty_input_returns_early(self) -> None:
        """Empty updates should return without error."""
        patcher = MetadataPatcher()
        view = MockParameterView()
        patcher.patch(updates={}, parameters=view, force=False)

    def test_patch_unknown_uid_raises(self) -> None:
        """Unknown UID should raise MetadataPatchError."""
        patcher = MetadataPatcher()
        view = MockParameterView()

        with pytest.raises(MetadataPatchError, match="Unknown parameter UIDs"):
            patcher.patch(
                updates={"nonexistent_uid": {"key": "value"}},
                parameters=view,
                force=False,
            )

    def test_patch_conflict_without_force_raises(self) -> None:
        """Existing meta key without force=True should raise."""
        patcher = MetadataPatcher()
        proxy = MockParameterProxy("uid1", other_meta={"existing_key": "old_value"})
        view = MockParameterView({"uid1": proxy})

        with pytest.raises(MetadataPatchError, match="Metadata conflicts"):
            patcher.patch(
                updates={"uid1": {"existing_key": "new_value"}},
                parameters=view,
                force=False,
            )

    def test_patch_conflict_with_force_overwrites(self) -> None:
        """Existing meta key with force=True should overwrite."""
        from diffract.core.data.nn.params.metadata import ParameterMetadata
        from diffract.core.data.nn.params.schema import ParameterType

        patcher = MetadataPatcher()

        # Create a real metadata object since replace() requires dataclass
        real_meta = ParameterMetadata(
            uid="uid1",
            name="test_param",
            ptype=ParameterType.DENSE,
            model_id="test_model",
            other_meta={"existing_key": "old_value"},
        )

        # Create proxy with real metadata
        proxy = MagicMock()
        proxy.meta = real_meta
        type(proxy).meta = PropertyMock(return_value=real_meta)

        # Create a mock repository with metadata_index
        mock_metadata_index = MagicMock()
        mock_repository = MagicMock()
        mock_repository.metadata_index = mock_metadata_index
        mock_repository.__enter__ = MagicMock(return_value=mock_repository)
        mock_repository.__exit__ = MagicMock(return_value=None)

        view = MockParameterView({"uid1": proxy})
        view._repository = mock_repository

        patcher.patch(
            updates={"uid1": {"existing_key": "new_value"}},
            parameters=view,
            force=True,
        )

        # Verify metadata_index.update was called
        mock_metadata_index.update.assert_called_once()


class TestResultsEraser:
    """Tests for ResultsEraser."""

    def test_resolve_unknown_field_raises(self) -> None:
        """Unknown field should raise ResultsEraserError."""
        registry = MockKernelRegistry(producible_fields={"known_field"})
        eraser = ResultsEraser(kernel_registry=registry)

        with pytest.raises(ResultsEraserError, match="cannot produce"):
            eraser.resolve_fields_to_erase(
                ["nonexistent_field"], erase_dependent_also=False
            )

    def test_resolve_known_field_succeeds(self) -> None:
        """Known field should be resolved successfully."""
        registry = MockKernelRegistry(producible_fields={"known_field"})
        eraser = ResultsEraser(kernel_registry=registry)

        result = eraser.resolve_fields_to_erase(
            ["known_field"], erase_dependent_also=False
        )

        assert "known_field" in result

    def test_erase_escapes_regex_chars(self) -> None:
        """Field names with regex chars should be escaped."""
        registry = MockKernelRegistry()
        eraser = ResultsEraser(kernel_registry=registry)
        view = MockParameterView()

        # Fields with regex special characters
        eraser.erase(view=view, fields={"metric[0]", "loss.total"})

        # Verify patterns are escaped (should contain backslash escapes)
        assert hasattr(view, "_erased_patterns")
        patterns = view._erased_patterns

        # Check that special chars are escaped
        assert any(r"\[0\]" in p for p in patterns)
        assert any(r"\." in p for p in patterns)

    def test_erase_includes_contextual_pattern(self) -> None:
        """Erase patterns should match contextual field suffixes."""
        registry = MockKernelRegistry()
        eraser = ResultsEraser(kernel_registry=registry)
        view = MockParameterView()

        eraser.erase(view=view, fields={"metric"})

        # Pattern should match "metric" and "metric@anything"
        assert hasattr(view, "_erased_patterns")
        patterns = view._erased_patterns
        assert len(patterns) == 1
        assert "(@.+)?" in patterns[0]
