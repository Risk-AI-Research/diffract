"""Unit tests for session helper classes."""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import MagicMock

import pytest

pytest.importorskip("plotly")

from diffract.session.namespaces.models.parameters.meta_patcher import (
    MetadataPatcher,
    MetadataPatchError,
)
from diffract.session.namespaces.results.eraser import ResultsEraser, ResultsEraserError
from diffract.session.namespaces.results.injester import (
    FieldIngester,
    FieldIngestionError,
)

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

    def __enter__(self) -> Self:
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
        from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
        from diffract.core.data.nn.params.metadata import ParameterMetadata
        from diffract.core.data.nn.params.proxy import ParameterDataProxy
        from diffract.core.data.nn.params.repository import ParameterRepository
        from diffract.core.data.nn.params.schema import ParameterType
        from diffract.core.storage.ram_manager import RAMStorageManager

        index = SQLiteMetadataIndex(":memory:")
        try:
            storage = RAMStorageManager()
            repo = ParameterRepository.initialize(storage, index)
            proxy = ParameterDataProxy.create_and_store(
                ParameterMetadata(
                    name="weight",
                    ptype=ParameterType.DENSE,
                    model_id="model",
                    step=7,
                    other_meta={"epoch": 1, "split": "train"},
                ),
                repo,
            )
            original_instance = proxy.meta.instance_uid
            MetadataPatcher().patch(
                updates={proxy.meta.uid: {"epoch": 2}},
                parameters=repo.create_view(),
                force=True,
            )
            stored = ParameterRepository(storage, index).get_proxy(proxy.meta.uid).meta
            assert stored == proxy.meta
            assert (stored.step, stored.instance_uid) == (7, original_instance)
            assert stored.other_meta == {"epoch": 2, "split": "train"}
        finally:
            index.close()


class TestResultsEraser:
    """Tests for ResultsEraser."""

    def test_resolve_unknown_field_raises(self) -> None:
        """Unknown field should raise ResultsEraserError."""
        registry = MockKernelRegistry(producible_fields={"known_field"})
        eraser = ResultsEraser(kernel_registry=registry)

        with pytest.raises(ResultsEraserError, match="nonexistent_field"):
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

    def test_erase_treats_requested_fields_literally(self, ram_container) -> None:
        """Erase returns the affected owner count and preserves similar field names."""
        import numpy as np

        from diffract.session import Session

        with Session(container=ram_container) as session:
            session.models.add({"w": np.eye(2), "v": np.eye(2)}, model_id="m")
            repo = ram_container.nn.parameter_repository()
            proxy, other = (repo.get_proxy(uid) for uid in repo.list_uids())
            other.set_field("loss.total", 7)
            for field in ("metric[0]", "metric0", "loss.total", "lossXtotal"):
                proxy.set_field(field, 1)
            eraser = ResultsEraser(kernel_registry=MockKernelRegistry())
            fields = {"metric[0]", "loss.total"}
            assert eraser.erase(parameters=repo.create_view(), fields=fields) == 2
            assert eraser.erase(parameters=repo.create_view(), fields=fields) == 0
            assert proxy.has_field("metric0")
            assert proxy.has_field("lossXtotal")
            assert not proxy.has_field("metric[0]")
            assert not proxy.has_field("loss.total")
