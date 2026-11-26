"""Generic data proxy for lazy-loading storage-backed entities.

This module provides the base DataProxy class that can be specialized
for different data types (parameters, relations, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Self

from typing_extensions import TypeVar

from diffract.core.storage.interface import DEFAULT_TABLE

from .interface import IDataProxy, IDataRepository, IMetadata

if TYPE_CHECKING:
    from diffract.core.storage.catalog import ValuePointer
    from diffract.core.storage.catalog_manager import CatalogStorageManager

TMetadata = TypeVar("TMetadata", bound=IMetadata)
TRepository = TypeVar(
    "TRepository",
    bound=IDataRepository[Any, Any],
    default=IDataRepository[TMetadata, Any],
)
T = TypeVar("T")
_CACHE_MISS = object()
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DataProxy(IDataProxy[TMetadata], Generic[TMetadata, TRepository]):
    """Access the stored fields of a metadata record through its owning repository."""

    meta: TMetadata
    _repository: TRepository = field(repr=False)

    def __class_getitem__(cls, parameters: Any) -> Any:
        """Allow DataProxy[Metadata] on Python 3.12 with older typing-extensions."""
        if not isinstance(parameters, tuple):
            parameters = (parameters, IDataRepository[parameters, Any])
        return super().__class_getitem__(parameters)  # type: ignore[misc]  # Generic hook on 3.12

    @classmethod
    def get_table(cls) -> str:
        """Return the storage table name for this proxy type.

        Override in subclasses to specify a different table.
        """
        return DEFAULT_TABLE

    @classmethod
    def create_and_store(
        cls,
        meta: TMetadata,
        repository: TRepository,
    ) -> Self:
        """Persist the metadata and add its proxy to the repository."""
        repository.metadata_index.upsert(
            cls.get_table(),
            **meta.to_dict(),
        )

        proxy = cls(
            meta=meta,
            _repository=repository,
        )
        repository.append(proxy)
        return proxy

    def _catalog_storage(self) -> CatalogStorageManager | None:
        storage = self._repository.storage_manager
        if getattr(
            storage, "uses_value_revisions", False
        ) is True and self.get_table() in {
            "parameters",
            "aggregates",
        }:
            return storage
        return None

    def set_field(self, name: str, value: Any, *, initial: bool = False) -> None:
        """Store a named field for this entity.

        Args:
            name: Field name to store.
            value: Value to store (must be serializable).
            initial: Prefer inserting a new field pointer.
        """
        if (storage := self._catalog_storage()) is not None:
            publish = getattr(type(storage), "publish_field", None)
            if initial:
                publish = getattr(type(storage), "publish_initial_field", publish)
            if publish is None:
                pointer = storage.write_field(
                    self.meta.uid, name, value, table=self.get_table()
                )
                payload_written = True
            else:
                pointer, payload_written = publish(
                    storage, self.meta.uid, name, value, table=self.get_table()
                )
            cache = self._repository.cache_manager
            if payload_written and getattr(cache, "supports_revisions", False) is True:
                cache.set_field(self.meta.uid, name, value, revision=pointer.revision)
            return
        self._repository.storage_manager.set_field(
            obj_uid=self.meta.uid,
            field_name=name,
            value=value,
            table=self.get_table(),
        )
        self.prefetch_field(name, value)

    def get_field(
        self,
        name: str,
        default: T | None = None,
        auto_prefetch: bool = True,
        *,
        kernel_input: bool = False,
    ) -> Any | T:
        """Retrieve a named field for this entity.

        Checks cache first for performance, falls back to storage if needed.
        Optionally prefetches the field to cache for future access.

        Args:
            name: Field name to retrieve.
            default: Default value if field doesn't exist.
            auto_prefetch: Whether to cache the value after loading.
            kernel_input: Allow borrowing a resident input for kernel execution.

        Returns:
            Field value or default if not found.

        Raises:
            KeyError: If field doesn't exist and no default provided.
        """
        cache = self._repository.cache_manager
        if (storage := self._catalog_storage()) is not None:
            try:
                pointer = storage.live_pointer(self.meta.uid, name, self.get_table())
            except KeyError:
                if default is not None:
                    return default
                raise
            return self.read_resolved_field(
                name, pointer, auto_prefetch=auto_prefetch, kernel_input=kernel_input
            )
        if (cache is not None) and self.is_field_prefetched(name):
            value = cache.get_field(obj_uid=self.meta.uid, field_name=name)
        else:
            try:
                value = self._repository.storage_manager.get_field(
                    obj_uid=self.meta.uid, field_name=name, table=self.get_table()
                )

            except KeyError:
                if default is not None:
                    value = default
                else:
                    raise

            if auto_prefetch:
                self.prefetch_field(name, value)

        return value

    def read_resolved_field(
        self,
        name: str,
        pointer: ValuePointer,
        *,
        auto_prefetch: bool,
        best_effort: bool = False,
        kernel_input: bool = False,
    ) -> Any:
        """Read a catalog selection using the cache entry for its precise revision."""
        cache = self._repository.cache_manager
        storage = self._repository.storage_manager
        versioned = getattr(cache, "supports_revisions", False) is True
        if versioned:
            value = cache.get_field(
                self.meta.uid, name, revision=pointer.revision, default=_CACHE_MISS
            )
            if value is not _CACHE_MISS:
                return value
        value = (
            storage.read_kernel_input(pointer)
            if kernel_input
            else storage.read_pointer(pointer)
        )
        if auto_prefetch and versioned:
            try:
                cache.set_field(self.meta.uid, name, value, revision=pointer.revision)
            except Exception:
                if not best_effort:
                    raise
                logger.debug("Could not cache exported field %s", name, exc_info=True)
        return value

    def has_field(self, name: str) -> bool:
        """Check if entity has a named field.

        Checks both cache and storage for field existence.

        Args:
            name: Field name to check.

        Returns:
            True if field exists, False otherwise.
        """
        if (storage := self._catalog_storage()) is not None:
            return storage.has_field(self.meta.uid, name, table=self.get_table())
        cache = self._repository.cache_manager
        if (cache is not None) and (self.is_field_prefetched(name)):
            return True
        return self._repository.storage_manager.has_field(
            obj_uid=self.meta.uid,
            field_name=name,
            table=self.get_table(),
        )

    def get_field_metadata(self, name: str) -> dict[str, Any] | None:
        """Return storage-level metadata for the given field if available."""
        return self._repository.storage_manager.get_field_metadata(
            obj_uid=self.meta.uid, field_name=name, table=self.get_table()
        )

    def is_field_prefetched(self, name: str) -> bool:
        """Check if field is cached for fast access.

        Args:
            name: Field name to check.

        Returns:
            True if field is in cache, False otherwise.
        """
        cache = self._repository.cache_manager
        if cache is None:
            return True
        if (storage := self._catalog_storage()) is not None:
            if getattr(cache, "supports_revisions", False) is not True:
                return False
            try:
                pointer = storage.live_pointer(self.meta.uid, name, self.get_table())
            except KeyError:
                return False
            return cache.has_field(self.meta.uid, name, revision=pointer.revision)
        return cache.has_field(obj_uid=self.meta.uid, field_name=name)

    def prefetch_field(self, name: str, value: Any) -> None:
        """Cache a field value for fast access.

        Args:
            name: Field name to cache.
            value: Value to cache.
        """
        if self._catalog_storage() is not None:
            # A caller-supplied value has no associated catalog revision.
            self.try_prefetch_field(name)
            return
        cache = self._repository.cache_manager
        if cache is not None:
            cache.set_field(obj_uid=self.meta.uid, field_name=name, value=value)

    def try_prefetch_field(self, name: str) -> bool:
        """Attempt to prefetch a field into cache.

        Args:
            name: Field name to prefetch.

        Returns:
            True if prefetching succeeds or caching is disabled. With a cache,
            False if the field is missing from storage.
        """
        if (storage := self._catalog_storage()) is not None:
            cache = self._repository.cache_manager
            if cache is None:
                return True
            try:
                pointer = storage.live_pointer(self.meta.uid, name, self.get_table())
            except KeyError:
                return False
            if getattr(
                cache, "supports_revisions", False
            ) is True and not cache.has_field(
                self.meta.uid, name, revision=pointer.revision
            ):
                value = storage.read_pointer(pointer)
                cache.set_field(self.meta.uid, name, value, revision=pointer.revision)
            return True
        if not self.is_field_prefetched(name):
            try:
                value = self._repository.storage_manager.get_field(
                    obj_uid=self.meta.uid, field_name=name, table=self.get_table()
                )

            except KeyError:
                return False

            self.prefetch_field(name, value)

        return True

    def erase_field(self, name: str) -> None:
        """Remove a field from storage and cache.

        Args:
            name: Field name to remove.
        """
        self._repository.storage_manager.erase_field(
            obj_uid=self.meta.uid,
            field_name=name,
            table=self.get_table(),
        )
        cache = self._repository.cache_manager
        if cache is not None:
            cache.erase_field(obj_uid=self.meta.uid, field_name=name)

    def list_fields(self) -> list[str]:
        """List all field names for this entity.

        Returns:
            List of field names stored for this entity.
        """
        return self._repository.storage_manager.list_fields(
            obj_uid=self.meta.uid, table=self.get_table()
        )

    def __hash__(self) -> int:
        """Return hash based on unique identifier.

        Returns:
            Hash of the entity UID.
        """
        return hash(self.meta.uid)
