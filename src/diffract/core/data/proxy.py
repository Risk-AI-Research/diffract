"""Generic data proxy for lazy-loading storage-backed entities.

This module provides the base DataProxy class that can be specialized
for different data types (parameters, relations, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from diffract.core.storage.interface import DEFAULT_TABLE

if TYPE_CHECKING:
    from .interface import IDataRepository

TMetadata = TypeVar("TMetadata")
T = TypeVar("T")


@dataclass(kw_only=True)
class DataProxy[TMetadata]:
    """Generic lazy-loading proxy for storage-backed data entities.

    Provides efficient access to data through intelligent caching and
    storage management. Entities are loaded on-demand and can be
    prefetched for batch operations. The proxy pattern allows working
    with large datasets without memory constraints.

    Type Parameters:
        TMetadata: The metadata type for this proxy (e.g., ParameterMetadata).

    Example:
        >>> proxy = DataProxy.create_and_store(
        ...     meta=metadata,
        ...     repository=repository,
        ... )
        >>> value = proxy.get_field("weights")
        >>> proxy.set_field("computed_metric", computed_value)

    Attributes:
        meta: Immutable entity metadata.
        _repository: Repository that owns storage/cache/metadata managers.
    """

    meta: TMetadata
    _repository: IDataRepository[TMetadata, DataProxy[TMetadata]] = field(repr=False)

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
        repository: IDataRepository[TMetadata, DataProxy[TMetadata]],
    ) -> DataProxy[TMetadata]:
        """Create proxy and store metadata in index.

        Factory method that creates a new proxy and immediately
        stores the metadata to the metadata index.

        Args:
            meta: Entity metadata.
            repository: Repository owning storage, cache, and metadata managers.

        Returns:
            New proxy with metadata stored.
        """
        # Store full metadata in the index
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

    def set_field(self, name: str, value: Any) -> None:
        """Store a named field for this entity.

        Args:
            name: Field name to store.
            value: Value to store (must be serializable).
        """
        self._repository.storage_manager.set_field(
            obj_uid=self.meta.uid,
            field_name=name,
            value=value,
            table=self.get_table(),
        )
        self.prefetch_field(name, value)

    def get_field(
        self, name: str, default: T | None = None, auto_prefetch: bool = True
    ) -> Any | T:
        """Retrieve a named field for this entity.

        Checks cache first for performance, falls back to storage if needed.
        Optionally prefetches the field to cache for future access.

        Args:
            name: Field name to retrieve.
            default: Default value if field doesn't exist.
            auto_prefetch: Whether to cache the value after loading.

        Returns:
            Field value or default if not found.

        Raises:
            KeyError: If field doesn't exist and no default provided.
        """
        cache = self._repository.cache_manager
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

    def has_field(self, name: str) -> bool:
        """Check if entity has a named field.

        Checks both cache and storage for field existence.

        Args:
            name: Field name to check.

        Returns:
            True if field exists, False otherwise.
        """
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
        return cache.has_field(obj_uid=self.meta.uid, field_name=name)

    def prefetch_field(self, name: str, value: Any) -> None:
        """Cache a field value for fast access.

        Args:
            name: Field name to cache.
            value: Value to cache.
        """
        cache = self._repository.cache_manager
        if cache is not None:
            cache.set_field(obj_uid=self.meta.uid, field_name=name, value=value)

    def try_prefetch_field(self, name: str) -> bool:
        """Attempt to prefetch a field into cache.

        Args:
            name: Field name to prefetch.

        Returns:
            True if field was prefetched or already cached, False if not found.
        """
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
        if (cache is not None) and self.is_field_prefetched(name):
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
