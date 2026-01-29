"""Generic protocols for data abstractions in diffract.

This module defines the core protocols for domain-agnostic data management.
It provides framework-agnostic contracts for:

1. IMetadata: Serializable metadata with unique identifier
2. IDataProxy: Lazy-loading proxy for storage-backed data
3. IDataView: Batch operations on filtered collections
4. IDataRepository: Ownership of storage/cache and membership management

These protocols are implemented by generic classes that can be specialized
for different data types (parameters, relations, etc.).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    overload,
    runtime_checkable,
)

if TYPE_CHECKING:
    import types
    from collections.abc import Iterable, Iterator, Sequence

    from diffract.core.cache.interface import ICacheManager
    from diffract.core.compute.parallel import ParallelContext
    from diffract.core.metadata.interface import IMetadataIndex
    from diffract.core.storage.interface import IStorageManager


@runtime_checkable
class IMetadata(Protocol):
    """Protocol for metadata objects with serialization support.

    Metadata objects must have a unique identifier and support
    dictionary serialization for storage persistence.
    """

    @property
    def uid(self) -> str:
        """Unique identifier for this metadata object."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary for storage.

        Returns:
            Dictionary representation suitable for storage.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize metadata from dictionary.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            Metadata instance.
        """
        ...


TMetadata = TypeVar("TMetadata", bound=IMetadata)
TProxy = TypeVar("TProxy", bound="IDataProxy")


@runtime_checkable
class IDataProxy(Protocol[TMetadata]):
    """Protocol for a proxy object representing a single data entity.

    This protocol describes how the system interacts with a single
    data object (conceptually: a single "row" in storage). The proxy
    is a *handle* for interacting with storage-backed data; it should not
    be interpreted as an owner of storage/cache managers.
    """

    @property
    def meta(self) -> TMetadata:
        """Metadata for this entity."""
        ...

    def set_field(self, name: str, value: Any) -> None:
        """Store a named field for this entity."""
        ...

    def get_field(
        self, name: str, default: Any | None = None, auto_prefetch: bool = True
    ) -> Any:
        """Retrieve a named field for this entity (cache-first)."""
        ...

    def has_field(self, name: str) -> bool:
        """Return True if the field exists (storage or cache)."""
        ...

    def get_field_metadata(self, name: str) -> dict[str, Any] | None:
        """Return storage-level metadata for the given field if available."""
        ...

    def is_field_prefetched(self, name: str) -> bool:
        """Return True if the field is cached for fast access."""
        ...

    def prefetch_field(self, name: str, value: Any) -> None:
        """Cache a field value for fast access."""
        ...

    def try_prefetch_field(self, name: str) -> bool:
        """Attempt to prefetch a field into cache; returns False if missing."""
        ...

    def erase_field(self, name: str) -> None:
        """Remove a field from storage and cache."""
        ...

    def list_fields(self) -> list[str]:
        """List all field names for this entity."""
        ...

    def __hash__(self) -> int:
        """Hash based on entity identity."""
        ...


@runtime_checkable
class IDataRepository(Protocol[TMetadata, TProxy]):
    """Protocol for a repository owning storage and cache infrastructure.

    Defines the interface for repositories that exclusively own and manage
    the IStorageManager and ICacheManager instances. The repository is the
    single source of truth for persistent data and acts as the exclusive
    owner of the underlying storage and caching infrastructure.
    """

    @property
    def storage_manager(self) -> IStorageManager:
        """Storage manager owned by this repository."""
        ...

    @property
    def metadata_index(self) -> IMetadataIndex:
        """Metadata index owned by this repository."""
        ...

    @property
    def cache_manager(self) -> ICacheManager | None:
        """Cache manager owned by this repository."""
        ...

    def __enter__(self) -> Self:
        """Enter a batch write context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the batch write context and flush pending operations."""
        ...

    def __len__(self) -> int:
        """Return the total number of entities in the repository."""
        ...

    def __iter__(self) -> Iterator[TProxy]:
        """Iterate over entities in the repository."""
        ...

    def __getitem__(self, index: int) -> TProxy:
        """Get an entity by index."""
        ...

    def get_proxy(self, uid: str) -> TProxy:
        """Return a proxy for an entity uid currently present in the repository."""
        ...

    def append(self, proxy: TProxy) -> None:
        """Add a single entity to the repository (mutates membership)."""
        ...

    def extend(self, proxies: Iterable[TProxy]) -> None:
        """Add multiple entities to the repository (mutates membership)."""
        ...

    def remove_by_uid(self, uid: str) -> None:
        """Remove an entity from the repository by UID (mutates membership)."""
        ...

    def clear(self, erase: bool = False) -> None:
        """Clear repository membership.

        Args:
            erase: If True, also erase data from storage and cache.
        """
        ...

    def create_view(self) -> IDataView[TMetadata, TProxy]:
        """Create a view for working with a subset of entities."""
        ...


type EntityUID = str
type EntityIndex = int
type FieldName = str


@runtime_checkable
class IDataView(Protocol[TMetadata, TProxy]):
    """Protocol for a view with filtering and data access.

    Defines the interface for views that provide efficient batch
    operations and data access. Views support filtering and provide
    convenient access to data through proxies.
    """

    def __len__(self) -> int:
        """Return the number of entities in this view."""
        ...

    def __iter__(self) -> Iterator[TProxy]:
        """Iterate over entity proxies in view order."""
        ...

    @overload
    def __getitem__(self, index: EntityIndex) -> TProxy: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, uid: EntityUID) -> TProxy: ...

    @overload
    def __getitem__(self, indices: Sequence[EntityIndex]) -> Self: ...

    @overload
    def __getitem__(self, uids: Sequence[EntityUID]) -> Self: ...

    def __getitem__(
        self,
        index: EntityIndex
        | slice
        | EntityUID
        | Sequence[EntityIndex]
        | Sequence[EntityUID],
    ) -> TProxy | Self:
        """Index into the view by position(s) or uid(s)."""
        ...

    def list_uids(self) -> list[EntityUID]:
        """List entity unique identifiers in this view."""
        ...

    def list_fields_by_uid(
        self, *, parallel: ParallelContext | None = None
    ) -> dict[EntityUID, list[FieldName]]:
        """Return a uid -> list(fields) mapping for entities in this view."""
        ...

    def prefetch_fields(
        self,
        *,
        fields_by_uid: dict[EntityUID, list[FieldName]] | None = None,
        fields: list[FieldName] | None = None,
        verify_prefetch: bool = False,
        parallel: ParallelContext | None = None,
    ) -> bool:
        """Prefetch specified fields for entities in this view into cache."""
        ...

    def erase_fields(self, *fields: FieldName) -> None:
        """Remove specified fields from entities in this view."""
        ...

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Remove fields matching regex patterns from entities in this view."""
        ...

    def filter_by_fields(
        self,
        *fields: FieldName,
        inverse_mask: bool = False,
        parallel: ParallelContext | None = None,
    ) -> Self:
        """Filter entities by field presence."""
        ...

    def clear(self, erase: bool = False) -> None:
        """Clear the view.

        Args:
            erase: If True, also erase data from storage and cache.
        """
        ...
