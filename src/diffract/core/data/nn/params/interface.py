"""Parameter repository and view interfaces and protocols.

This module defines the core protocols for parameter management in diffract.
It provides framework-agnostic contracts for two key concepts:

1. IParameterRepository: Exclusively owns and manages IStorageManager and ICacheManager
   instances, serving as the single source of truth for persistent parameter data.
   The repository controls the lifecycle of storage and cache resources and acts
   as the entry point for creating parameter views.

2. IParameterView: Provides numpy-like views for batch operations on parameters,
   supporting filtering and efficient data access. Views contain methods for data
   operations (prefetching, field management) that work only on parameters within
   the view, similar to numpy array slicing. Views access storage and cache through
   the repository but do not own these resources.

The interfaces support rich filtering capabilities by parameter name,
type, model ID, and custom fields, as well as batch operations for
efficient parameter management.

Example:
    >>> repository: IParameterRepository = get_parameter_repository()
    >>> view: IParameterView = repository.create_view()
    >>> dense_params = view.filter_by_ptype("DENSE")
    >>> dense_params.prefetch_fields(fields=["weights", "gradients"])
    >>> dense_params.erase_fields("temporary_field")
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
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
    from diffract.core.data.metadata.interface import IMetadataIndex
    from diffract.core.data.nn.params.schema import (
        FieldName,
        ParameterIndex,
        ParameterType,
        ParameterUID,
    )
    from diffract.core.parallel import ParallelContext
    from diffract.core.storage.interface import IStorageManager

T = TypeVar("T")


@runtime_checkable
class IParameterProxy(Protocol):
    """Protocol for a proxy object representing a single parameter.

    This protocol describes how the rest of the system interacts with a single
    parameter (conceptually: a single "row" in storage). Importantly, the proxy
    is a *handle* for interacting with storage-backed data; it should not be
    interpreted as an owner of storage/cache managers.

    Implementations may internally use storage/cache managers, but the public
    interface is expressed in terms of parameter identity/metadata and field
    operations.
    """

    def set_field(self, name: FieldName, value: Any) -> None:
        """Store a named field for this parameter."""
        ...

    def get_field(
        self, name: FieldName, default: T | None = None, auto_prefetch: bool = True
    ) -> Any | T:
        """Retrieve a named field for this parameter (cache-first)."""
        ...

    def has_field(self, name: FieldName) -> bool:
        """Return True if the field exists (storage or cache)."""
        ...

    def get_field_metadata(self, name: FieldName) -> dict[str, Any] | None:
        """Return storage-level metadata for the given field if available."""
        ...

    def is_field_prefetched(self, name: FieldName) -> bool:
        """Return True if the field is cached for fast access."""
        ...

    def prefetch_field(self, name: FieldName, value: Any) -> None:
        """Cache a field value for fast access."""
        ...

    def try_prefetch_field(self, name: FieldName) -> bool:
        """Attempt to prefetch a field into cache; returns False if missing."""
        ...

    def erase_field(self, name: FieldName) -> None:
        """Remove a field from storage and cache."""
        ...

    def list_fields(self) -> list[FieldName]:
        """List all field names for this parameter."""
        ...

    def __hash__(self) -> int:
        """Hash based on parameter identity."""
        ...


@runtime_checkable
class IParameterRepository(Protocol):
    """Protocol for a parameter repository owning storage and cache infrastructure.

    Defines the interface for parameter repositories that exclusively own and manage
    the IStorageManager and ICacheManager instances. The repository is the single
    source of truth for persistent parameter data and acts as the exclusive owner
    of the underlying storage and caching infrastructure.

    As the owner of storage and cache managers, the repository:
    - Controls the lifecycle of storage and cache resources
    - Ensures data consistency across storage and cache layers
    - Provides access to owned managers via properties
    - Manages repository membership (adding/removing parameters)
    - Provides the foundation for creating parameter views
    - Delegates data operations to views for efficient batch processing

    Views created by the repository share access to the owned storage and cache
    but do not own these resources themselves. The repository exposes its owned
    managers through properties for introspection and advanced usage.
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
        """Cache manager owned by this repository (optional)."""
        ...

    def __enter__(self) -> Self:
        """Enter a batch write context (delegated to the owned storage manager)."""
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
        """Return the total number of parameters in the repository.

        Returns:
            Number of parameters currently stored in the repository.
        """
        ...

    def __iter__(self) -> Iterator[IParameterProxy]:
        """Iterate over parameters in the repository.

        Note:
            Iteration order is implementation-defined.
        """
        ...

    def __getitem__(self, index: int) -> IParameterProxy:
        """Get a parameter by index.

        Note:
            Indexing is provided for quasi list-compatibility; ordering is
            implementation-defined and may not be stable across sessions.
        """
        ...

    def get_proxy(self, uid: ParameterUID) -> IParameterProxy:
        """Return a proxy for a parameter uid currently present in the repository."""
        ...

    def append(self, param: IParameterProxy) -> None:
        """Add a single parameter to the repository (mutates membership)."""
        ...

    def extend(self, params: Iterable[IParameterProxy]) -> None:
        """Add multiple parameters to the repository (mutates membership)."""
        ...

    def remove_by_uid(self, uid: ParameterUID) -> None:
        """Remove a parameter from the repository by UID (mutates membership).

        Args:
            uid: Unique identifier of the parameter to remove.
        """
        ...

    def clear(self, erase: bool = False) -> None:
        """Clear repository membership.

        Args:
            erase: If True, also erase data from storage and cache.
        """
        ...

    def create_view(self) -> IParameterView:
        """Create a parameter view for working with a subset of parameters."""
        ...


@runtime_checkable
class IParameterView(Protocol):
    """Protocol for a parameter view with rich filtering and data access.

    Defines the interface for parameter views that provide efficient batch
    operations and data access for neural network parameters. Views support
    rich filtering by various criteria and provide convenient access to
    parameter data through proxies.

    This protocol represents a working view of parameters that can be filtered,
    iterated, and manipulated as a collection. It provides access to data operations
    (prefetching, field management) for the parameters within this view only,
    similar to numpy's array views.

    This protocol is the main interface for parameter data operations. It may be
    extended with new methods over time, but existing methods should remain
    backwards compatible to ensure compatibility across implementations.
    """

    def __len__(self) -> int:
        """Return the number of parameters in this view.

        Returns:
            Number of parameters in the view.
        """
        ...

    @overload
    def __getitem__(self, index: ParameterIndex) -> IParameterProxy: ...

    @overload
    def __getitem__(self, index: slice) -> IParameterView: ...

    @overload
    def __getitem__(self, uid: ParameterUID) -> IParameterProxy: ...

    @overload
    def __getitem__(self, indices: Sequence[ParameterIndex]) -> IParameterView: ...

    @overload
    def __getitem__(self, uids: Sequence[ParameterUID]) -> IParameterView: ...

    def __getitem__(
        self,
        index: ParameterIndex
        | slice
        | ParameterUID
        | Sequence[ParameterIndex]
        | Sequence[ParameterUID],
    ) -> IParameterProxy | IParameterView:
        """Index into the view by position(s) or uid(s).

        Args:
            index: One of:
                - ParameterIndex: zero-based positional index, returns a proxy
                - slice: positional slice, returns a new view
                - ParameterUID: a single uid, returns a proxy
                - Sequence[ParameterIndex]: positional indices, returns a new view
                - Sequence[ParameterUID]: uids, returns a new view

        Returns:
            Either a single parameter proxy (for int/str) or a new view (for
            slice/sequence).
        """
        ...

    def list_uids(self) -> list[ParameterUID]:
        """List parameter unique identifiers in this view.

        Returns:
            List of UIDs for parameters in this view.
        """
        ...

    def list_fields_by_uid(
        self, *, parallel: ParallelContext | None = None
    ) -> dict[ParameterUID, list[FieldName]]:
        """Return a uid -> list(fields) mapping for parameters in this view."""
        ...

    def prefetch_fields(
        self,
        *,
        fields_by_uid: dict[ParameterUID, list[FieldName]] | None = None,
        fields: list[FieldName] | None = None,
        verify_prefetch: bool = False,
        parallel: ParallelContext | None = None,
    ) -> bool:
        """Prefetch specified fields for parameters in this view into cache.

        Args:
            fields_by_uid: Mapping of uid -> list of fields to prefetch.
            fields: List of fields to prefetch for all parameters in the view.
            verify_prefetch: If True, verify cached presence after prefetch.
            parallel: Optional per-method parallel context for prefetching.

        Returns:
            True if all requested fields were successfully prefetched.
        """
        ...

    def erase_fields(self, *fields: FieldName) -> None:
        """Remove specified fields from parameters in this view.

        Args:
            *fields: Field names to remove from parameters.
        """
        ...

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Remove fields matching regex patterns from parameters in this view.

        Args:
            *patterns: Regex patterns for field names to remove.
        """
        ...

    def filter_by_name(self, *names: str, inverse_mask: bool = False) -> IParameterView:
        """Filter parameters by name.

        Args:
            *names: Parameter names to filter by.
            inverse_mask: If True, exclude matching names instead.

        Returns:
            New view containing only matching parameters.
        """
        ...

    def filter_by_ptype(self, *ptypes: str | ParameterType) -> IParameterView:
        """Filter parameters by type.

        Args:
            *ptypes: Parameter types to filter by (strings or ParameterType).

        Returns:
            New view containing only matching parameters.
        """
        ...

    def filter_by_model_id(
        self, *model_ids: str, inverse_mask: bool = False
    ) -> IParameterView:
        """Filter parameters by model ID.

        Args:
            *model_ids: Model IDs to filter by.
            inverse_mask: If True, exclude matching model IDs instead.

        Returns:
            New view containing only matching parameters.
        """
        ...

    def filter_by_fields(
        self,
        *fields: str,
        inverse_mask: bool = False,
        parallel: ParallelContext | None = None,
    ) -> IParameterView:
        """Filter parameters by field presence.

        Args:
            *fields: Field names that parameters must have.
            inverse_mask: If True, exclude parameters with these fields.
            parallel: Optional per-method parallel context for field checks.

        Returns:
            New view containing only matching parameters.
        """
        ...

    def clear(self, erase: bool = False) -> None:
        """Clear the view.

        Args:
            erase: If True, also erase data from storage and cache.
        """
        ...
