"""Parameter contracts for stored fields, repository membership and scoped reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from diffract.core.data.interface import IDataProxy, IDataRepository, IDataView

from .metadata import ParameterMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from diffract.core.data.nn.instances.repository import InstanceRepository
    from diffract.core.parallel import ParallelContext
    from diffract.core.storage.catalog import ValuePointer

    from .schema import FieldName, ParameterType, ParameterUID


@runtime_checkable
class IParameterProxy(IDataProxy[ParameterMetadata], Protocol):
    """Read a parameter by field name or by a resolved catalog pointer."""

    def update_other_meta(self, updates: dict[str, Any]) -> None:
        """Merge metadata keys into the persisted parameter record."""
        ...

    def read_resolved_field(
        self,
        name: FieldName,
        pointer: ValuePointer,
        *,
        auto_prefetch: bool,
        best_effort: bool = False,
        kernel_input: bool = False,
    ) -> Any:
        """Read the value selected by a catalog pointer."""
        ...


@runtime_checkable
class IParameterRepository(
    IDataRepository[ParameterMetadata, IParameterProxy], Protocol
):
    """Persist parameter identities, checkpoint metadata and computed fields."""

    instances: InstanceRepository

    def prefetch_proxies(
        self, uids: list[str], parallel: ParallelContext | None = None
    ) -> None:
        """Read uncached parameter metadata in a batch."""
        ...

    def refresh_names(self) -> None:
        """Invalidate cached model names."""
        ...

    def merge_target(
        self, source: ParameterMetadata, *, overwrite: bool
    ) -> IParameterProxy:
        """Locate or create the destination for an imported parameter."""
        ...

    def store_metadata(
        self,
        meta: ParameterMetadata,
        *,
        instance_uid: str | None = None,
        update_existing: bool = False,
    ) -> ParameterMetadata:
        """Write parameter metadata using the store-local instance identity."""
        ...

    def remove_membership(self, uids: Sequence[str]) -> int:
        """Delete the selected parameter metadata rows."""
        ...

    def erase_result_groups(
        self, uids: Sequence[str], fields: Sequence[str] | None = None
    ) -> list[tuple[str, str]]:
        """Erase selected result groups and return changed owner and field pairs."""
        ...

    def create_view(self) -> IParameterView:
        """Create a parameter view for working with a subset of parameters."""
        ...


@runtime_checkable
class IParameterView(IDataView[ParameterMetadata, IParameterProxy], Protocol):
    """Select parameters and read their fields in bounded batches."""

    def erase_exact_fields(self, *fields: FieldName) -> int:
        """Erase literal field names and return the number of affected owners."""
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

    def iter_chunks_by_read_budget(
        self,
        *,
        budget_bytes: int | None = None,
        required_fields_by_uid: dict[ParameterUID, list[FieldName]],
        default_field_bytes: int = 1024 * 1024,
        parallel: ParallelContext | None = None,
    ) -> Iterator[Self]:
        """Yield views bounded by the estimated size of the requested fields."""
        ...

    def step_labels(self) -> dict[tuple[str, int], str]:
        """Return recorded display labels keyed by stored checkpoint coordinates."""
        ...

    def coordinates(self, *, whole_store: bool = False) -> list[tuple[str, int]]:
        """Read (instance_uid, step) pairs in the view or the whole store."""
        ...

    def filter_by_coordinates(
        self, coordinates: list[tuple[str, int]]
    ) -> IParameterView:
        """Restrict the view to stored (instance_uid, step) pairs."""
        ...
