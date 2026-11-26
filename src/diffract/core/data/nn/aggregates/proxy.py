"""Storage-backed aggregate data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from diffract.core.constants import TABLE_AGGREGATES
from diffract.core.data.proxy import DataProxy

from .metadata import AggregateMetadata

if TYPE_CHECKING:
    from .repository import AggregateRepository


@dataclass(kw_only=True, eq=False)
class AggregateProxy(DataProxy[AggregateMetadata, "AggregateRepository"]):
    """Read and write fields for an aggregate through its owning repository."""

    meta: AggregateMetadata
    _repository: AggregateRepository = field(repr=False)

    @classmethod
    def get_table(cls) -> str:
        """Return the storage table name for aggregates."""
        return TABLE_AGGREGATES

    @classmethod
    def create_and_store(
        cls,
        meta: AggregateMetadata,
        repository: AggregateRepository,
    ) -> AggregateProxy:
        """Return the proxy for this context and its ordered member UIDs."""
        return repository.get_or_create(
            meta.field_name,
            meta.context_models,
            meta.context_params,
            member_uids=meta.member_uids or None,
            directed=meta.directed,
        )
