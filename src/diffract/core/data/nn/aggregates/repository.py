"""Repository for aggregate entities.

This module provides the AggregateRepository class for managing
aggregate data storage and membership.
"""

from __future__ import annotations

from typing import ClassVar

from diffract.core.constants import TABLE_AGGREGATES
from diffract.core.data.repository import DataRepository

from .metadata import AggregateMetadata
from .proxy import AggregateProxy
from .view import AggregateView


class AggregateRepository(DataRepository[AggregateMetadata, AggregateProxy]):
    """Repository owning storage/cache/metadata and managing aggregate membership.

    Extends the generic DataRepository with aggregate-specific configuration
    for metadata class, proxy class, view class, storage table, and metadata schema.
    """

    METADATA_CLASS = AggregateMetadata
    PROXY_CLASS = AggregateProxy
    VIEW_CLASS = AggregateView
    TABLE = TABLE_AGGREGATES

    # Schema for MetadataIndex
    METADATA_COLUMNS: ClassVar[dict[str, type]] = {
        "field_name": str,
        "context_models": str,  # JSON-serialized tuple
        "context_params": str,  # JSON-serialized tuple
    }
    METADATA_INDEXES: ClassVar[list[str]] = ["field_name"]

    def get_or_create(
        self,
        field_name: str,
        context_models: tuple[str, ...],
        context_params: tuple[str, ...],
    ) -> AggregateProxy:
        """Get existing aggregate by context or create a new one.

        This method provides deduplication for aggregates with the same context.

        Args:
            field_name: Base field name of the aggregate.
            context_models: Tuple of model IDs participating in this aggregate.
            context_params: Tuple of parameter names participating in this aggregate.

        Returns:
            Existing or newly created AggregateProxy.
        """
        # Create deterministic UID
        uid = AggregateMetadata.create_uid_from_context(
            field_name=field_name,
            context_models=context_models,
            context_params=context_params,
        )

        # Check if already exists in cache or index
        if uid in self._proxy_cache:
            return self._proxy_cache[uid]

        if self.has_uid(uid):
            return self.get_proxy(uid)

        # Create new aggregate
        meta = AggregateMetadata(
            uid=uid,
            field_name=field_name,
            context_models=context_models,
            context_params=context_params,
        )

        return AggregateProxy.create_and_store(meta=meta, repository=self)

    def create_view(self) -> AggregateView:
        """Create a view over all aggregates in this repository.

        Returns:
            AggregateView containing all aggregates.
        """
        return AggregateView(repository=self, uids=None)
