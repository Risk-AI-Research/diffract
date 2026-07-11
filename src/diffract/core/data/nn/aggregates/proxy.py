"""Proxy for aggregate entities.

This module provides the AggregateProxy class for accessing
aggregated data stored in the aggregates table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from diffract.core.constants import TABLE_AGGREGATES
from diffract.core.data.proxy import DataProxy

from .metadata import AggregateMetadata

if TYPE_CHECKING:
    from .repository import AggregateRepository


@dataclass(kw_only=True)
class AggregateProxy(DataProxy[AggregateMetadata]):
    """Lazy-loading proxy for aggregate data.

    Extends the generic DataProxy with aggregate-specific configuration.
    Aggregates are stored in the TABLE_AGGREGATES table and represent
    computed aggregations between models/parameters.

    Attributes:
        meta: Immutable aggregate metadata.
        _repository: Repository that owns storage/cache managers.
    """

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
        """Create aggregate proxy and store metadata in index.

        Args:
            meta: Aggregate metadata.
            repository: Repository owning storage, cache, and metadata managers.

        Returns:
            New aggregate proxy with metadata stored.
        """
        return super().create_and_store(meta=meta, repository=repository)
