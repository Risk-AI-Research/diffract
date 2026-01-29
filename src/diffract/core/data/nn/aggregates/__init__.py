"""Aggregates module for managing aggregated data between models/parameters.

This module provides infrastructure for storing and managing aggregate data
(computed aggregations between parameters/models) separately from parameters.

Key Components:
    - AggregateMetadata: Metadata for aggregate entities
    - AggregateProxy: Proxy for accessing aggregate data
    - AggregateView: View for batch operations on aggregates
    - AggregateRepository: Repository managing aggregate storage

Example:
    >>> metadata = AggregateMetadata(
    ...     field_name="l_overlap",
    ...     context_models=("model_a", "model_b"),
    ...     context_params=("layer.weight", "layer.weight"),
    ... )
    >>> proxy = AggregateProxy.create_and_store(
    ...     meta=metadata,
    ...     repository=repository,
    ... )
    >>> proxy.set_field("value", computed_overlap)
"""

from .metadata import AggregateMetadata
from .proxy import AggregateProxy
from .repository import AggregateRepository
from .view import AggregateView

__all__ = [
    "AggregateMetadata",
    "AggregateProxy",
    "AggregateRepository",
    "AggregateView",
]
