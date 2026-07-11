"""Core parameter data structures and type definitions.

This module provides the fundamental data structures for representing neural
network parameters, including metadata management, type definitions, and
proxy objects for efficient data access.

Key Components:
    - ParameterType: Extensible enum for parameter classification
    - ParameterMetadata: Immutable metadata container for parameters
    - ParameterDataProxy: Lazy-loading proxy for parameter data access

The ParameterDataProxy provides intelligent caching and storage management,
allowing efficient access to large parameter datasets without loading
everything into memory at once.

Example:
    >>> metadata = ParameterMetadata(
    ...     name="conv1.weight", ptype=ParameterType.DENSE, model_id="resnet50"
    ... )
    >>> proxy = ParameterDataProxy.create_and_store(
    ...     meta=metadata,
    ...     weights=weight_array,
    ...     repository=repository,
    ... )
    >>> weights = proxy.get_field("weights")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.proxy import DataProxy

from .metadata import ParameterMetadata

if TYPE_CHECKING:
    from .interface import IParameterRepository


@dataclass(kw_only=True)
class ParameterDataProxy(DataProxy[ParameterMetadata]):
    """Lazy-loading proxy for neural network parameter data.

    Extends the generic DataProxy with parameter-specific functionality.
    Provides efficient access to parameter data through intelligent caching
    and storage management. Parameters are loaded on-demand and can be
    prefetched for batch operations. The proxy pattern allows working with
    large parameter datasets without memory constraints.

    Example:
        >>> # Create and store a new parameter
        >>> proxy = ParameterDataProxy.create_and_store(
        ...     meta=metadata,
        ...     repository=repository,
        ... )
        >>> # Access fields
        >>> weights = proxy.get_field("weights")
        >>> proxy.set_field("frob_norm", computed_value)

    Attributes:
        meta: Immutable parameter metadata.
        _repository: Repository that owns storage/cache managers.
    """

    meta: ParameterMetadata
    _repository: IParameterRepository = field(repr=False)

    @classmethod
    def get_table(cls) -> str:
        """Return the storage table name for parameters."""
        return TABLE_PARAMETERS

    @classmethod
    def create_and_store(
        cls,
        meta: ParameterMetadata,
        repository: IParameterRepository,
    ) -> ParameterDataProxy:
        """Create parameter proxy and store metadata in index.

        Factory method that creates a new parameter proxy and immediately
        stores the metadata to the metadata index.

        Args:
            meta: Parameter metadata.
            repository: Repository owning storage, cache, and metadata managers.

        Returns:
            New parameter proxy with metadata stored.
        """
        return super().create_and_store(meta=meta, repository=repository)
