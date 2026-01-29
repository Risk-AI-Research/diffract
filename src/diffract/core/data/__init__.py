"""Generic data abstractions for diffract.

This module provides domain-agnostic protocols and implementations for
data management, including:

- IMetadata: Protocol for serializable metadata with unique identifier
- IDataProxy: Protocol for lazy-loading storage-backed data
- IDataView: Protocol for batch operations on filtered collections
- IDataRepository: Protocol for ownership of storage/cache infrastructure

- DataProxy: Generic implementation of IDataProxy
- DataView: Generic implementation of IDataView
- DataRepository: Generic implementation of IDataRepository

These abstractions are specialized by domain-specific modules:
- core/data/nn/params: Parameter-specific implementations
- core/data/nn/relations: Relation-specific implementations
"""

from .interface import EntityIndex, EntityUID, FieldName
from .proxy import DataProxy
from .repository import DataRepository
from .view import DataView

__all__ = [
    "DataProxy",
    "DataRepository",
    "DataView",
    "EntityIndex",
    "EntityUID",
    "FieldName",
]
