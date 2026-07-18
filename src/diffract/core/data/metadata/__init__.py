"""Metadata index module for structured metadata storage and querying.

This module provides a generic metadata indexing system that supports
SQL-based filtering and querying of entity metadata.
"""

from __future__ import annotations

from .interface import IMetadataIndex
from .migrations import (
    CURRENT_SCHEMA_VERSION,
    MigrationError,
    SchemaVersionError,
    UniqueIndexDuplicatesError,
    upgrade_metadata_index,
)
from .sqlite_index import SQLiteMetadataIndex

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "IMetadataIndex",
    "MigrationError",
    "SQLiteMetadataIndex",
    "SchemaVersionError",
    "UniqueIndexDuplicatesError",
    "upgrade_metadata_index",
]
