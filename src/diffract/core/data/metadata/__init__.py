"""Metadata index module for structured metadata storage and querying.

This module provides a generic metadata indexing system that supports
SQL-based filtering and querying of entity metadata.
"""

from __future__ import annotations

from .interface import IMetadataIndex
from .migrations import (
    CURRENT_SCHEMA_VERSION,
    DIFFRACT_APPLICATION_ID,
    MigrationError,
    SchemaVersionError,
    StoreIdentityError,
    UniqueIndexDuplicatesError,
    upgrade_metadata_index,
)
from .sqlite_index import SQLiteMetadataIndex

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DIFFRACT_APPLICATION_ID",
    "IMetadataIndex",
    "MigrationError",
    "SQLiteMetadataIndex",
    "SchemaVersionError",
    "StoreIdentityError",
    "UniqueIndexDuplicatesError",
    "upgrade_metadata_index",
]
