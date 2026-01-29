"""Metadata index module for structured metadata storage and querying.

This module provides a generic metadata indexing system that supports
SQL-based filtering and querying of entity metadata.
"""

from __future__ import annotations

from .interface import IMetadataIndex
from .sqlite_index import SQLiteMetadataIndex

__all__ = [
    "IMetadataIndex",
    "SQLiteMetadataIndex",
]
