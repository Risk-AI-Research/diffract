"""Metadata index interface and protocols.

This module defines the core protocol that all metadata index
implementations must follow. It provides a framework-agnostic
contract for structured metadata storage and querying.
"""

from __future__ import annotations

import types
from typing import Any, Protocol, Self, runtime_checkable


@runtime_checkable
class IMetadataIndex(Protocol):
    """Protocol defining the metadata index interface.

    Provides a unified interface for structured metadata storage with
    SQL-like querying capabilities. All implementations must support
    table definition, CRUD operations, and flexible querying.

    The index is organized as tables containing records identified by UID,
    where each record has typed columns for efficient querying.
    """

    def __enter__(self) -> Self:
        """Enter batch operation context for optimized writes."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit batch context and commit pending operations."""
        ...

    def define_table(
        self,
        table: str,
        columns: dict[str, type],
        indexes: list[str] | None = None,
    ) -> None:
        """Define a table schema with typed columns.

        Creates a table if it doesn't exist. Idempotent - calling multiple
        times with the same schema has no effect.

        Args:
            table: Table name.
            columns: Mapping of column names to Python types (str, int, float, bool).
            indexes: List of column names to create indexes on.
        """
        ...

    def insert(self, table: str, uid: str, **fields: Any) -> None:
        """Insert a new record into the table.

        Args:
            table: Table name.
            uid: Unique identifier for the record.
            **fields: Column values to insert.

        Raises:
            ValueError: If record with uid already exists.
        """
        ...

    def update(self, table: str, uid: str, **fields: Any) -> None:
        """Update an existing record (partial update).

        Args:
            table: Table name.
            uid: Unique identifier of the record to update.
            **fields: Column values to update.

        Raises:
            KeyError: If record with uid doesn't exist.
        """
        ...

    def upsert(self, table: str, uid: str, **fields: Any) -> None:
        """Insert or update a record.

        Args:
            table: Table name.
            uid: Unique identifier for the record.
            **fields: Column values to insert or update.
        """
        ...

    def get(self, table: str, uid: str) -> dict[str, Any] | None:
        """Get a single record by uid.

        Args:
            table: Table name.
            uid: Unique identifier of the record.

        Returns:
            Dictionary of column values, or None if not found.
        """
        ...

    def get_batch(self, table: str, uids: list[str]) -> list[dict[str, Any] | None]:
        """Get multiple records by uids.

        Args:
            table: Table name.
            uids: List of unique identifiers.

        Returns:
            List of dictionaries (or None for missing records), same order as uids.
        """
        ...

    def query(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        where_in: dict[str, list[Any]] | None = None,
        where_like: dict[str, str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Query UIDs matching criteria.

        Args:
            table: Table name.
            where: Exact match conditions {column: value}.
            where_in: IN conditions {column: [values]}.
            where_like: LIKE conditions {column: pattern}.
            order_by: List of column names to sort by.
            limit: Maximum number of results.

        Returns:
            List of UIDs matching the query.
        """
        ...

    def delete(self, table: str, uid: str) -> None:
        """Delete a record by uid.

        Args:
            table: Table name.
            uid: Unique identifier of the record to delete.
        """
        ...

    def delete_batch(self, table: str, uids: list[str]) -> None:
        """Delete multiple records by uids.

        Args:
            table: Table name.
            uids: List of unique identifiers to delete.
        """
        ...

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """Count records in table.

        Args:
            table: Table name.
            where: Optional filter conditions.

        Returns:
            Number of matching records.
        """
        ...

    def distinct(self, table: str, column: str) -> list[Any]:
        """Get distinct values for a column.

        Args:
            table: Table name.
            column: Column name.

        Returns:
            List of distinct values.
        """
        ...

    def list_uids(self, table: str) -> list[str]:
        """List all UIDs in a table.

        Args:
            table: Table name.

        Returns:
            List of all UIDs.
        """
        ...

    def clear(self, table: str | None = None) -> None:
        """Clear data from index.

        Args:
            table: If provided, clear only this table. If None, clear all data.
        """
        ...

    def connect(self) -> None:
        """Initialize connection to the index backend."""
        ...

    def close(self) -> None:
        """Close connection to the index backend."""
        ...
