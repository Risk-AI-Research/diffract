"""Storage manager interface and protocols.

This module defines the core protocol and type aliases that all storage
manager implementations must follow. It provides a framework-agnostic
contract for persistent data storage operations.

The interface supports hierarchical data organization where objects are
identified by UID, organized into tables, and can contain multiple named
fields with arbitrary values. All storage operations are designed to be
atomic and durable.

Example:
    >>> storage_manager: IStorageManager = get_storage_manager()
    >>> storage_manager.set_field(
    ...     "obj123", "result", computed_array, table="parameters"
    ... )
    >>> value = storage_manager.get_field("obj123", "result", table="parameters")
    >>> all_objects = storage_manager.list_objs(table="parameters")
"""

from __future__ import annotations

import types
from typing import Any, Protocol, Self, runtime_checkable

type UID = str

DEFAULT_TABLE = "default"
"""Default table name for storage operations when not specified."""


@runtime_checkable
class IStorageManager(Protocol):
    """Protocol defining the storage manager interface.

    Provides a unified interface for persistent storage operations across
    different backends. All storage manager implementations must implement
    these methods to ensure consistent behavior and data durability.

    The storage is organized as a three-level structure: tables contain objects
    identified by UID, and objects contain named fields with arbitrary values.
    Storage operations should be atomic and provide durability guarantees.
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
        """Exit batch context and flush pending operations."""
        ...

    def has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Check if an object has a specific field in storage.

        Args:
            obj_uid: Unique identifier for the stored object.
            field_name: Name of the field to check.
            table: Table name for logical data separation.

        Returns:
            True if the field exists in storage, False otherwise.
        """
        ...

    def get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Retrieve a field value from storage.

        Args:
            obj_uid: Unique identifier for the stored object.
            field_name: Name of the field to retrieve.
            table: Table name for logical data separation.

        Returns:
            The stored value.

        Raises:
            KeyError: If the field doesn't exist.
            OSError: If storage read operation fails.
            ValueError: If stored data is corrupted.
        """
        ...

    def list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List all field names for a specific object.

        Args:
            obj_uid: Unique identifier for the stored object.
            table: Table name for logical data separation.

        Returns:
            List of field names that exist for the specified object.

        Raises:
            OSError: If storage read operation fails.
        """
        ...

    def list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """List all object UIDs in storage.

        Args:
            table: Table name for logical data separation.

        Returns:
            List of all unique object identifiers in storage.

        Raises:
            OSError: If storage read operation fails.
        """
        ...

    def list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]:
        """List all object UIDs that have a specific field.

        Args:
            field_name: Name of the field to search for.
            table: Table name for logical data separation.

        Returns:
            List of UIDs for objects that have the specified field.

        Raises:
            OSError: If storage read operation fails.
        """
        ...

    def set_field(
        self, obj_uid: UID, field_name: str, value: Any, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Store a field value in persistent storage.

        Args:
            obj_uid: Unique identifier for the stored object.
            field_name: Name of the field to store.
            value: Value to store (must be serializable by the backend).
            table: Table name for logical data separation.

        Raises:
            OSError: If storage write operation fails.
        """
        ...

    def erase_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a specific field from storage.

        Args:
            obj_uid: Unique identifier for the stored object.
            field_name: Name of the field to remove.
            table: Table name for logical data separation.

        Raises:
            OSError: If storage delete operation fails.
        """
        ...

    def erase_obj(self, obj_uid: UID, *, table: str = DEFAULT_TABLE) -> None:
        """Remove an entire object and all its fields from storage.

        Args:
            obj_uid: Unique identifier for the object to remove.
            table: Table name for logical data separation.

        Raises:
            OSError: If storage delete operation fails.
        """
        ...

    def erase_field_for_all(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a field from all stored objects.

        Args:
            field_name: Name of the field to remove from all objects.
            table: Table name for logical data separation.

        Raises:
            OSError: If storage delete operation fails.
        """
        ...

    def clear(self, *, table: str | None = None) -> None:
        """Remove stored data.

        Args:
            table: If provided, clear only this table. If None, clear all data.

        Raises:
            OSError: If storage clear operation fails.
        """
        ...

    def get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Return stored metadata (dtype/shape/kind) if available.

        Args:
            obj_uid: Unique identifier for the stored object.
            field_name: Name of the field.
            table: Table name for logical data separation.

        Returns:
            Metadata dictionary or None if not available.
        """
        ...
