"""Cache manager interface and protocols.

This module defines the core protocol and type aliases that all cache
manager implementations must follow. It provides a framework-agnostic
contract for cache operations.

Example:
    >>> cache_manager: ICacheManager = get_cache_manager()
    >>> cache_manager.set_field("obj123", "result", computed_value)
    >>> value = cache_manager.get_field("obj123", "result")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

type UID = str


@runtime_checkable
class ICacheManager(Protocol):
    """Protocol defining the cache manager interface.

    Provides a unified interface for cache operations across different
    backends. All cache manager implementations must implement these methods
    to ensure consistent behavior.

    The cache is organized as a two-level structure where objects are
    identified by UID and contain named fields with arbitrary values.
    """

    def has_field(self, obj_uid: UID, field_name: str) -> bool:
        """Check if an object has a specific field in cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to check.

        Returns:
            True if the field exists and is not expired, False otherwise.
        """
        ...

    def get_field(self, obj_uid: UID, field_name: str) -> Any:
        """Retrieve a field value from cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to retrieve.

        Returns:
            The cached value or None if field doesn't exist or is expired.
        """
        ...

    def list_objs_has_field(self, field_name: str) -> list[UID]:
        """List all object UIDs that have a specific field.

        Args:
            field_name: Name of the field to search for.

        Returns:
            List of UIDs for objects that have the specified field.
        """
        ...

    def set_field(self, obj_uid: UID, field_name: str, value: Any) -> None:
        """Store a field value in cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to store.
            value: Value to cache (must be pickle-serializable).

        Raises:
            Exception: If storage operation fails.
        """
        ...

    def erase_field(self, obj_uid: UID, field_name: str) -> None:
        """Remove a specific field from cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to remove.

        Raises:
            Exception: If removal operation fails.
        """
        ...

    def erase_field_for_all(self, field_name: str) -> None:
        """Remove a field from all cached objects.

        Args:
            field_name: Name of the field to remove from all objects.

        Raises:
            Exception: If removal operation fails.
        """
        ...

    def clear(self) -> None:
        """Remove all cached data.

        Raises:
            Exception: If clear operation fails.
        """
        ...

    def get_available_bytes(self) -> int | None:
        """Return estimated available cache capacity in bytes.

        Returns:
            Available bytes, or None if capacity cannot be determined.
        """
        ...
