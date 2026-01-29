"""Hybrid storage manager combining light and heavy backends.

Routes small/structured data to light storage and large arrays to heavy storage.
Uses a sentinel value ("__HEAVY__") in light storage to indicate the data lives
in the heavy backend.

Thread-safety: inherits the concurrency models of the underlying managers.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any, Self

import numpy as np

from .interface import DEFAULT_TABLE, UID, IStorageManager


class HybridStorageManager(IStorageManager):
    """Storage manager that routes data between light and heavy backends.

    Small values and metadata go to light storage for fast random access. Large
    data (exceeding threshold) are stored in heavy storage with a sentinel value
    in light storage to mark the indirection.

    This provides a unified interface while allowing optimization for different
    data access patterns and storage backends.

    Example:
        >>> # Local SQLite + Cloud Zarr
        >>> light = SQLiteStorageManager("meta.db")
        >>> heavy = ZarrStorageManager("s3://bucket/data")
        >>> storage = HybridStorageManager(
        ...     light, heavy, array_threshold=128 * 1024 * 1024
        ... )
        >>> with storage:
        ...     storage.set_field("param_001", "weights", large_array)

    Args:
        light_storage: Storage backend for small/structured data and metadata.
        heavy_storage: Storage backend for large data arrays.
        array_threshold: Byte threshold above which data goes to heavy storage.
    """

    _HEAVY_SENTINEL = "__HEAVY__"

    def __init__(
        self,
        light_storage: IStorageManager,
        heavy_storage: IStorageManager,
        array_threshold: int = 128 * 1024 * 1024,
    ) -> None:
        """Initialize hybrid storage manager.

        Args:
            light_storage: Storage for small data and metadata.
            heavy_storage: Blob storage for large data arrays.
            array_threshold: Byte threshold for routing to heavy storage.
        """
        self.light = light_storage
        self.heavy = heavy_storage
        self.array_threshold = array_threshold

    def _should_use_heavy(self, value: Any) -> bool:
        """Determine if value should be stored in heavy storage."""
        if isinstance(value, np.ndarray):
            return value.nbytes >= self.array_threshold
        return False

    def has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Check if a field exists in storage."""
        return self.light.has_field(obj_uid, field_name, table=table)

    def get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Retrieve field value, following heavy storage indirection if needed.

        Raises:
            KeyError: If the field doesn't exist.
        """
        value = self.light.get_field(obj_uid, field_name, table=table)
        if isinstance(value, str) and value == self._HEAVY_SENTINEL:
            return self.heavy.get_field(obj_uid, field_name, table=table)
        return value

    def set_field(
        self, obj_uid: UID, field_name: str, value: Any, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Store field value, routing data according to routing rules.

        Large or heavy data goes to heavy storage with a sentinel in light storage.
        Smaller values go directly to light storage. If a field moves between
        backends, the old copy is erased.
        """
        if self._should_use_heavy(value):
            self.light.set_field(obj_uid, field_name, self._HEAVY_SENTINEL, table=table)
            self.heavy.set_field(obj_uid, field_name, value, table=table)
        else:
            self.light.set_field(obj_uid, field_name, value, table=table)
            self.heavy.erase_field(obj_uid, field_name, table=table)

    def erase_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a field from storage."""
        self.light.erase_field(obj_uid, field_name, table=table)
        self.heavy.erase_field(obj_uid, field_name, table=table)

    def erase_obj(self, obj_uid: UID, *, table: str = DEFAULT_TABLE) -> None:
        """Remove an object and all its fields from storage."""
        self.light.erase_obj(obj_uid, table=table)
        self.heavy.erase_obj(obj_uid, table=table)

    def list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List all fields for an object."""
        return self.light.list_fields(obj_uid, table=table)

    def list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """List all objects."""
        return self.light.list_objs(table=table)

    def list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]:
        """List objects having a field."""
        return self.light.list_objs_has_field(field_name, table=table)

    def erase_field_for_all(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a field from all objects.

        Delegates to underlying backends which handle batch optimization.
        """
        self.light.erase_field_for_all(field_name, table=table)
        self.heavy.erase_field_for_all(field_name, table=table)

    def get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Return stored metadata from whichever backend holds the value."""
        try:
            value = self.light.get_field(obj_uid, field_name, table=table)
        except KeyError:
            return None

        if isinstance(value, str) and value == self._HEAVY_SENTINEL:
            return self.heavy.get_field_metadata(obj_uid, field_name, table=table)
        return self.light.get_field_metadata(obj_uid, field_name, table=table)

    def clear(self, *, table: str | None = None) -> None:
        """Clear data from both backends.

        Args:
            table: If provided, clear only this table. If None, clear all data.
        """
        self.light.clear(table=table)
        self.heavy.clear(table=table)

    def connect(self) -> None:
        """Initialize connections for backends that need it."""
        if hasattr(self.light, "connect"):
            self.light.connect()
        if hasattr(self.heavy, "connect"):
            self.heavy.connect()

    def close(self) -> None:
        """Close both backends. Raises ExceptionGroup on errors."""
        exceptions: list[Exception] = []

        for manager in [self.light, self.heavy]:
            try:
                if hasattr(manager, "close"):
                    manager.close()
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        if exceptions:
            raise ExceptionGroup("Error occurred during closing", exceptions)

    def __enter__(self) -> Self:
        """Enter batch context for both backends."""
        if hasattr(self.light, "__enter__"):
            self.light.__enter__()
        if hasattr(self.heavy, "__enter__"):
            self.heavy.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit batch context for both backends. Raises ExceptionGroup on errors."""
        exceptions: list[Exception] = []

        for manager in [self.light, self.heavy]:
            try:
                if hasattr(manager, "__exit__"):
                    manager.__exit__(exc_type, exc_val, exc_tb)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        if exceptions:
            raise ExceptionGroup("Error occurred during exiting contexts", exceptions)

    def __del__(self) -> None:
        """Best-effort cleanup; suppresses exceptions during GC."""
        with contextlib.suppress(Exception):
            self.close()
