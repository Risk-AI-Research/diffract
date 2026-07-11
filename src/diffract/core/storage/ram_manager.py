"""RAM-based storage manager implementation for diffract.

Provides high-performance, RAM-only storage (no disk persistence).

Example:
>>> storage = RAMStorageManager()
>>> storage.set_field("obj1", "data", [1, 2, 3], table="parameters")
>>> storage.get_field("obj1", "data", table="parameters")
[1, 2, 3]
>>> storage.clear()
"""

from __future__ import annotations

import types
from typing import Any, Self

from .interface import DEFAULT_TABLE, UID, IStorageManager
from .metadata import infer_value_metadata


class RAMStorageManager(IStorageManager):
    """RAM-only storage manager with table support."""

    def __init__(self) -> None:
        # Maps a table and field name to a mapping from object uid to value.
        self._storage: dict[tuple[str, str], dict[UID, Any]] = {}
        # Maps a table, field name and object uid to that value's metadata.
        self._metadata: dict[tuple[str, str, UID], dict[str, Any]] = {}

    def _key(self, table: str, field_name: str) -> tuple[str, str]:
        """Create storage key from table and field name."""
        return (table, field_name)

    def has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Return True if the field exists in memory."""
        key = self._key(table, field_name)
        return key in self._storage and obj_uid in self._storage[key]

    def get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Get a field value from memory."""
        key = self._key(table, field_name)
        return self._storage[key][obj_uid]

    def list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List field names (optionally only those containing the given object)."""
        if obj_uid is None:
            return [field_name for (tbl, field_name) in self._storage if tbl == table]
        return [
            field_name
            for (tbl, field_name), objs in self._storage.items()
            if tbl == table and obj_uid in objs
        ]

    def list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """List all object UIDs in specified table."""
        seen: set[str] = set()
        for (tbl, _), objs in self._storage.items():
            if tbl == table:
                seen.update(objs.keys())
        return list(seen)

    def list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]:
        """List objects that have the given field."""
        key = self._key(table, field_name)
        if key in self._storage:
            return list(self._storage[key].keys())
        return []

    def set_field(
        self, obj_uid: UID, field_name: str, value: Any, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Store a field value in memory."""
        key = self._key(table, field_name)
        if key not in self._storage:
            self._storage[key] = {}
        self._storage[key][obj_uid] = value
        self._metadata[(table, field_name, obj_uid)] = infer_value_metadata(
            value
        ).to_jsonable()

    def erase_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a field for an object."""
        if self.has_field(obj_uid, field_name, table=table):
            key = self._key(table, field_name)
            del self._storage[key][obj_uid]
            self._metadata.pop((table, field_name, obj_uid), None)
            if not self._storage[key]:
                del self._storage[key]

    def erase_obj(self, obj_uid: UID, *, table: str = DEFAULT_TABLE) -> None:
        """Remove an object and all its fields from specified table."""
        keys_to_remove = []
        for (tbl, field_name), objs in self._storage.items():
            if tbl == table and obj_uid in objs:
                del objs[obj_uid]
                self._metadata.pop((table, field_name, obj_uid), None)
                if not objs:
                    keys_to_remove.append((tbl, field_name))

        for key in keys_to_remove:
            del self._storage[key]

    def erase_field_for_all(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove a field from all objects in specified table."""
        key = self._key(table, field_name)
        if key in self._storage:
            del self._storage[key]
        to_delete = [k for k in self._metadata if k[0] == table and k[1] == field_name]
        for k in to_delete:
            del self._metadata[k]

    def clear(self, *, table: str | None = None) -> None:
        """Remove data from memory.

        Args:
            table: If provided, clear only this table. If None, clear all data.
        """
        if table is None:
            self._storage.clear()
            self._metadata.clear()
        else:
            keys_to_remove = [k for k in self._storage if k[0] == table]
            for k in keys_to_remove:
                del self._storage[k]
            meta_to_remove = [k for k in self._metadata if k[0] == table]
            for k in meta_to_remove:
                del self._metadata[k]

    def get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Return stored metadata for a field if available."""
        return self._metadata.get((table, field_name, obj_uid))

    def __enter__(self) -> Self:
        """Enter context; returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context; no-op for RAM backend."""
        return
