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
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Self

from .interface import DEFAULT_TABLE, UID
from .keys import row_address
from .metadata import infer_value_metadata
from .row_blobs import RowBlobs


class RAMStorageManager(RowBlobs):
    """RAM-only storage manager with table support."""

    backend_name = "ram"

    def __init__(self) -> None:
        # Maps a table and field name to a mapping from object uid to value.
        self._storage: dict[tuple[str, str], dict[UID, Any]] = {}
        # Maps a table, field name and object uid to that value's metadata.
        self._metadata: dict[tuple[str, str, UID], dict[str, Any]] = {}

    def move_blob(self, source: str, target: str) -> None:
        """Move one in-memory value without copying its payload."""
        src_table, src_field, src_uid = row_address(source)
        dst_table, dst_field, dst_uid = row_address(target)
        if not self.has_field(src_uid, src_field, table=src_table):
            raise KeyError(source)
        if source == target:
            return
        if self.has_field(dst_uid, dst_field, table=dst_table):
            raise FileExistsError(target)
        source_values = self._storage[self._key(src_table, src_field)]
        self._storage.setdefault(self._key(dst_table, dst_field), {})[dst_uid] = (
            source_values[src_uid]
        )
        self._metadata[(dst_table, dst_field, dst_uid)] = self._metadata[
            (src_table, src_field, src_uid)
        ]
        self.erase_field(src_uid, src_field, table=src_table)

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
        """Return an independent copy of the stored field."""
        return deepcopy(self._storage[self._key(table, field_name)][obj_uid])

    borrows_kernel_inputs = True

    def borrow_blob(self, key: str) -> Any:
        """Borrow a resident field for kernel execution."""
        table, field, uid = row_address(key)
        return self._storage[self._key(table, field)][uid]

    def owners_with_fields(
        self, uids: Sequence[str], fields: Sequence[str], *, table: str, inverse: bool
    ) -> set[str]:
        """Find owners with all requested fields, or with none when inverted."""
        owners = set(uids)
        for field in fields:
            bucket = self._storage.get((table, field), {})
            if inverse:
                owners.difference_update(bucket)
            else:
                owners.intersection_update(bucket)
        return owners

    def field_metadata_for(
        self, required: dict[str, list[str]], *, table: str
    ) -> dict[tuple[str, str], dict[str, Any] | None]:
        """Read metadata for the requested owner and field pairs."""
        return {
            (uid, field): self._metadata.get((table, field, uid))
            for uid, fields in required.items()
            for field in fields
        }

    def fields_by_owner(
        self, uids: Sequence[str], *, table: str
    ) -> dict[str, list[str]]:
        """List requested owners' fields without revisiting each storage bucket."""
        result = {uid: [] for uid in uids}
        for (stored_table, field), objects in self._storage.items():
            if stored_table == table:
                for uid in result.keys() & objects.keys():
                    result[uid].append(field)
        return result

    def list_fields(
        self, obj_uid: UID | None = None, *, table: str = DEFAULT_TABLE
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
        snapshot = deepcopy(value)
        metadata = infer_value_metadata(snapshot).to_jsonable()
        self._storage.setdefault(key, {})[obj_uid] = snapshot
        self._metadata[(table, field_name, obj_uid)] = metadata

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

    def clear_raw_tables(self, protected: set[str]) -> None:
        """Clear raw tables while preserving catalog blobs."""
        for table in {table for table, _ in self._storage} - protected:
            self.clear(table=table)

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
