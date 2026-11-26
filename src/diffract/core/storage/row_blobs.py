"""Physical-key operations for row-oriented value managers."""

from __future__ import annotations

from typing import Any, Protocol

from .interface import IPhysicalStorageManager
from .keys import row_address


class RowBlobs(IPhysicalStorageManager, Protocol):
    """Read catalog-pointed cells using the backend's row operations."""

    def put_blob(self, key: str, value: Any) -> None:
        """Write a value under its catalog key."""
        table, field, uid = row_address(key)
        self.set_field(uid, field, value, table=table)

    def read_blob(self, key: str) -> Any:
        """Read a value through its physical key."""
        table, field, uid = row_address(key)
        return self.get_field(uid, field, table=table)

    def blob_metadata(self, key: str) -> dict[str, Any] | None:
        """Return the metadata saved alongside a physical value."""
        table, field, uid = row_address(key)
        return self.get_field_metadata(uid, field, table=table)

    def delete_blob(self, key: str) -> None:
        """Physically remove a value during collection."""
        table, field, uid = row_address(key)
        self.erase_field(uid, field, table=table)
