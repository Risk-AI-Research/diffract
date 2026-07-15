import threading
from abc import abstractmethod
from contextlib import suppress
from types import TracebackType
from typing import Any, Self

import numpy as np

from diffract.core.storage.interface import DEFAULT_TABLE, UID, IStorageManager
from diffract.core.storage.serialization import encode_value


class BaseStorageManager(IStorageManager):
    """Base storage manager with batching support and table separation."""

    def __init__(
        self,
        batch_size_limit_bytes: int = 50 * 1024 * 1024,
        batch_soft_limit_ratio: float = 0.9,
    ) -> None:
        self._batch_size_limit_bytes = batch_size_limit_bytes
        self._batch_soft_limit_bytes = int(
            batch_size_limit_bytes * batch_soft_limit_ratio
        )

        self._pending_set_bytes: int = 0
        # Keys are (table, obj_uid, field_name)
        self._set_field_sizes: dict[tuple[str, UID, str], int] = {}
        self._set_field_batch: dict[tuple[str, UID, str], Any] = {}
        self._erase_field_batch: set[tuple[str, UID, str]] = set()
        self._erase_obj_batch: set[tuple[str, UID]] = set()
        self._erase_field_for_all_batch: set[tuple[str, str]] = set()

        self._thread_local = threading.local()  # Thread-local context depth
        self._performing_batch_operations: bool = False

    def has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Check if field exists.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            True if field exists, False otherwise.
        """
        self._auto_flush_before_read()
        return self._has_field(obj_uid, field_name, table=table)

    def get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Retrieve field value.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            Field value.

        Raises:
            KeyError: If field not found.
        """
        self._auto_flush_before_read()
        return self._get_field(obj_uid, field_name, table=table)

    def get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Return stored metadata for a field if present."""
        self._auto_flush_before_read()
        return self._get_field_metadata(obj_uid, field_name, table=table)

    def list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List fields.

        Args:
            obj_uid: Optional object unique identifier. If None, lists all fields.
            table: Table name for logical data separation.

        Returns:
            List of field names.
        """
        self._auto_flush_before_read()
        return self._list_fields(obj_uid, table=table)

    def list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """List all objects.

        Args:
            table: Table name for logical data separation.

        Returns:
            List of object unique identifiers.
        """
        self._auto_flush_before_read()
        return self._list_objs(table=table)

    def list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]:
        """List objects with specific field.

        Args:
            field_name: Field name.
            table: Table name for logical data separation.

        Returns:
            List of object unique identifiers.
        """
        self._auto_flush_before_read()
        return self._list_objs_has_field(field_name, table=table)

    def set_field(
        self, obj_uid: UID, field_name: str, value: Any, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Store field value.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            value: Value to store.
            table: Table name for logical data separation.
        """
        key = (table, obj_uid, field_name)
        prev_size = self._set_field_sizes.pop(key, 0)
        if prev_size:
            self._pending_set_bytes = max(0, self._pending_set_bytes - prev_size)

        estimated = self._estimate_value_size(value)
        self._set_field_sizes[key] = estimated
        self._pending_set_bytes += estimated
        self._set_field_batch[key] = value

        if not self._in_batch_context:
            self._flush_set_field_batch()
            return

        if (
            self._batch_size_limit_bytes > 0
            and self._pending_set_bytes >= self._batch_soft_limit_bytes
        ):
            self._perform_batch_operations()

    def erase_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove specific field.

        Args:
            obj_uid: Object unique identifier.
            field_name: Field name.
            table: Table name for logical data separation.
        """
        self._erase_field_batch.add((table, obj_uid, field_name))
        if not self._in_batch_context:
            self._flush_erase_field_batch()

    def erase_obj(self, obj_uid: UID, *, table: str = DEFAULT_TABLE) -> None:
        """Remove entire object and all its fields.

        Args:
            obj_uid: Object unique identifier.
            table: Table name for logical data separation.
        """
        self._erase_obj_batch.add((table, obj_uid))
        if not self._in_batch_context:
            self._flush_erase_obj_batch()

    def erase_field_for_all(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Remove field from all objects.

        Args:
            field_name: Field name.
            table: Table name for logical data separation.
        """
        self._erase_field_for_all_batch.add((table, field_name))
        if not self._in_batch_context:
            self._flush_erase_field_for_all_batch()

    def clear(self, *, table: str | None = None) -> None:
        """Clear data from storage.

        Args:
            table: If provided, clear only this table. If None, clear all data.
        """
        self._set_field_batch.clear()
        self._erase_field_batch.clear()
        self._erase_obj_batch.clear()
        self._erase_field_for_all_batch.clear()

        self._clear(table=table)

    def _auto_flush_before_read(self) -> None:
        if self._in_batch_context and self._has_pending_operations:
            self._perform_batch_operations()

    @property
    def _context_depth(self) -> int:
        """Thread-local context depth for nested batch contexts."""
        return getattr(self._thread_local, "context_depth", 0)

    @_context_depth.setter
    def _context_depth(self, value: int) -> None:
        """Set thread-local context depth."""
        self._thread_local.context_depth = value

    def _estimate_value_size(self, value: Any) -> int:
        """Best-effort estimate of serialized size in bytes."""
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        with suppress(TypeError, ValueError):
            return len(encode_value(value)[0])
        return len(repr(value).encode("utf-8"))

    @property
    def _in_batch_context(self) -> bool:
        """True if inside a batch context (with statement)."""
        return self._context_depth > 0

    @property
    def _has_pending_operations(self) -> bool:
        """True if there are pending buffered operations that require flushing."""
        return bool(
            self._set_field_batch
            or self._erase_field_batch
            or self._erase_obj_batch
            or self._erase_field_for_all_batch
        )

    def _perform_batch_operations(self) -> None:
        """Execute all pending batch operations."""
        if not self._performing_batch_operations:
            self._performing_batch_operations = True
            try:
                self._flush_set_field_batch()
                self._flush_erase_field_batch()
                self._flush_erase_obj_batch()
                self._flush_erase_field_for_all_batch()
            finally:
                self._performing_batch_operations = False

    @abstractmethod
    def _has_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool: ...

    @abstractmethod
    def _get_field(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any: ...

    @abstractmethod
    def _get_field_metadata(
        self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None: ...

    @abstractmethod
    def _list_fields(
        self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]: ...

    @abstractmethod
    def _list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]: ...

    @abstractmethod
    def _list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[UID]: ...

    @abstractmethod
    def _flush_set_field_batch(self) -> None: ...

    @abstractmethod
    def _flush_erase_field_batch(self) -> None: ...

    @abstractmethod
    def _flush_erase_obj_batch(self) -> None: ...

    @abstractmethod
    def _flush_erase_field_for_all_batch(self) -> None: ...

    @abstractmethod
    def _clear(self, *, table: str | None = None) -> None: ...

    def __enter__(self) -> Self:
        """Enter batch mode: queue operations and flush them on exit.

        Supports nested context managers via reference counting. Only the
        outermost entry clears batches; only the outermost exit flushes.
        """
        if self._context_depth == 0:
            self._set_field_batch.clear()
            self._set_field_sizes.clear()
            self._pending_set_bytes = 0
            self._erase_field_batch.clear()
            self._erase_obj_batch.clear()
            self._erase_field_for_all_batch.clear()
        self._context_depth += 1

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit batch write context manager.

        Only the outermost exit flushes operations and commits/rollbacks.
        """
        try:
            if self._context_depth == 1:
                self._perform_batch_operations()
        finally:
            self._context_depth = max(0, self._context_depth - 1)
