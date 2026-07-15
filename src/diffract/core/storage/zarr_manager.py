"""Zarr-backed storage manager (cloud-friendly via fsspec)."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote, unquote, urlparse

import numpy as np

import diffract.core.utils.imports as import_utils
from diffract.core.constants import (
    STORAGE_ATTR_META,
    STORAGE_ATTR_TYPE,
    ZARR_INDEX_FIELDS,
    ZARR_INDEX_GROUP,
    ZARR_INDEX_OBJS,
)
from diffract.core.utils.exceptions import format_exception_message

from .base_manager import BaseStorageManager
from .interface import DEFAULT_TABLE, UID
from .metadata import infer_value_metadata
from .serialization import decode_value, encode_value

_TYPE_NDARRAY = "ndarray"

_READONLY_ERROR = "ZarrStorageManager is readonly; write operation is not allowed"


class _TableRemovalInfo(NamedTuple):
    """Tracks removed objects and fields for a table during batch operations."""

    removed_objs: set[str]
    removed_fields: set[str]


if TYPE_CHECKING:
    from typing import Self

logger = logging.getLogger(__name__)


if not import_utils.is_available("zarr"):
    logger.debug("zarr not available, disabling Zarr storage manager")

    class ZarrStorageManager(BaseStorageManager):
        """Stub implementation when zarr is not available."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError because the optional dependency is missing."""
            msg = (
                "zarr package not available; "
                'install with: pip install "diffract-core[zarr]"'
            )
            raise ImportError(msg)

else:
    zarr = import_utils.require("zarr")

    class ZarrStorageManager(BaseStorageManager):
        """Persistent storage manager using Zarr v3 with fsspec-compatible stores.

        Layout:
            <root>/<table>/<field_name>/<obj_uid>

        Each value is stored as a Zarr array with attributes:
            - type: "ndarray" | "json" | "bytes"
            - value_meta: JSON string from infer_value_metadata(...)
        """

        def __init__(
            self,
            store_url: str,
            *,
            storage_options: dict[str, Any] | None = None,
            root: str = "root",
            readonly: bool = False,
            chunks: bool | tuple[int, ...] | None = None,
            sort_list_objs: bool = True,
            compressor: str | None = "lz4",
            target_chunk_mb: float = 16.0,
            lazy_index_sync: bool = True,
            max_concurrency: int = 16,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._store_url = store_url
            self._storage_options = storage_options or {}
            self._root = root
            self._readonly = readonly
            self._chunks = chunks
            self._sort_list_objs = sort_list_objs
            self._compressor = compressor
            self._target_chunk_bytes = int(target_chunk_mb * 1024 * 1024)
            self._lazy_index_sync = lazy_index_sync
            self._max_concurrency = max_concurrency

            self._write_lock = threading.Lock()
            self._store: Any | None = None
            self._root_group: Any | None = None

            # Per-table index caches
            self._index_cache_objs: dict[str, frozenset[str]] = {}
            self._index_cache_fields: dict[str, frozenset[str]] = {}
            self._index_dirty: dict[str, bool] = {}

        def connect(self) -> None:
            """Initialize store and root group (idempotent)."""
            if self._root_group is not None:
                return

            with self._write_lock:
                if self._root_group is not None:
                    return
                store = self._create_store()
                mode = "r" if self._readonly else "a"
                try:
                    root_group = zarr.open_group(
                        store=store, path=self._root, mode=mode
                    )
                except Exception as exc:
                    msg = (
                        "Failed to open Zarr store at "
                        f"'{self._store_url}': {format_exception_message(exc)}"
                    )
                    raise OSError(msg) from exc

                self._store = store
                self._root_group = root_group

        def close(self) -> None:
            """Close underlying store (best-effort). Syncs index if dirty."""
            with self._write_lock:
                self._sync_dirty_indexes()
                store = self._store
                self._store = None
                self._root_group = None
                self._index_cache_objs.clear()
                self._index_cache_fields.clear()
                self._index_dirty.clear()

            if store is not None:
                self._close_store(store)

        def _sync_dirty_indexes(self) -> None:
            """Sync all dirty indexes to persistent store."""
            if self._readonly or self._root_group is None:
                return
            for table, is_dirty in self._index_dirty.items():
                if not is_dirty:
                    continue
                try:
                    self._sync_index(table)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to sync index on close for table %s: %s",
                        table,
                        format_exception_message(exc),
                        exc_info=True,
                    )

        def _close_store(self, store: Any) -> None:
            """Close store handle if it has a close method."""
            try:
                close_fn = getattr(store, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to close Zarr store: %s",
                    format_exception_message(exc),
                    exc_info=True,
                )

        def _create_store(self) -> Any:
            parsed = urlparse(self._store_url)
            scheme = parsed.scheme
            if scheme in ("", "file"):
                from zarr.storage import LocalStore  # type: ignore[import-untyped]

                local_path = parsed.path if scheme == "file" else self._store_url
                return LocalStore(local_path)

            import fsspec
            from zarr.storage import FsspecStore  # type: ignore[import-untyped]

            fs_kwargs = {
                **self._storage_options,
                "asynchronous": True,
                "max_concurrency": self._max_concurrency,
            }
            fs = fsspec.filesystem(scheme, **fs_kwargs)
            path = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
            return FsspecStore(fs=fs, path=path)

        def _get_root(self) -> Any:
            if self._root_group is None:
                self.connect()
            assert self._root_group is not None
            return self._root_group

        def _get_table_group(self, table: str) -> Any:
            """Get or create table group."""
            root = self._get_root()
            if not self._readonly:
                table_group = root.require_group(table)
                self._ensure_index(table_group, table)
                return table_group
            if table in root:
                return root[table]
            return None

        def _encode_component(self, value: str) -> str:
            return quote(value, safe="")

        def _decode_component(self, value: str) -> str:
            return unquote(value)

        def _compute_optimal_chunks(
            self, shape: tuple[int, ...], dtype: np.dtype
        ) -> tuple[int, ...]:
            """Compute chunks targeting _target_chunk_bytes for optimal cloud I/O."""
            if not shape:
                return ()

            item_size = dtype.itemsize
            total_elements = max(1, self._target_chunk_bytes // item_size)

            ndim = len(shape)
            if ndim == 2:  # noqa: PLR2004
                rows_per_chunk = max(1, total_elements // shape[1])
                return (min(rows_per_chunk, shape[0]), shape[1])

            if ndim == 1:
                return (min(total_elements, shape[0]),)

            chunk_side = max(1, int(total_elements ** (1.0 / ndim)))
            return tuple(min(chunk_side, dim) for dim in shape)

        _MIN_BYTES_FOR_COMPRESSION = 1024

        def _get_compressors(self, nbytes: int) -> list[Any] | None:
            """Get compressor list for array creation (None if no compression)."""
            if not self._compressor or nbytes < self._MIN_BYTES_FOR_COMPRESSION:
                return None
            try:
                from zarr.codecs import BloscCodec

                return [BloscCodec(cname=self._compressor, clevel=5, shuffle="shuffle")]
            except ImportError:
                return None

        def _ensure_index(self, table_group: Any, _table: str) -> None:
            """Ensure index exists for a table."""
            idx_group = table_group.require_group(ZARR_INDEX_GROUP)
            if ZARR_INDEX_OBJS not in idx_group:
                self._write_index_set(idx_group, ZARR_INDEX_OBJS, set())
            if ZARR_INDEX_FIELDS not in idx_group:
                self._write_index_set(idx_group, ZARR_INDEX_FIELDS, set())

        def _read_index_set(self, table_group: Any, name: str) -> set[str] | None:
            idx_group = table_group.get(ZARR_INDEX_GROUP)
            if idx_group is None or name not in idx_group:
                return None
            dataset = idx_group[name]
            try:
                payload = bytes(dataset[:]).decode("utf-8")
                values = json.loads(payload)
                if not isinstance(values, list):
                    msg = "Index payload is not a list"
                    raise TypeError(msg)  # noqa: TRY301
                return {str(v) for v in values}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to read Zarr index '%s': %s",
                    name,
                    format_exception_message(exc),
                    exc_info=True,
                )
                return None

        def _write_index_set(self, idx_group: Any, name: str, values: set[str]) -> None:
            payload = json.dumps(sorted(values), ensure_ascii=False).encode("utf-8")
            data = np.frombuffer(payload, dtype=np.uint8)
            idx_group.create_array(name, data=data, overwrite=True)

        def _get_index_cache(self, table: str, name: str) -> frozenset[str] | None:
            if name == ZARR_INDEX_OBJS:
                return self._index_cache_objs.get(table)
            return self._index_cache_fields.get(table)

        def _set_index_cache(
            self, table: str, name: str, values: frozenset[str] | None
        ) -> None:
            if name == ZARR_INDEX_OBJS:
                if values is not None:
                    self._index_cache_objs[table] = values
                elif table in self._index_cache_objs:
                    del self._index_cache_objs[table]
            elif values is not None:
                self._index_cache_fields[table] = values
            elif table in self._index_cache_fields:
                del self._index_cache_fields[table]

        def _load_index_or_scan(
            self,
            table_group: Any,
            table: str,
            name: str,
            scan_func: Callable[[Any], set[str]],
        ) -> set[str]:
            cached = self._get_index_cache(table, name)
            if cached is not None:
                return set(cached)

            stored = self._read_index_set(table_group, name)
            if stored is not None:
                self._set_index_cache(table, name, frozenset(stored))
                return stored

            actual = scan_func(table_group)
            if not self._readonly:
                idx_group = table_group.require_group(ZARR_INDEX_GROUP)
                self._write_index_set(idx_group, name, actual)
                self._set_index_cache(table, name, frozenset(actual))
            return actual

        def _scan_actual_fields(self, table_group: Any) -> set[str]:
            return {k for k in table_group if k != ZARR_INDEX_GROUP}

        def _scan_actual_objs(self, table_group: Any) -> set[str]:
            objs: set[str] = set()
            for field in table_group:
                if field == ZARR_INDEX_GROUP:
                    continue
                objs.update(table_group[field].keys())
            return objs

        def _update_index_sets(
            self,
            table_group: Any,
            table: str,
            *,
            add_objs: set[str] | None = None,
            add_fields: set[str] | None = None,
            remove_objs: set[str] | None = None,
            remove_fields: set[str] | None = None,
        ) -> None:
            if self._readonly:
                return

            add_objs = add_objs or set()
            add_fields = add_fields or set()
            remove_objs = remove_objs or set()
            remove_fields = remove_fields or set()

            objs = self._load_index_or_scan(
                table_group, table, ZARR_INDEX_OBJS, self._scan_actual_objs
            )
            fields = self._load_index_or_scan(
                table_group, table, ZARR_INDEX_FIELDS, self._scan_actual_fields
            )

            if add_objs:
                objs.update(add_objs)
            if remove_objs:
                objs.difference_update(remove_objs)
            if add_fields:
                fields.update(add_fields)
            if remove_fields:
                fields.difference_update(remove_fields)

            self._set_index_cache(table, ZARR_INDEX_OBJS, frozenset(objs))
            self._set_index_cache(table, ZARR_INDEX_FIELDS, frozenset(fields))

            if self._lazy_index_sync:
                self._index_dirty[table] = True
            else:
                self._write_index_to_store(table_group, objs, fields)
                self._index_dirty[table] = False

        def _write_index_to_store(
            self, table_group: Any, objs: set[str], fields: set[str]
        ) -> None:
            """Write index sets to persistent store."""
            idx_group = table_group.require_group(ZARR_INDEX_GROUP)
            self._write_index_set(idx_group, ZARR_INDEX_OBJS, objs)
            self._write_index_set(idx_group, ZARR_INDEX_FIELDS, fields)

        def _sync_index(self, table: str) -> None:
            """Sync cached index to persistent store if dirty."""
            if not self._index_dirty.get(table) or self._readonly:
                return
            table_group = self._get_table_group(table)
            if table_group is None:
                return
            objs = self._get_index_cache(table, ZARR_INDEX_OBJS) or frozenset()
            fields = self._get_index_cache(table, ZARR_INDEX_FIELDS) or frozenset()
            self._write_index_to_store(table_group, set(objs), set(fields))
            self._index_dirty[table] = False

        def _obj_exists_anywhere(
            self,
            table_group: Any,
            table: str,
            enc_uid: str,
            skip_field: str | None,
            *,
            use_cache: bool = True,
        ) -> bool:
            """Check if object exists in any field."""
            if use_cache:
                cached_objs = self._get_index_cache(table, ZARR_INDEX_OBJS)
                if cached_objs is not None:
                    return enc_uid in cached_objs

            fields = self._load_index_or_scan(
                table_group, table, ZARR_INDEX_FIELDS, self._scan_actual_fields
            )
            for field in fields:
                if field == skip_field:
                    continue
                if field not in table_group:
                    continue
                if enc_uid in table_group[field]:
                    return True
            return False

        def _has_field(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> bool:
            table_group = self._get_table_group(table)
            if table_group is None:
                return False
            enc_field = self._encode_component(field_name)
            enc_uid = self._encode_component(obj_uid)
            try:
                if enc_field not in table_group:
                    return False
                return enc_uid in table_group[enc_field]
            except Exception as exc:
                msg = (
                    f"Failed to read Zarr field '{field_name}' for '{obj_uid}': "
                    f"{format_exception_message(exc)}"
                )
                raise OSError(msg) from exc

        def _get_field(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> Any:
            table_group = self._get_table_group(table)
            if table_group is None:
                msg = f"Field '{field_name}' of '{obj_uid}' not found"
                raise KeyError(msg)
            enc_field = self._encode_component(field_name)
            enc_uid = self._encode_component(obj_uid)
            try:
                field_missing = (
                    enc_field not in table_group
                    or enc_uid not in table_group[enc_field]
                )
                if field_missing:
                    msg = f"Field '{field_name}' of '{obj_uid}' not found"
                    raise KeyError(msg)  # noqa: TRY301

                arr = table_group[enc_field][enc_uid]
                kind = arr.attrs.get(STORAGE_ATTR_TYPE)

                if kind == _TYPE_NDARRAY:
                    return arr[()]
                return decode_value(bytes(arr[:]), kind)
            except KeyError:
                raise
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                msg = (
                    f"Data corruption in '{field_name}/{obj_uid}': "
                    f"{format_exception_message(exc)}"
                )
                raise ValueError(msg) from exc
            except Exception as exc:
                msg = (
                    f"Failed to read Zarr field '{field_name}' for '{obj_uid}': "
                    f"{format_exception_message(exc)}"
                )
                raise OSError(msg) from exc

        def _get_field_metadata(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> dict[str, Any] | None:
            table_group = self._get_table_group(table)
            if table_group is None:
                return None
            enc_field = self._encode_component(field_name)
            enc_uid = self._encode_component(obj_uid)
            try:
                field_missing = (
                    enc_field not in table_group
                    or enc_uid not in table_group[enc_field]
                )
                if field_missing:
                    return None
                arr = table_group[enc_field][enc_uid]
                raw_meta = arr.attrs.get(STORAGE_ATTR_META)
                if raw_meta is None:
                    return None
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                return json.loads(raw_meta)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                return None
            except Exception as exc:
                msg = (
                    f"Failed to read metadata for '{field_name}/{obj_uid}': "
                    f"{format_exception_message(exc)}"
                )
                raise OSError(msg) from exc

        def _list_fields(
            self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
        ) -> list[str]:
            table_group = self._get_table_group(table)
            if table_group is None:
                return []

            if obj_uid is None:
                fields = self._load_index_or_scan(
                    table_group, table, ZARR_INDEX_FIELDS, self._scan_actual_fields
                )
                return [self._decode_component(f) for f in sorted(fields)]

            enc_uid = self._encode_component(obj_uid)
            fields = self._load_index_or_scan(
                table_group, table, ZARR_INDEX_FIELDS, self._scan_actual_fields
            )
            out = []
            for field in fields:
                if field not in table_group:
                    continue
                # Direct membership check avoids creating intermediate set
                if enc_uid in table_group[field]:
                    out.append(self._decode_component(field))
            return sorted(out)

        def _list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
            table_group = self._get_table_group(table)
            if table_group is None:
                return []
            objs = self._load_index_or_scan(
                table_group, table, ZARR_INDEX_OBJS, self._scan_actual_objs
            )
            decoded = [self._decode_component(o) for o in objs]
            return sorted(decoded) if self._sort_list_objs else decoded

        def _list_objs_has_field(
            self, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> list[UID]:
            table_group = self._get_table_group(table)
            if table_group is None:
                return []
            enc_field = self._encode_component(field_name)
            try:
                if enc_field not in table_group:
                    return []
                return [self._decode_component(x) for x in table_group[enc_field]]
            except Exception as exc:
                msg = (
                    f"Failed to list objects for field '{field_name}': "
                    f"{format_exception_message(exc)}"
                )
                raise OSError(msg) from exc

        def _serialize_value(
            self, field_group: Any, enc_uid: str, value: Any
        ) -> tuple[Any, str]:
            """Serialize value to Zarr array. Returns (array, type_marker)."""
            if isinstance(value, np.ndarray):
                chunks = (
                    self._chunks
                    if isinstance(self._chunks, tuple)
                    else self._compute_optimal_chunks(value.shape, value.dtype)
                )
                compressors = self._get_compressors(value.nbytes)
                arr = field_group.create_array(
                    enc_uid,
                    data=value,
                    chunks=chunks,
                    compressors=compressors,
                    overwrite=True,
                )
                return arr, _TYPE_NDARRAY

            payload, tag = encode_value(value)
            data = np.frombuffer(payload, dtype=np.uint8)
            arr = field_group.create_array(
                enc_uid, data=data, chunks=(len(data),), overwrite=True
            )
            return arr, tag

        def _flush_set_field_batch(self) -> None:
            if not self._set_field_batch:
                return
            if self._readonly:
                raise OSError(_READONLY_ERROR)

            done: list[tuple[str, UID, str]] = []
            new_objs_by_table: dict[str, set[str]] = {}
            new_fields_by_table: dict[str, set[str]] = {}

            # Cache table_groups and field_group.keys() to reduce remote calls
            table_groups: dict[str, Any] = {}
            field_keys_cache: dict[tuple[str, str], set[str]] = {}

            with self._write_lock:
                for (tbl, obj_uid, field_name), value in self._set_field_batch.items():
                    # Cache table_group lookup
                    if tbl not in table_groups:
                        table_groups[tbl] = self._get_table_group(tbl)
                    table_group = table_groups[tbl]

                    enc_field = self._encode_component(field_name)
                    enc_uid = self._encode_component(obj_uid)
                    field_group = table_group.require_group(enc_field)

                    # Cache field keys lookup
                    cache_key = (tbl, enc_field)
                    if cache_key not in field_keys_cache:
                        field_keys_cache[cache_key] = set(field_group.keys())

                    if enc_uid in field_keys_cache[cache_key]:
                        del field_group[enc_uid]
                        field_keys_cache[cache_key].discard(enc_uid)

                    meta_json = json.dumps(
                        infer_value_metadata(value).to_jsonable(), ensure_ascii=False
                    )
                    arr, type_marker = self._serialize_value(
                        field_group, enc_uid, value
                    )
                    arr.attrs[STORAGE_ATTR_TYPE] = type_marker
                    arr.attrs[STORAGE_ATTR_META] = meta_json

                    new_objs_by_table.setdefault(tbl, set()).add(enc_uid)
                    new_fields_by_table.setdefault(tbl, set()).add(enc_field)
                    done.append((tbl, obj_uid, field_name))

                # Update indexes once per table
                for tbl, table_group in table_groups.items():
                    self._update_index_sets(
                        table_group,
                        tbl,
                        add_objs=new_objs_by_table.get(tbl),
                        add_fields=new_fields_by_table.get(tbl),
                    )

            self._clear_done_from_batch(done)

        def _clear_done_from_batch(self, done: list[tuple[str, UID, str]]) -> None:
            """Remove completed items from batch and update pending bytes."""
            for key in done:
                self._set_field_batch.pop(key, None)
                size = self._set_field_sizes.pop(key, 0)
                if size:
                    self._pending_set_bytes = max(0, self._pending_set_bytes - size)
            if not self._set_field_sizes:
                self._pending_set_bytes = 0

        def _group_erase_batch_by_table_field(
            self,
        ) -> dict[tuple[str, str], set[str]]:
            """Group erase_field_batch by (table, enc_field) -> set of enc UIDs."""
            grouped: dict[tuple[str, str], set[str]] = {}
            for tbl, obj_uid, field_name in self._erase_field_batch:
                enc_field = self._encode_component(field_name)
                enc_uid = self._encode_component(obj_uid)
                grouped.setdefault((tbl, enc_field), set()).add(enc_uid)
            return grouped

        def _delete_uids_from_field(
            self,
            table_group: Any,
            enc_field: str,
            enc_uids_to_remove: set[str],
        ) -> tuple[set[str], bool]:
            """Delete UIDs from a field group.

            Returns:
                Tuple of (actually deleted UIDs, whether field became empty).
            """
            if enc_field not in table_group:
                return set(), False

            field_group = table_group[enc_field]
            existing_uids = set(field_group.keys())
            uids_to_delete = enc_uids_to_remove & existing_uids

            for enc_uid in uids_to_delete:
                del field_group[enc_uid]

            if uids_to_delete:
                remaining = existing_uids - uids_to_delete
                if not remaining:
                    del table_group[enc_field]
                    return uids_to_delete, True

            return uids_to_delete, False

        def _find_orphaned_objs(
            self,
            table_group: Any,
            affected_objs: set[str],
        ) -> set[str]:
            """Find objects that no longer exist in any field."""
            cached_fields = [f for f in table_group if f != ZARR_INDEX_GROUP]
            # Build set of all existing UIDs across fields in one pass
            all_existing_uids: set[str] = set()
            for field in cached_fields:
                all_existing_uids.update(table_group[field].keys())

            return affected_objs - all_existing_uids

        def _update_indexes_for_removals(
            self,
            table_groups: dict[str, Any],
            removed_by_table: dict[str, _TableRemovalInfo],
        ) -> None:
            """Update indexes for all tables with removals."""
            for tbl, removal_info in removed_by_table.items():
                if not removal_info.removed_fields and not removal_info.removed_objs:
                    continue
                table_group = table_groups.get(tbl)
                if table_group is not None:
                    self._update_index_sets(
                        table_group,
                        tbl,
                        remove_fields=removal_info.removed_fields,
                        remove_objs=removal_info.removed_objs,
                    )

        def _flush_erase_field_batch(self) -> None:
            if not self._erase_field_batch:
                return
            if self._readonly:
                raise OSError(_READONLY_ERROR)

            by_table_field = self._group_erase_batch_by_table_field()
            removed_by_table: dict[str, _TableRemovalInfo] = {}
            affected_by_table: dict[str, set[str]] = {}
            table_groups: dict[str, Any] = {}

            with self._write_lock:
                self._process_field_deletions(
                    by_table_field, table_groups, removed_by_table, affected_by_table
                )
                self._process_orphaned_objects(
                    table_groups, affected_by_table, removed_by_table
                )
                self._update_indexes_for_removals(table_groups, removed_by_table)

            self._erase_field_batch.clear()

        def _process_field_deletions(
            self,
            by_table_field: dict[tuple[str, str], set[str]],
            table_groups: dict[str, Any],
            removed_by_table: dict[str, _TableRemovalInfo],
            affected_by_table: dict[str, set[str]],
        ) -> None:
            """Delete UIDs from fields and track affected state."""
            for (tbl, enc_field), enc_uids in by_table_field.items():
                if tbl not in table_groups:
                    table_groups[tbl] = self._get_table_group(tbl)
                table_group = table_groups[tbl]
                if table_group is None:
                    continue

                deleted, field_empty = self._delete_uids_from_field(
                    table_group, enc_field, enc_uids
                )
                if deleted:
                    affected_by_table.setdefault(tbl, set()).update(deleted)
                if field_empty:
                    self._ensure_removal_info(removed_by_table, tbl).removed_fields.add(
                        enc_field
                    )

        def _ensure_removal_info(
            self, removed_by_table: dict[str, _TableRemovalInfo], tbl: str
        ) -> _TableRemovalInfo:
            """Get or create removal info for a table."""
            if tbl not in removed_by_table:
                removed_by_table[tbl] = _TableRemovalInfo(set(), set())
            return removed_by_table[tbl]

        def _process_orphaned_objects(
            self,
            table_groups: dict[str, Any],
            affected_by_table: dict[str, set[str]],
            removed_by_table: dict[str, _TableRemovalInfo],
        ) -> None:
            """Find and mark orphaned objects for removal."""
            for tbl, affected_objs in affected_by_table.items():
                table_group = table_groups.get(tbl)
                if table_group is None:
                    continue
                orphaned = self._find_orphaned_objs(table_group, affected_objs)
                self._ensure_removal_info(removed_by_table, tbl).removed_objs.update(
                    orphaned
                )

        def _group_erase_obj_batch_by_table(self) -> dict[str, set[str]]:
            """Group erase_obj_batch by table -> set of encoded UIDs."""
            grouped: dict[str, set[str]] = {}
            for tbl, obj_uid in self._erase_obj_batch:
                enc_uid = self._encode_component(obj_uid)
                grouped.setdefault(tbl, set()).add(enc_uid)
            return grouped

        def _erase_objs_from_table(
            self,
            table_group: Any,
            enc_uids_to_remove: set[str],
        ) -> set[str]:
            """Erase objects from all fields in table.

            Returns:
                Set of fields that became empty.
            """
            cached_fields = [f for f in table_group if f != ZARR_INDEX_GROUP]
            empty_fields: set[str] = set()

            for field in cached_fields:
                field_group = table_group[field]
                existing_uids = set(field_group.keys())
                uids_to_delete = enc_uids_to_remove & existing_uids
                for enc_uid in uids_to_delete:
                    del field_group[enc_uid]
                # Track remaining to avoid extra remote call for emptiness check
                remaining = existing_uids - uids_to_delete
                if not remaining:
                    del table_group[field]
                    empty_fields.add(field)

            return empty_fields

        def _flush_erase_obj_batch(self) -> None:
            if not self._erase_obj_batch:
                return
            if self._readonly:
                raise OSError(_READONLY_ERROR)

            objs_by_table = self._group_erase_obj_batch_by_table()
            removed_by_table: dict[str, _TableRemovalInfo] = {}
            table_groups: dict[str, Any] = {}

            with self._write_lock:
                for tbl, enc_uids_to_remove in objs_by_table.items():
                    if tbl not in table_groups:
                        table_groups[tbl] = self._get_table_group(tbl)
                    table_group = table_groups[tbl]
                    if table_group is None:
                        continue

                    empty_fields = self._erase_objs_from_table(
                        table_group, enc_uids_to_remove
                    )
                    removed_by_table[tbl] = _TableRemovalInfo(
                        enc_uids_to_remove, empty_fields
                    )

                self._update_indexes_for_removals(table_groups, removed_by_table)

            self._erase_obj_batch.clear()

        def _erase_field_from_table(
            self,
            table_group: Any,
            enc_field: str,
        ) -> set[str]:
            """Erase field from table and return affected UIDs."""
            if enc_field not in table_group:
                return set()
            field_group = table_group[enc_field]
            affected_uids = set(field_group.keys())
            del table_group[enc_field]
            return affected_uids

        def _flush_erase_field_for_all_batch(self) -> None:
            if not self._erase_field_for_all_batch:
                return
            if self._readonly:
                raise OSError(_READONLY_ERROR)

            removed_by_table: dict[str, _TableRemovalInfo] = {}
            affected_by_table: dict[str, set[str]] = {}
            table_groups: dict[str, Any] = {}

            with self._write_lock:
                self._collect_fields_to_erase(
                    table_groups, removed_by_table, affected_by_table
                )
                self._process_orphaned_objects(
                    table_groups, affected_by_table, removed_by_table
                )
                self._update_indexes_for_removals(table_groups, removed_by_table)

            self._erase_field_for_all_batch.clear()

        def _collect_fields_to_erase(
            self,
            table_groups: dict[str, Any],
            removed_by_table: dict[str, _TableRemovalInfo],
            affected_by_table: dict[str, set[str]],
        ) -> None:
            """Delete fields and collect affected UIDs for erase_field_for_all."""
            for tbl, field_name in self._erase_field_for_all_batch:
                if tbl not in table_groups:
                    table_groups[tbl] = self._get_table_group(tbl)
                table_group = table_groups[tbl]
                if table_group is None:
                    continue

                enc_field = self._encode_component(field_name)
                affected = self._erase_field_from_table(table_group, enc_field)
                if affected:
                    affected_by_table.setdefault(tbl, set()).update(affected)
                    self._ensure_removal_info(removed_by_table, tbl).removed_fields.add(
                        enc_field
                    )

        def _clear(self, *, table: str | None = None) -> None:
            if self._readonly:
                raise OSError(_READONLY_ERROR)

            with self._write_lock:
                root = self._get_root()
                if table is None:
                    self._clear_all_tables(root)
                else:
                    self._clear_single_table(root, table)

        def _clear_all_tables(self, root: Any) -> None:
            """Clear all tables from root."""
            for key in tuple(root.keys()):
                del root[key]
            self._index_cache_objs.clear()
            self._index_cache_fields.clear()
            self._index_dirty.clear()

        def _clear_single_table(self, root: Any, table: str) -> None:
            """Clear and recreate a single table."""
            if table in root:
                del root[table]
            table_group = root.require_group(table)
            idx_group = table_group.require_group(ZARR_INDEX_GROUP)
            self._write_index_set(idx_group, ZARR_INDEX_OBJS, set())
            self._write_index_set(idx_group, ZARR_INDEX_FIELDS, set())
            self._set_index_cache(table, ZARR_INDEX_OBJS, frozenset())
            self._set_index_cache(table, ZARR_INDEX_FIELDS, frozenset())
            self._index_dirty[table] = False

        def __del__(self) -> None:
            """Clean up resources on deletion."""
            with contextlib.suppress(Exception):
                self.close()
