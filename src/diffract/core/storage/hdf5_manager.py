"""HDF5-backed storage manager optimized for high-throughput reads.

Design notes:
- Writes are serialized via a single write handle under a lock.
- Reads use per-thread read-only file handles (thread-local) to avoid
  Python-level locks.
- Object existence is tracked by a dataset-based index (resizable 1D string dataset).
- Data is organized by tables for logical separation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

import diffract.core.utils.imports as import_utils
from diffract.core.constants import (
    HDF5_INDEX_DATASET,
    HDF5_INDEX_GROUP,
    HDF5_INDEX_TOMBSTONE,
    STORAGE_ATTR_META,
    STORAGE_ATTR_TYPE,
)
from diffract.core.utils.exceptions import format_exception_message

from .base_manager import BaseStorageManager
from .interface import DEFAULT_TABLE, UID
from .metadata import infer_value_metadata

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


if not import_utils.is_available("h5py"):
    logger.debug("h5py not available, disabling HDF5 storage manager")

    class HDF5StorageManager(BaseStorageManager):
        """Stub implementation when h5py is not available."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError because the optional dependency is missing."""
            msg = "h5py package not available"
            raise ImportError(msg)

else:
    h5py = import_utils.require("h5py")

    class HDF5StorageManager(BaseStorageManager):
        """Persistent storage manager using an HDF5 file.

        Supports storing arbitrary Python values (pickled fallback) and NumPy arrays.
        Intended for workloads with frequent multi-threaded reads and serialized writes.

        Example:
            >>> storage = HDF5StorageManager("data.h5")
            >>> with storage:
            ...     storage.set_field("param_001", "weights", np.random.randn(100, 100))
            ...     weights = storage.get_field("param_001", "weights")
        """

        def __init__(
            self,
            path: str,
            *,
            root: str = "root",
            compression: str | None = None,
            compression_opts: int | None = None,
            shuffle: bool = False,
            fletcher: bool = False,
            swmr: bool = True,
            verify_index: bool = False,
            keep_file_open: bool = True,
            readonly: bool = False,
            refresh_on_read: bool = False,
            chunks: bool | tuple[int, ...] | None = None,
            index_cache: bool = True,
            index_group: str = HDF5_INDEX_GROUP,
            index_all_objs: str = HDF5_INDEX_DATASET,
            index_tombstone: str = HDF5_INDEX_TOMBSTONE,
            index_encoding: str = "utf-8",
            index_chunk_len: int = 4096,
            sort_list_objs: bool = True,
            libver: str = "latest",
            **kwargs: Any,
        ) -> None:
            """Create a storage manager.

            Args:
                path: Path to the HDF5 file.
                root: Top-level group name used for data storage.
                compression: Compression algorithm (e.g., "gzip", "lzf").
                compression_opts: Compression options (e.g., compression level).
                shuffle: Enable shuffle filter for compression.
                fletcher: Enable Fletcher32 checksum.
                swmr: Enable SWMR mode (Single Writer Multiple Reader) when supported.
                keep_file_open: Keep file handles open between operations.
                readonly: If True, reject any write operation.
                refresh_on_read: If True and SWMR is enabled, call `file.refresh()`
                    on reads.
                chunks: Chunking policy for ndarray datasets; use contiguous layout
                    when possible.
                verify_index: If True, verify index consistency in `list_objs()`
                    and rebuild if needed.
                index_cache: Enable in-memory index cache.
                index_group: Group name for object index.
                index_all_objs: Dataset name for "all objects" index.
                index_tombstone: Dataset name for tombstone markers.
                index_encoding: Encoding for index string datasets.
                index_chunk_len: Chunk size for index datasets.
                sort_list_objs: Sort object IDs in list_objs() output.
                libver: HDF5 library format bounds (e.g. "latest").
                **kwargs: Additional keyword arguments for BaseStorageManager.
            """
            super().__init__(**kwargs)

            self._path = path
            self._root = root
            self._compression = compression
            self._compression_opts = compression_opts
            self._shuffle = shuffle
            self._fletcher = fletcher
            self._swmr = swmr
            self._verify_index = verify_index
            self._keep_file_open = keep_file_open
            self._readonly = readonly
            self._refresh_on_read = refresh_on_read
            self._chunks = chunks
            self._index_cache_enabled = index_cache
            self._index_group = index_group
            self._index_all_objs = index_all_objs
            self._index_tombstone = index_tombstone
            self._index_encoding = index_encoding
            if index_chunk_len <= 0:
                msg = "index_chunk_len must be > 0"
                raise ValueError(msg)
            self._index_chunk_len = index_chunk_len
            self._sort_list_objs = sort_list_objs
            self._libver = libver

            # Per-table index caches (only created when index_cache=True)
            self._index_caches: dict[str, set[str]] = {}
            self._index_cache_enabled = index_cache

            # Writes serialized; reads use per-thread read-only file handles.
            self._write_lock = threading.Lock()
            self._write_handle: h5py.File | None = None
            self._read_local = threading.local()

            dirpath = Path(self._path).parent
            if not dirpath.exists():
                dirpath.mkdir(parents=True)

            # Ensure schema exists unless in readonly mode.
            if not self._readonly:
                with h5py.File(self._path, "a", libver=self._libver) as f:
                    f.require_group(self._root)
                    if self._swmr:
                        with contextlib.suppress(Exception):
                            if not f.swmr_mode:
                                f.swmr_mode = True

            # Track read handles across threads so we can close them before opening a
            # write handle. HDF5 disallows opening the same file read-only and
            # read-write concurrently within a single process.
            self._read_handles_lock = threading.Lock()
            self._read_handles_by_thread: dict[int, h5py.File] = {}

        def _register_read_handle(self, handle: h5py.File) -> None:
            tid = threading.get_ident()
            with self._read_handles_lock:
                self._read_handles_by_thread[tid] = handle

        def _unregister_read_handle(self, handle: h5py.File | None) -> None:
            if handle is None:
                return
            tid = threading.get_ident()
            with self._read_handles_lock:
                current = self._read_handles_by_thread.get(tid)
                if current is handle:
                    self._read_handles_by_thread.pop(tid, None)

        def _close_all_read_handles(self) -> int:
            """Close all known thread-local read handles (best-effort)."""
            with self._read_handles_lock:
                items = list(self._read_handles_by_thread.items())
                self._read_handles_by_thread.clear()

            closed = 0
            for _, handle in items:
                try:
                    if handle is not None and handle.id.valid:
                        handle.close()
                        closed += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to close read handle %s: %s",
                        handle,
                        format_exception_message(exc),
                    )
                    continue
            return closed

        def _close_current_thread_read_handle(self) -> None:
            """Close the current thread's read handle if valid."""
            read_handle = getattr(self._read_local, "handle", None)
            if read_handle is not None and read_handle.id.valid:
                with contextlib.suppress(Exception):
                    read_handle.close()
                self._read_local.handle = None
                self._unregister_read_handle(read_handle)

        def _need_reopen_write(self, reopen: bool) -> bool:
            """Check if write handle needs to be reopened."""
            if self._write_handle is None:
                return True
            if not self._write_handle.id.valid:
                return True
            if reopen:
                return True
            return self._write_handle.mode != "r+"

        def _open_write_handle(self) -> h5py.File:
            """Open or reopen write handle, closing existing handles first."""
            if self._write_handle is not None:
                with contextlib.suppress(Exception):
                    self._write_handle.close()

            self._close_current_thread_read_handle()
            self._close_all_read_handles()

            self._write_handle = h5py.File(self._path, "a", libver=self._libver)
            if self._swmr and not self._write_handle.swmr_mode:
                with contextlib.suppress(Exception):
                    self._write_handle.swmr_mode = True
            return self._write_handle

        def _need_reopen_read(self, handle: h5py.File | None, reopen: bool) -> bool:
            """Check if read handle needs to be reopened."""
            if handle is None or not handle.id.valid:
                return True
            if reopen:
                return True
            mode = None
            with contextlib.suppress(Exception):
                mode = str(handle.mode)
            return mode is not None and not mode.startswith("r")

        def _open_read_handle(self, handle: h5py.File | None) -> h5py.File:
            """Open or reopen read handle."""
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
                self._unregister_read_handle(handle)
            handle = h5py.File(self._path, "r", libver=self._libver, swmr=self._swmr)
            self._read_local.handle = handle
            self._register_read_handle(handle)
            return handle

        def _get_or_open_file(self, *, write: bool, reopen: bool) -> h5py.File:
            """Return an HDF5 file handle (write-shared or read thread-local)."""
            if write and self._readonly:
                msg = "HDF5StorageManager is readonly; write operation is not allowed"
                raise OSError(msg)

            if write:
                if self._need_reopen_write(reopen):
                    return self._open_write_handle()
                return self._write_handle  # type: ignore[return-value]

            # Read-only: use thread-local handle to avoid python-level locks on read.
            handle = getattr(self._read_local, "handle", None)
            if self._need_reopen_read(handle, reopen):
                return self._open_read_handle(handle)
            return handle  # type: ignore[return-value]

        @contextlib.contextmanager
        def _open(self, *, write: bool = False) -> Generator[h5py.File, None, None]:
            """Context manager that yields an opened HDF5 file handle."""
            if write:
                with self._write_lock:
                    file = self._get_or_open_file(
                        write=True, reopen=(not self._keep_file_open)
                    )
                    try:
                        yield file
                        file.flush()
                    finally:
                        if not self._keep_file_open:
                            with contextlib.suppress(Exception):
                                file.close()
                            self._write_handle = None
                return

            reopen = not self._keep_file_open
            file = self._get_or_open_file(write=False, reopen=reopen)
            if self._swmr and self._refresh_on_read:
                with contextlib.suppress(Exception):
                    file.refresh()
            try:
                yield file
            finally:
                if not self._keep_file_open:
                    with contextlib.suppress(Exception):
                        file.close()
                    if getattr(self._read_local, "handle", None) is file:
                        self._read_local.handle = None

        def _table_path(self, table: str) -> str:
            """Build path to a table group."""
            return f"{self._root}/{table}"

        def _h5path(self, table: str, field: str, uid: UID) -> str:
            """Build an absolute HDF5 path for a stored field dataset."""
            return f"{self._root}/{table}/{field}/{uid}"

        def _index_group_path(self, table: str) -> str:
            """Path to the index group under table."""
            return f"{self._root}/{table}/{self._index_group}"

        def _index_all_objs_path(self, table: str) -> str:
            """Path to the dataset-based all-objs index for a table."""
            return f"{self._root}/{table}/{self._index_group}/{self._index_all_objs}"

        def _decode_index_value(self, raw_value: bytes | str) -> str:
            """Decode a raw index value to string."""
            if isinstance(raw_value, bytes):
                return raw_value.decode(self._index_encoding)
            return str(raw_value)

        def _ensure_table(self, f: h5py.File, table: str) -> h5py.Group:
            """Ensure table group and its index exist."""
            table_group = f.require_group(self._table_path(table))
            idx_root = table_group.require_group(self._index_group)
            self._ensure_index(idx_root)
            return table_group

        def _ensure_index(self, idx_root: h5py.Group) -> None:
            """Ensure the all-objs index exists."""
            if self._index_all_objs not in idx_root:
                dt = h5py.string_dtype(encoding=self._index_encoding)
                idx_root.create_dataset(
                    self._index_all_objs,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt,
                    chunks=(self._index_chunk_len,),
                )
                return

            obj = idx_root[self._index_all_objs]
            if isinstance(obj, h5py.Dataset):
                return

            msg = "Unsupported index format for HDF5 all-objs index."
            raise TypeError(msg)

        def _index_dataset(self, f: h5py.File, table: str) -> h5py.Dataset | None:
            """Return the dataset-based all-objs index dataset for a table."""
            idx_path = self._index_all_objs_path(table)
            if idx_path not in f:
                return None
            obj = f[idx_path]
            return obj if isinstance(obj, h5py.Dataset) else None

        def _get_index_cache(self, table: str) -> set[str] | None:
            """Get or create index cache for a table."""
            if not self._index_cache_enabled:
                return None
            if table not in self._index_caches:
                self._index_caches[table] = set()
            return self._index_caches[table]

        def _load_index_cache(self, f: h5py.File, table: str) -> None:
            """Load the all-objs index into memory for a table."""
            cache = self._get_index_cache(table)
            if cache is None:
                return
            if cache:
                return
            index_dataset = self._index_dataset(f, table)
            if index_dataset is not None:
                vals = [
                    self._decode_index_value(raw) for raw in index_dataset[:]
                ]
                cache.update(x for x in vals if x and x != self._index_tombstone)

        def _index_add_obj(self, f: h5py.File, table: str, uid: UID) -> None:
            """Ensure `uid` is present in the index dataset."""
            index_dataset = self._index_dataset(f, table)
            if index_dataset is None:
                msg = "Missing dataset-based all-objs index."
                raise KeyError(msg)

            cache = self._get_index_cache(table)
            if cache is not None:
                self._load_index_cache(f, table)
                if uid in cache:
                    return
                cache.add(uid)
            else:
                # Use set for O(1) lookup instead of O(n) any() iteration
                existing = {
                    self._decode_index_value(raw) for raw in index_dataset[:]
                }
                if uid in existing:
                    return

            current_len = index_dataset.shape[0]
            index_dataset.resize((current_len + 1,))
            index_dataset[current_len] = uid

        def _index_remove_obj(self, f: h5py.File, table: str, uid: UID) -> None:
            """Mark `uid` as removed in the index dataset (tombstone).
            
            If the index dataset doesn't exist, this is a no-op (object was never
            stored in this backend, common in hybrid storage scenarios).
            """
            index_dataset = self._index_dataset(f, table)
            if index_dataset is None:
                return

            vals = index_dataset[:]
            for index, raw_value in enumerate(vals):
                if self._decode_index_value(raw_value) == uid:
                    index_dataset[index] = self._index_tombstone
                    break
            cache = self._get_index_cache(table)
            if cache is not None:
                cache.discard(uid)

        def _scan_actual_objs(self, f: h5py.File, table: str) -> set[str]:
            """Scan and return UIDs that exist in any field for a table."""
            objs: set[str] = set()
            table_path = self._table_path(table)
            if table_path not in f:
                return objs

            table_group = f[table_path]
            for field in table_group:
                if field == self._index_group:
                    continue
                group = f[f"{table_path}/{field}"]
                for uid in group:
                    objs.add(uid)

            return objs

        def _rebuild_index(self, f: h5py.File, table: str) -> None:
            """Rebuild the dataset-based all-objs index from actual file contents."""
            actual = self._scan_actual_objs(f, table)
            idx_root = f[self._index_group_path(table)]

            self._ensure_index(idx_root)
            index_dataset = self._index_dataset(f, table)
            if index_dataset is None:
                return
            uids = sorted(actual)

            index_dataset.resize((len(uids),))
            if uids:
                index_dataset[:] = uids
            cache = self._get_index_cache(table)
            if cache is not None:
                cache.clear()
                cache.update(uids)

        def _has_field(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> bool:
            """Return True if the given object has the specified field."""
            with self._open(write=False) as f:
                return self._h5path(table, field_name, obj_uid) in f

        def _get_field(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> Any:
            """Load a stored field value."""
            try:
                with self._open(write=False) as f:
                    path = self._h5path(table, field_name, obj_uid)

                    if path not in f:
                        msg = f"Field '{field_name}' of '{obj_uid}' not found"
                        raise KeyError(msg)  # noqa: TRY301

                    dataset = f[path]
                    kind = dataset.attrs.get(STORAGE_ATTR_TYPE, "ndarray")
                    if kind == "ndarray":
                        data = dataset[()]
                        if not data.shape:
                            data = data.item()
                        return data

                    if kind == "bytes":
                        raw = dataset[()]
                        if isinstance(raw, np.ndarray):
                            data_bytes = raw.tobytes()
                            return pickle.loads(data_bytes)  # noqa: S301
                        if isinstance(raw, (bytes, bytearray)):
                            return raw
                        return bytes(raw)

                    msg = f"Unknown stored type: {kind}"
                    raise TypeError(msg)  # noqa: TRY301

            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                msg = f"Data corruption in '{field_name}/{obj_uid}': {exc}"
                raise ValueError(msg) from exc

            except Exception:
                logger.exception(
                    "Failed to read field '%s' for '%s'", field_name, obj_uid
                )
                raise

        def _get_field_metadata(
            self, obj_uid: UID, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> dict[str, Any] | None:
            """Return stored metadata (dtype/shape/kind) if present."""
            with self._open(write=False) as f:
                path = self._h5path(table, field_name, obj_uid)
                if path not in f:
                    return None

                dataset = f[path]
                raw_meta = dataset.attrs.get(STORAGE_ATTR_META)
                if raw_meta is None:
                    return None
                try:
                    if isinstance(raw_meta, bytes):
                        raw_meta = raw_meta.decode("utf-8")
                    return json.loads(raw_meta)
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    return None

        def _list_fields(
            self, obj_uid: UID = None, *, table: str = DEFAULT_TABLE
        ) -> list[str]:
            """List stored field names."""
            with self._open(write=False) as f:
                table_path = self._table_path(table)
                if table_path not in f:
                    return []
                table_group = f[table_path]

                out: list[str] = []

                if obj_uid is None:
                    return [
                        fld for fld in table_group if fld != self._index_group
                    ]
                for field in table_group:
                    if field == self._index_group:
                        continue

                    if f"{table_path}/{field}/{obj_uid}" in f:
                        out.append(field)

                return out

        def _parse_index_dataset(self, index_dataset: h5py.Dataset) -> set[str]:
            """Parse index dataset into a set of valid UIDs."""
            vals = [self._decode_index_value(raw) for raw in index_dataset[:]]
            return {x for x in vals if x and x != self._index_tombstone}

        def _format_uid_list(self, uids: set[str]) -> list[str]:
            """Format UID set as list, optionally sorted."""
            return sorted(uids) if self._sort_list_objs else list(uids)

        def _list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
            """List known object UIDs."""
            with self._open(write=False) as f:
                table_path = self._table_path(table)
                if table_path not in f:
                    return []

                index_dataset = self._index_dataset(f, table)
                if index_dataset is None and self._readonly:
                    actual = self._scan_actual_objs(f, table)
                    return self._format_uid_list(actual)

                if index_dataset is not None:
                    indexed = self._parse_index_dataset(index_dataset)
                    if not self._verify_index:
                        return self._format_uid_list(indexed)

                    actual = self._scan_actual_objs(f, table)
                    if indexed == actual:
                        return self._format_uid_list(indexed)

                    logger.warning(
                        "HDF5 index out of sync; rebuilding or returning actual."
                    )
                    if self._readonly:
                        return self._format_uid_list(actual)

            # Rebuild index
            with self._open(write=True, reopen=True) as f:
                self._ensure_table(f, table)
                self._rebuild_index(f, table)

            with self._open(write=False, reopen=True) as f:
                actual = self._scan_actual_objs(f, table)
                return self._format_uid_list(actual)

        def _list_objs_has_field(
            self, field_name: str, *, table: str = DEFAULT_TABLE
        ) -> list[UID]:
            """List object UIDs that have the given field."""
            with self._open(write=False) as f:
                field_path = f"{self._table_path(table)}/{field_name}"
                if field_path not in f:
                    return []
                return list(f[field_path].keys())

        def _get_chunk_config(self) -> bool | tuple[int, ...] | None:
            """Determine chunk configuration for dataset creation."""
            if self._chunks is None and self._compression is None:
                return None
            return self._chunks if self._chunks is not None else True

        def _write_ndarray_dataset(
            self, group: h5py.Group, obj_uid: UID, value: np.ndarray, meta_json: str
        ) -> None:
            """Write a numpy array as an HDF5 dataset."""
            chunks = self._get_chunk_config()
            dataset = group.create_dataset(
                obj_uid,
                data=value,
                compression=self._compression,
                compression_opts=self._compression_opts,
                shuffle=self._shuffle,
                fletcher32=self._fletcher,
                chunks=chunks,
            )
            dataset.attrs[STORAGE_ATTR_TYPE] = "ndarray"
            dataset.attrs[STORAGE_ATTR_META] = meta_json

        def _write_generic_dataset(
            self, group: h5py.Group, obj_uid: UID, value: Any, meta_json: str
        ) -> None:
            """Write a non-ndarray value as an HDF5 dataset."""
            try:
                dataset = group.create_dataset(obj_uid, data=np.array(value))
                dataset.attrs[STORAGE_ATTR_TYPE] = "ndarray"
            except (TypeError, ValueError):
                dataset = group.create_dataset(
                    obj_uid,
                    data=np.frombuffer(
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                        dtype=np.uint8,
                    ),
                    chunks=True,
                )
                dataset.attrs[STORAGE_ATTR_TYPE] = "bytes"
            dataset.attrs[STORAGE_ATTR_META] = meta_json

        def _flush_set_field_batch(self) -> None:
            """Flush queued `set_field` operations."""
            done: list[tuple[str, UID, str]] = []

            try:
                with self._open(write=True) as f:
                    for key, value in self._set_field_batch.items():
                        tbl, obj_uid, field_name = key
                        self._ensure_table(f, tbl)
                        table_path = self._table_path(tbl)
                        group = f.require_group(f"{table_path}/{field_name}")
                        dest = self._h5path(tbl, field_name, obj_uid)
                        meta = infer_value_metadata(value).to_jsonable()
                        meta_json = json.dumps(meta, ensure_ascii=False)

                        if dest in f:
                            del f[dest]

                        if isinstance(value, np.ndarray):
                            self._write_ndarray_dataset(
                                group, obj_uid, value, meta_json
                            )
                        else:
                            self._write_generic_dataset(
                                group, obj_uid, value, meta_json
                            )

                        self._index_add_obj(f, tbl, obj_uid)
                        done.append((tbl, obj_uid, field_name))

            except Exception:
                logger.exception("Failed to flush set_field batch")
                raise

            finally:
                for key in done:
                    del self._set_field_batch[key]
                    size = self._set_field_sizes.pop(key, 0)
                    if size:
                        self._pending_set_bytes = max(0, self._pending_set_bytes - size)
                if not self._set_field_sizes:
                    self._pending_set_bytes = 0

        def _obj_has_any_field(
            self, f: h5py.File, table_path: str, obj_uid: UID
        ) -> bool:
            """Check if object has any field in the table."""
            if table_path not in f:
                return False
            table_group = f[table_path]
            for field in table_group:
                if field == self._index_group:
                    continue
                if f"{table_path}/{field}/{obj_uid}" in f:
                    return True
            return False

        def _flush_erase_field_batch(self) -> None:
            """Flush queued `erase_field` operations."""
            done: list[tuple[str, UID, str]] = []

            try:
                with self._open(write=True) as f:
                    for tbl, obj_uid, field_name in self._erase_field_batch:
                        path = self._h5path(tbl, field_name, obj_uid)
                        if path in f:
                            del f[path]

                        table_path = self._table_path(tbl)
                        if not self._obj_has_any_field(f, table_path, obj_uid):
                            self._index_remove_obj(f, tbl, obj_uid)

                        done.append((tbl, obj_uid, field_name))

            except Exception:
                logger.exception("Failed to flush erase_field batch")
                raise

            finally:
                for key in done:
                    self._erase_field_batch.discard(key)

        def _flush_erase_obj_batch(self) -> None:
            """Flush queued `erase_obj` operations."""
            done: list[tuple[str, UID]] = []

            try:
                with self._open(write=True) as f:
                    for tbl, obj_uid in self._erase_obj_batch:
                        table_path = self._table_path(tbl)
                        if table_path not in f:
                            done.append((tbl, obj_uid))
                            continue

                        table_group = f[table_path]
                        for field in tuple(table_group.keys()):
                            if field == self._index_group:
                                continue

                            path = self._h5path(tbl, field, obj_uid)
                            if path in f:
                                del f[path]

                        self._index_remove_obj(f, tbl, obj_uid)
                        done.append((tbl, obj_uid))

            except Exception:
                logger.exception("Failed to flush erase_obj batch")
                raise

            finally:
                for key in done:
                    self._erase_obj_batch.discard(key)

        def _sync_index_after_field_erase(
            self, f: h5py.File, uids_by_table: dict[str, set[UID]]
        ) -> None:
            """Remove UIDs from index if they have no remaining fields."""
            for tbl, uids in uids_by_table.items():
                if not uids:
                    continue
                actual = self._scan_actual_objs(f, tbl)
                for uid in uids:
                    if uid not in actual:
                        self._index_remove_obj(f, tbl, uid)

        def _flush_erase_field_for_all_batch(self) -> None:
            """Flush queued `erase_field_for_all` operations."""
            done: list[tuple[str, str]] = []
            uids_by_table: dict[str, set[UID]] = {}

            try:
                with self._open(write=True) as f:
                    for tbl, field_name in self._erase_field_for_all_batch:
                        field_path = f"{self._table_path(tbl)}/{field_name}"

                        if field_path not in f:
                            continue

                        if tbl not in uids_by_table:
                            uids_by_table[tbl] = set()
                        uids_by_table[tbl].update(f[field_path].keys())

                        del f[field_path]
                        done.append((tbl, field_name))

                    self._sync_index_after_field_erase(f, uids_by_table)

            except Exception:
                logger.exception("Failed to flush erase_field_for_all batch")
                raise

            finally:
                for key in done:
                    self._erase_field_for_all_batch.discard(key)

        def _clear(self, *, table: str | None = None) -> None:
            """Remove data and recreate structure."""
            with self._open(write=True) as f:
                if table is None:
                    # Clear all data
                    if self._root in f:
                        del f[self._root]
                        f.create_group(self._root)
                        self._index_caches.clear()
                else:
                    # Clear specific table
                    table_path = self._table_path(table)
                    if table_path in f:
                        del f[table_path]
                    self._ensure_table(f, table)
                    if table in self._index_caches:
                        self._index_caches[table].clear()

        def close(self) -> None:
            """Close open file handles owned by this manager (best-effort)."""
            handle = getattr(getattr(self, "_read_local", None), "handle", None)
            if handle is not None and handle.id.valid:
                with contextlib.suppress(Exception):
                    handle.close()
                self._read_local.handle = None
                self._unregister_read_handle(handle)

            self._close_all_read_handles()

            write_handle = getattr(self, "_write_handle", None)
            if write_handle is not None and write_handle.id.valid:
                with contextlib.suppress(Exception):
                    write_handle.close()
                self._write_handle = None
            logger.debug("HDF5 file descriptor(s) closed")

        def __del__(self) -> None:
            """Best-effort cleanup for GC/abnormal shutdown scenarios."""
            with contextlib.suppress(Exception):
                self.close()
