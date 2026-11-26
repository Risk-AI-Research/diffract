"""Inspect and rekey legacy persistent stores during an offline upgrade."""

from __future__ import annotations

import ast
import hashlib
import json
import os

# Match encoded scalars against locally generated reference bytes.
import pickle  # nosec B403
import pickletools  # nosec B403
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .catalog import OBJECT_KIND_BY_TABLE
from .hdf5_manager import HDF5StorageManager
from .keys import legacy_sqlite_key, row_address, value_key
from .serialization import encode_value
from .sqlite_manager import (
    SQLiteStorageManager,
    has_catalog_rows,
    iter_storage_rows,
    legacy_row_keys,
)
from .zarr_manager import ZarrStorageManager

_CLASSES = {
    "sqlite": SQLiteStorageManager,
    "hdf5": HDF5StorageManager,
    "zarr": ZarrStorageManager,
}
_MAX_SCALAR_BYTES = 256
_TABLES = ("parameters", "aggregates")


@dataclass(frozen=True)
class StoreSpec:
    """Location and construction options of one physical store in a profile."""

    name: str
    backend: str
    location: str
    options: dict

    @property
    def remote(self) -> bool:
        """Whether object-store values retain their existing paths."""
        return self.backend == "zarr" and "://" in self.location

    @property
    def keeps_legacy_keys(self) -> bool:
        """Keep container-backed and remote values at their original addresses."""
        return self.backend in {"sqlite", "hdf5"} or self.remote

    def open(self, *, readonly: bool) -> Any:
        """Open the configured existing physical backend."""
        if not self.remote and not Path(self.location).exists():
            raise FileNotFoundError(self.location)
        return _CLASSES[self.backend](
            self.location, **(self.options | {"readonly": readonly})
        )

    def old_key(self, table: str, field: str, uid: str) -> str:
        """Return the path used by this backend for a legacy field."""
        if self.backend == "sqlite":
            return legacy_sqlite_key(table, field, uid)
        if self.backend == "zarr":
            return "/".join(quote(s, safe="") for s in (table, field, uid))
        return f"{table}/{field}/{uid}"


@dataclass
class LegacyValue:
    """One retained logical field and its preflight verification signature."""

    table: str
    field: str
    old_uid: str
    backend: str
    old_key: str
    new_key: str
    new_uid: str
    size: int | None
    signature: str
    legacy_scalar: bool = False


def _scalar_shape(payload: bytes) -> tuple[tuple, bytes]:
    """Inspect a bounded numeric pickle without loading objects or invoking opcodes."""
    if len(payload) > _MAX_SCALAR_BYTES:
        raise ValueError("Unsupported legacy scalar payload")
    pattern, scalar = [], b""
    end = -1
    for op, operand, pos in pickletools.genops(payload):
        arg = operand
        if op.name == "FRAME":
            arg = None
        elif op.name in {"SHORT_BINBYTES", "BINBYTES"}:
            scalar, arg = arg, None
        elif op.name == "SHORT_BINUNICODE" and arg == "numpy._core.multiarray":
            arg = "numpy.core.multiarray"
        pattern.append((op.name, arg))
        end = pos
    if end != len(payload) - 1:
        raise ValueError("Trailing data in legacy scalar")
    return tuple(pattern), scalar


@lru_cache(maxsize=1)
def _numeric_patterns() -> dict[tuple, np.dtype]:
    patterns = {}
    for code in ("?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8"):
        dtype = np.dtype(code)
        for protocol in (4, 5):
            # Only these locally generated numeric scalars define accepted forms.
            shape, _ = _scalar_shape(pickle.dumps(dtype.type(0), protocol=protocol))
            patterns[shape] = dtype
    return patterns


def converted_numeric_scalar(payload: bytes) -> bytes:
    """Convert an exact NumPy numeric-scalar encoding to JSON without executing it."""
    shape, data = _scalar_shape(payload)
    dtype = _numeric_patterns().get(shape)
    if dtype is None or len(data) != dtype.itemsize:
        raise ValueError(
            "Unsupported legacy pickle. Only primitive NumPy numeric scalars "
            "can be migrated."
        )
    (value,) = np.frombuffer(data, dtype=dtype)
    return encode_value(value)[0]


def _nbytes(metadata: dict) -> int | None:
    shape = metadata.get("shape")
    if shape is None:
        return None
    descriptor = metadata["dtype"]
    try:
        dtype = np.dtype(descriptor)
    except TypeError:
        # The original metadata uses str(dtype), including NumPy record descriptors.
        dtype = np.dtype(ast.literal_eval(descriptor))
    return prod(shape) * dtype.itemsize


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sqlite_rows(spec: StoreSpec) -> Iterator[tuple]:
    path = Path(spec.location)
    for table, field, uid, tag, data, external, meta in iter_storage_rows(
        path, _TABLES
    ):
        conversion = converted_numeric_scalar(data) if tag == "pickle" else None
        if tag not in {"pickle", "ndarray", "json", "bytes"}:
            raise ValueError(
                f"Unsupported SQLite serialization tag {tag!r} for {uid}/{field}"
            )
        if external:
            digest = _file_digest(path.parent / external)
        else:
            digest = hashlib.sha256(
                conversion if conversion is not None else data
            ).hexdigest()
        signature = ("json" if conversion is not None else tag) + ":" + digest
        sentinel = tag == "json" and data == b'"__HEAVY__"'
        yield (
            table,
            field,
            uid,
            _nbytes(json.loads(meta or "{}")),
            signature,
            conversion is not None,
            sentinel,
        )


def _node_info(spec: StoreSpec, store: Any, key: str) -> tuple[int | None, str]:
    metadata = store.blob_metadata(key) or {}
    if spec.backend == "hdf5":
        signature = store.blob_identity(key)
    elif spec.remote:
        # Remote re-key retains the very same object path.
        signature = (spec.location, key, metadata, store.blob_identity(key))
    else:
        path = Path(spec.location) / spec.options.get("root", "root") / key
        signature = (path.stat().st_ino, metadata, store.blob_identity(key))
    return _nbytes(metadata), json.dumps(signature, sort_keys=True)


def inventory(
    specs: list[StoreSpec],
    retained: dict[tuple[str, str], str],
    owners: dict[tuple[str, str], str],
) -> list[LegacyValue]:
    """Inspect the authoritative legacy keys and check every routed value."""
    primary, *heavy = specs
    values = []
    managers = {}
    sqlite_heavy = (
        {
            (table, field, uid): (size, sig, legacy_scalar)
            for table, field, uid, size, sig, legacy_scalar, _ in _sqlite_rows(heavy[0])
        }
        if heavy and heavy[0].backend == "sqlite"
        else {}
    )

    def manager(spec: StoreSpec) -> Any:
        if spec.name not in managers:
            managers[spec.name] = spec.open(readonly=True)
        return managers[spec.name]

    def add(
        table: str,
        field: str,
        uid: str,
        spec: StoreSpec,
        size: int | None,
        signature: str,
        legacy_scalar: bool = False,
    ) -> None:
        if (OBJECT_KIND_BY_TABLE[table], uid) not in owners:
            raise ValueError(f"Value references an unknown {table} owner {uid!r}")
        new_uid = retained.get((table, uid))
        if new_uid is None:
            return
        old = spec.old_key(table, field, uid)
        target = old if spec.keeps_legacy_keys else value_key(new_uid, field)
        values.append(
            LegacyValue(
                table,
                field,
                uid,
                spec.name,
                old,
                target,
                new_uid,
                size,
                signature,
                legacy_scalar,
            )
        )

    try:
        for spec in specs:
            if spec.remote:
                continue
            catalog_values = (
                has_catalog_rows(Path(spec.location))
                if spec.backend == "sqlite"
                else manager(spec).has_catalog_values()
            )
            if catalog_values:
                raise ValueError(
                    f"Value store {spec.name!r} already contains migrated values. "
                    "Restore the metadata index and every value store "
                    "from the same backup."
                )
        if primary.backend == "sqlite":
            for table, field, uid, size, sig, legacy_scalar, sentinel in _sqlite_rows(
                primary
            ):
                spec = primary
                if heavy and sentinel:
                    spec = heavy[0]
                    heavy_size, heavy_signature = _node_info(
                        spec, manager(spec), spec.old_key(table, field, uid)
                    )
                    add(table, field, uid, spec, heavy_size, heavy_signature)
                else:
                    add(table, field, uid, spec, size, sig, legacy_scalar)
        else:
            for table in _TABLES:
                primary_store = manager(primary)
                for field in primary_store.list_fields(table=table):
                    for uid in primary_store.list_objs_has_field(field, table=table):
                        spec = primary
                        if heavy:
                            value = primary_store.get_field(uid, field, table=table)
                            if isinstance(value, str) and value == "__HEAVY__":
                                spec = heavy[0]
                            del value
                        key = spec.old_key(table, field, uid)
                        if spec.backend == "sqlite":
                            size, sig, legacy_scalar = sqlite_heavy[table, field, uid]
                        else:
                            size, sig = _node_info(spec, manager(spec), key)
                            legacy_scalar = False
                        add(table, field, uid, spec, size, sig, legacy_scalar)
        return values
    finally:
        for store in managers.values():
            store.close()


def legacy_keys(specs: list[StoreSpec]) -> set[tuple[str, str, str, str]]:
    """Enumerate existing physical legacy keys, including unselected duplicates."""
    result = set()
    for spec in specs:
        if spec.backend == "sqlite":
            result.update(
                (spec.name, *key) for key in legacy_row_keys(Path(spec.location))
            )
        else:
            store = spec.open(readonly=True)
            try:
                result.update(
                    (spec.name, table, field, uid)
                    for table in _TABLES
                    for field in store.list_fields(table=table)
                    for uid in store.list_objs_has_field(field, table=table)
                )
            finally:
                store.close()
    return result


def local_file_versions(specs: list[StoreSpec]) -> bytes:
    """Fingerprint closed HDF5 files and local Zarr chunks across preflight."""
    digest = hashlib.sha256()
    for spec in specs:
        if spec.remote or spec.backend == "sqlite":
            continue
        path = Path(spec.location)
        files = sorted(path.rglob("*")) if path.is_dir() else (path,)
        for entry in files:
            st = entry.stat()
            digest.update(os.fsencode(entry))
            digest.update(
                str(
                    (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                ).encode()
            )
    return digest.digest()


def verify_values(specs: list[StoreSpec], values: list[LegacyValue]) -> None:
    """Reconcile every target against its pre-migration data or object identity."""
    for spec in specs:
        owned = [v for v in values if v.backend == spec.name]
        if spec.backend == "sqlite":
            observed = {
                (table, field, uid): sig
                for table, field, uid, _, sig, _, _ in _sqlite_rows(spec)
            }
            for value in owned:
                if observed.get(row_address(value.new_key)) != value.signature:
                    raise ValueError(f"Verification failed for {value.new_key}")
        else:
            store = spec.open(readonly=True)
            try:
                for value in owned:
                    if _node_info(spec, store, value.new_key)[1] != value.signature:
                        raise ValueError(f"Verification failed for {value.new_key}")
            finally:
                store.close()
