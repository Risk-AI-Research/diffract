"""Reconcile a catalog descriptor after an interrupted in-place overwrite."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

from diffract.core.data.metadata.runs import Generation

from .catalog import ObjectKind, ValueCatalog, ValuePointer
from .serialization import encode_value

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .interface import IPhysicalStorageManager


@dataclass(frozen=True, slots=True)
class _Overwrite:
    kind: ObjectKind
    uid: str
    field: str
    backend: str
    storage_key: str
    original_digest: str | None
    previous_revision: str | None = None
    generation: Generation | None = None


def _digest(value: Any) -> str:
    data, tag = encode_value(value)
    digest = hashlib.sha256(tag.encode())
    digest.update(data)
    return digest.hexdigest()


class OverwriteRecovery:
    """Record small intents when overwriting a live physical key.

    The metadata index serializes writers. An intent survives process loss
    independently of the index transaction. It contains no value payload.
    """

    def __init__(self, path: str | None) -> None:
        self.directory = (
            Path(path + ".pending") if path and path != ":memory:" else None
        )
        self._memory: list[_Overwrite] = []
        self.pending = self.directory is not None and self.directory.is_dir()

    def record(
        self,
        kind: ObjectKind,
        uid: str,
        field: str,
        old: ValuePointer,
        new: ValuePointer,
        read: Callable[[str], Any],
    ) -> None:
        """Record an in-place change to a live value descriptor."""
        if (old.backend, old.storage_key) != (
            new.backend,
            new.storage_key,
        ) or old == new:
            return
        signature = _digest(read(old.storage_key)) if old.format_version == 0 else None
        intent = _Overwrite(
            kind,
            uid,
            field,
            new.backend,
            new.storage_key,
            signature,
            old.revision,
            new.generation,
        )
        if self.directory is None:
            self._memory.append(intent)
        else:
            self.directory.mkdir(exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as temp:
                payload = asdict(intent)
                if intent.generation is None:
                    payload.pop("generation")
                json.dump(payload, temp)
                temp.flush()
                os.fsync(temp.fileno())
            # Only a complete, closed record is visible to another session.
            Path(temp.name).rename(temp.name + ".json")
        self.pending = True

    def _has_intents(self) -> bool:
        if not self.pending and self.directory and self.directory.is_dir():
            self.pending = next(self.directory.glob("*.json"), None) is not None
        return self.pending

    def reconcile_locked(
        self, catalog: ValueCatalog, stores: Mapping[str, IPhysicalStorageManager]
    ) -> tuple[tuple[Path, ...], int]:
        """Apply existing intents while the caller holds the metadata writer lock."""
        if not self._has_intents():
            return (), 0
        files = tuple(self.directory.glob("*.json")) if self.directory else ()
        memory = tuple(self._memory)
        for path in files:
            payload = json.loads(path.read_text())
            generation = payload.pop("generation", None)
            if generation is not None:
                payload["generation"] = Generation(**generation)
            self._reconcile_one(_Overwrite(**payload), catalog, stores)
        for intent in memory:
            self._reconcile_one(intent, catalog, stores)
        return files, len(memory)

    def reconcile(
        self, catalog: ValueCatalog, stores: Mapping[str, IPhysicalStorageManager]
    ) -> None:
        """Match interrupted catalog updates to the value that survived."""
        if not self._has_intents() or getattr(catalog.index, "in_write_context", False):
            return
        with catalog.index:
            files, memory_count = self.reconcile_locked(catalog, stores)
        # The corrected index is committed before the corresponding intent is dropped.
        with catalog.index:
            for path in files:
                with suppress(FileNotFoundError):
                    path.unlink()
            del self._memory[:memory_count]
            self.pending = bool(self._memory) or bool(
                self.directory and next(self.directory.glob("*.json"), None)
            )

            if self.directory and not self.pending:
                with suppress(OSError):
                    self.directory.rmdir()

    @staticmethod
    def _reconcile_one(
        intent: _Overwrite,
        catalog: ValueCatalog,
        stores: Mapping[str, IPhysicalStorageManager],
    ) -> None:
        try:
            pointer = catalog.get(intent.kind, intent.uid, intent.field)
        except KeyError:
            return
        if (pointer.backend, pointer.storage_key) != (
            intent.backend,
            intent.storage_key,
        ):
            return
        if (
            intent.previous_revision is not None
            and pointer.revision != intent.previous_revision
        ):
            return
        store = stores[pointer.backend]
        metadata = store.blob_metadata(pointer.storage_key)
        if metadata is None:
            raise OSError(
                f"Cannot recover value metadata for {intent.uid}/{intent.field}"
            )
        size = None
        if metadata.get("shape") is not None:
            try:
                size = np.dtype(metadata["dtype"]).itemsize * prod(metadata["shape"])
            except TypeError:
                size = store.read_blob(pointer.storage_key).nbytes
        version = pointer.format_version
        if version == 0 and intent.original_digest is not None:
            if _digest(store.read_blob(pointer.storage_key)) == intent.original_digest:
                return
            version = 1
        # An interrupted in-place write might have landed its payload before
        # the index committed. Make it visible to recovery as a suspect value.
        generation = intent.generation or pointer.generation
        corrected = ValuePointer(
            pointer.backend,
            pointer.storage_key,
            size,
            version,
            generation=generation,
        )
        catalog.bind(intent.kind, intent.uid, intent.field, corrected)
