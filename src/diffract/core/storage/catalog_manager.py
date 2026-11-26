"""Logical fields backed by catalog pointers and physical value stores."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from numbers import Real
from threading import local
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from diffract.core.constants import TABLE_AGGREGATES
from diffract.core.data.metadata.interface import JoinSpec
from diffract.core.data.metadata.runs import Generation, RunJournal
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex

from .catalog import OBJECT_KIND_BY_TABLE, ValueCatalog, ValuePointer
from .hybrid_manager import HybridStorageManager
from .interface import DEFAULT_TABLE, IStorageManager
from .keys import value_key
from .overwrite_recovery import OverwriteRecovery

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from diffract.core.compute.config import KernelConfigIdentity, OutputComparison
    from diffract.core.data.metadata.interface import IMetadataIndex
    from diffract.core.data.metadata.runs import (
        OwnerVerification,
        Run,
        RunKind,
        ValueTrace,
    )

    from .interface import IPhysicalStorageManager


def _kernel_borrower(store: IStorageManager) -> Callable[[str], Any] | None:
    """Subclasses opt in explicitly to sharing mutable storage with kernels."""
    if vars(type(store)).get("borrows_kernel_inputs") is True:
        return getattr(store, "borrow_blob", None)
    return None


def storage_activity(
    storage: IStorageManager, kind: RunKind, config: KernelConfigIdentity | None = None
) -> AbstractContextManager[Run | None]:
    """Record an operation on a storage manager with a provenance journal."""
    journal = getattr(storage, "journal", None)
    return (
        journal.activity(kind, config)
        if isinstance(journal, RunJournal)
        else nullcontext()
    )


class ValueMigrationRequiredError(RuntimeError):
    """A store has pre-catalog owners whose values require a coordinated migration."""


class CatalogStorageManager(IStorageManager):
    """Route entity field operations through live catalog entries."""

    uses_value_revisions = True

    def __init__(
        self,
        backend: IPhysicalStorageManager
        | HybridStorageManager[IPhysicalStorageManager],
        index: IMetadataIndex,
    ) -> None:
        if index.query_rows("schema_meta", where={"values_require_migration": 1}):
            raise ValueMigrationRequiredError(
                "Existing owners require a coordinated migration of the metadata "
                "index and its original value store before fields can be accessed."
            )
        self.backend = backend
        self.journal = (
            RunJournal(index) if isinstance(index, SQLiteMetadataIndex) else None
        )
        self.catalog = ValueCatalog(
            index, supports_generations=self.journal is not None
        )
        self._local = local()
        self._stores: dict[str, IPhysicalStorageManager]
        if isinstance(backend, HybridStorageManager):
            self._stores = {
                "light/" + backend.light.backend_name: backend.light,
                "heavy/" + backend.heavy.backend_name: backend.heavy,
            }
        else:
            self._stores = {backend.backend_name: backend}
        self._recovery = OverwriteRecovery(getattr(index, "database_path", None))
        self._recovery.reconcile(self.catalog, self._stores)

    def connect(self) -> None:
        """Connect the physical store for this catalog."""
        if hasattr(self.backend, "connect"):
            self.backend.connect()

    def close(self) -> None:
        """Release the physical store connection."""
        self._recovery.reconcile(self.catalog, self._stores)
        if hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self) -> Self:
        """Begin a batch whose physical writes commit before its catalog entries."""
        with ExitStack() as stack:
            if self.journal is not None and not self.journal.recording:
                stack.enter_context(self.journal.activity("ingest"))
            stack.enter_context(self.catalog.index)
            if not getattr(self._local, "batches", ()):
                self._recovery.reconcile_locked(self.catalog, self._stores)
            stack.enter_context(self.backend)
            if not hasattr(self._local, "batches"):
                self._local.batches = []
            self._local.batches.append(stack.pop_all())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Flush blobs and propagate any backend failure to the catalog transaction."""
        try:
            self._local.batches.pop().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._recovery.reconcile(self.catalog, self._stores)

    def _select(self, value: Any) -> tuple[str, IPhysicalStorageManager]:
        if isinstance(self.backend, HybridStorageManager):
            heavy = self.backend.uses_heavy_store(value)
            store = self.backend.heavy if heavy else self.backend.light
            return ("heavy/" if heavy else "light/") + store.backend_name, store
        return self.backend.backend_name, self.backend

    def has_field(
        self, obj_uid: str, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> bool:
        """Check for a live catalog entry."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.has_field(obj_uid, field_name, table=table)
        try:
            self.catalog.get(OBJECT_KIND_BY_TABLE[table], obj_uid, field_name)
        except KeyError:
            return False
        return True

    def get_field(
        self, obj_uid: str, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> Any:
        """Read the value at its cataloged backend and key."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.get_field(obj_uid, field_name, table=table)
        pointer = self.catalog.get(OBJECT_KIND_BY_TABLE[table], obj_uid, field_name)
        return self._stores[pointer.backend].read_blob(pointer.storage_key)

    def live_pointer(self, obj_uid: str, field_name: str, table: str) -> ValuePointer:
        """Return the revision and physical address used by a cache-backed read."""
        self._reconcile_reads()
        return self.catalog.get(OBJECT_KIND_BY_TABLE[table], obj_uid, field_name)

    def _reconcile_reads(self) -> None:
        if getattr(self.catalog.index, "in_write_context", False) and not getattr(
            self._local, "batches", ()
        ):
            self._recovery.reconcile_locked(self.catalog, self._stores)
        else:
            self._recovery.reconcile(self.catalog, self._stores)

    def live_pointers(
        self,
        uids: Sequence[str] | None,
        fields: Sequence[str],
        *,
        table: str,
        include_generation: bool = False,
    ) -> dict[tuple[str, str], ValuePointer]:
        """Resolve fields for selected owners, or for the whole kind when unscoped."""
        self._reconcile_reads()
        scope = {"field": fields}
        if uids is not None:
            scope["obj_uid"] = uids
        columns = (
            "obj_uid",
            "field",
            "backend",
            "storage_key",
            "size_bytes",
            "format_version",
            "revision",
        )
        if include_generation and self.catalog.supports_generations:
            columns += ("run_id", "output_name", "built_at_seq", "changed_at_seq")
        rows = self.catalog.index.query_rows(
            "live_field_values",
            columns=columns,
            where={"obj_kind": OBJECT_KIND_BY_TABLE[table]},
            where_in=scope,
            order_by=("obj_uid", "field"),
        )
        return {
            (r["obj_uid"], r["field"]): ValuePointer(
                r["backend"],
                r["storage_key"],
                r["size_bytes"],
                r["format_version"],
                r["revision"],
                generation=(
                    Generation(
                        r["run_id"],
                        r["output_name"],
                        r["built_at_seq"],
                        r["changed_at_seq"],
                    )
                    if r.get("run_id") is not None
                    else None
                ),
            )
            for r in rows
        }

    @property
    def supports_unscoped_pointers(self) -> bool:
        """Whether the bulk lookup accepts an omitted owner predicate."""
        return type(self).live_pointers is CatalogStorageManager.live_pointers

    supports_direct_kernel_reads = True

    @property
    def kernel_inputs_are_resident(self) -> bool:
        """Whether this backend explicitly guarantees borrowed reads for every input."""
        return _kernel_borrower(self.backend) is not None

    def read_pointer(self, pointer: ValuePointer) -> Any:
        """Read directly from a previously resolved catalog pointer."""
        return self._stores[pointer.backend].read_blob(pointer.storage_key)

    def read_kernel_input(self, pointer: ValuePointer) -> Any:
        """Borrow a resident input or load it from its physical store."""
        store = self._stores[pointer.backend]
        borrow = _kernel_borrower(store)
        if borrow is not None:
            return borrow(pointer.storage_key)
        return self.read_pointer(pointer)

    def get_field_metadata(
        self, obj_uid: str, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> dict[str, Any] | None:
        """Read backend metadata for a live field."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.get_field_metadata(obj_uid, field_name, table=table)
        try:
            pointer = self.catalog.get(OBJECT_KIND_BY_TABLE[table], obj_uid, field_name)
        except KeyError:
            return None
        return self._stores[pointer.backend].blob_metadata(pointer.storage_key)

    def set_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Write the blob at its final key, then publish the pointer."""
        if table not in OBJECT_KIND_BY_TABLE:
            self.backend.set_field(obj_uid, field_name, value, table=table)
            return
        self.write_field(obj_uid, field_name, value, table=table)

    def write_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str
    ) -> ValuePointer:
        """Publish a value and return its current catalog pointer."""
        pointer, _ = self.publish_field(obj_uid, field_name, value, table=table)
        return pointer

    def publish_initial_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str
    ) -> tuple[ValuePointer, bool]:
        """Try inserting the first field pointer in the active write transaction."""
        return self.publish_field(obj_uid, field_name, value, table=table, initial=True)

    def publish_field(
        self,
        obj_uid: str,
        field_name: str,
        value: Any,
        *,
        table: str,
        initial: bool = False,
    ) -> tuple[ValuePointer, bool]:
        """Return the live revision and whether its payload was written."""
        key = value_key(obj_uid, field_name)
        name, store = self._select(value)
        batch = nullcontext() if getattr(self._local, "batches", ()) else self
        with batch:
            size = value.nbytes if isinstance(value, np.ndarray) else None
            generation = self._generation_for_write(obj_uid, field_name, table)
            kind = OBJECT_KIND_BY_TABLE[table]
            pointer = None
            if initial:
                pointer = ValuePointer(name, key, size, generation=generation)
                if self.catalog.bind(
                    kind, obj_uid, field_name, pointer, insert_only=True
                ):
                    try:
                        store.put_blob(key, value)
                    except BaseException:
                        self.catalog.discard_binding(
                            obj_uid, field_name, pointer.revision
                        )
                        raise
                    return pointer, True
            previous = self.catalog.find(kind, obj_uid, field_name)
            compare = self.journal.output_comparison if self.journal else None
            if (
                compare is not None
                and generation is not None
                and previous is not None
                and previous.generation is not None
                and self._matches_previous(previous, value, compare)
            ):
                generation = replace(
                    generation, changed_at_seq=previous.generation.changed_at_seq
                )
                pointer = replace(previous, generation=generation)
                self.catalog.bind(kind, obj_uid, field_name, pointer)
                return pointer, False
            if pointer is None:
                pointer = ValuePointer(name, key, size, generation=generation)
            if previous is not None:
                self._recovery.record(
                    kind,
                    obj_uid,
                    field_name,
                    previous,
                    pointer,
                    store.read_blob,
                )
            store.put_blob(key, value)
            self.catalog.bind(kind, obj_uid, field_name, pointer)
        return pointer, True

    def _matches_previous(
        self,
        pointer: ValuePointer,
        value: object,
        compare: OutputComparison,
    ) -> bool:
        if isinstance(value, np.ndarray):
            if (
                compare.max_array_bytes == 0
                or pointer.size_bytes is None
                or pointer.size_bytes + value.nbytes > compare.max_array_bytes
            ):
                return False
        elif not isinstance(value, Real) or pointer.size_bytes is not None:
            return False
        try:
            previous = self.read_pointer(pointer)
        except KeyError:
            return False
        return compare.equivalent(previous, value)

    def _generation_for_write(
        self, uid: str, field: str, table: str
    ) -> Generation | None:
        journal = self.journal
        if journal is None:
            return None
        run = journal.active
        if run is None:
            raise RuntimeError("Publishing a value requires an active run")
        output = field
        if table == TABLE_AGGREGATES and field == "value":
            row = self.catalog.index.find_row(
                TABLE_AGGREGATES, columns=("field_name",), where={"uid": uid}
            )
            if row is None:
                raise KeyError(uid)
            output = row["field_name"]
        return Generation(run.run_id, output, run.seq, run.seq)

    def fields_by_owner(
        self, uids: Sequence[str], *, table: str
    ) -> dict[str, list[str]]:
        """List live field names within the selected owners."""
        if table not in OBJECT_KIND_BY_TABLE:
            return {uid: self.backend.list_fields(uid, table=table) for uid in uids}
        self._reconcile_reads()
        result = {uid: [] for uid in uids}
        rows = self.catalog.index.query_rows(
            "live_field_values",
            columns=("obj_uid", "field"),
            where={"obj_kind": OBJECT_KIND_BY_TABLE[table]},
            where_in={"obj_uid": uids},
            order_by=("field",),
        )
        for row in rows:
            result[row["obj_uid"]].append(row["field"])
        return result

    def owners_missing_outputs(
        self, fields: Sequence[str], *, table: str
    ) -> list[str] | None:
        """Select incomplete owners across an entire catalog-backed entity table."""
        index = self.catalog.index
        lookup = getattr(type(index), "uids_missing_related_values", None)
        if lookup is None or table not in OBJECT_KIND_BY_TABLE:
            return None
        self._reconcile_reads()
        return lookup(
            index,
            table,
            JoinSpec("live_field_values", "uid", "obj_uid"),
            column="field",
            required=fields,
            where={"obj_kind": OBJECT_KIND_BY_TABLE[table]},
        )

    def has_aggregate_records(self, fields: Sequence[str]) -> bool:
        """Check whether any aggregate identity is already recorded."""
        index = self.catalog.index
        if not isinstance(index, SQLiteMetadataIndex):
            return True
        self._reconcile_reads()
        return bool(
            index.query_rows(
                TABLE_AGGREGATES,
                columns=("uid",),
                where_in={"field_name": fields},
                limit=1,
            )
        )

    def unverified_parameter_owners(
        self,
        owners: Sequence[str] | None,
        inputs: Sequence[str],
        outputs: Sequence[str],
        config_hash: str,
    ) -> OwnerVerification | None:
        """Find parameter groups requiring detailed verification."""
        index = self.catalog.index
        if not isinstance(index, SQLiteMetadataIndex):
            return None
        self._reconcile_reads()
        return index.unverified_parameter_owners(owners, inputs, outputs, config_hash)

    def read_value_traces(
        self,
        selections: Mapping[str, tuple[Sequence[str], Sequence[str]]],
        *,
        inspect_only: bool = False,
    ) -> dict[tuple[str, str, str], ValueTrace]:
        """Project live generation metadata for compute without reading values."""
        index = self.catalog.index
        if not isinstance(index, SQLiteMetadataIndex):
            raise TypeError("Value tracing requires a SQLite metadata index")
        if not inspect_only:
            self._reconcile_reads()
        return index.read_value_traces(selections, include_config=inspect_only)

    def adopt_value(
        self,
        uid: str,
        field: str,
        trace: ValueTrace,
        run: Run,
        *,
        table: str,
        output_name: str,
    ) -> None:
        """Record verification of an inherited value, preserving its payload."""
        updated = self.catalog.index.update_rows(
            "field_values",
            where={
                "obj_kind": OBJECT_KIND_BY_TABLE[table],
                "obj_uid": uid,
                "field": field,
                "revision": trace.revision,
            },
            where_null=("tombstoned_at", "built_at_seq"),
            values={
                "run_id": run.run_id,
                "output_name": output_name,
                "built_at_seq": run.seq,
            },
        )
        if updated != 1:
            raise RuntimeError(
                "A value changed during verification. Retry compute.apply."
            )

    def owners_with_fields(
        self, uids: Sequence[str], fields: Sequence[str], *, table: str, inverse: bool
    ) -> set[str]:
        """Select owners with all requested live fields, or none when inverted."""
        if table not in OBJECT_KIND_BY_TABLE:
            return {
                uid
                for uid in uids
                if all(
                    self.backend.has_field(uid, f, table=table) != inverse
                    for f in fields
                )
            }
        self._reconcile_reads()
        rows = self.catalog.index.query_rows(
            "live_field_values",
            columns=("obj_uid",),
            where={"obj_kind": OBJECT_KIND_BY_TABLE[table]},
            where_in={"obj_uid": uids, "field": fields},
        )
        counts = Counter(row["obj_uid"] for row in rows)
        target = 0 if inverse else len(set(fields))
        return {uid for uid in uids if counts[uid] == target}

    def read_sizes(
        self, required: dict[str, list[str]], *, table: str
    ) -> dict[tuple[str, str], int]:
        """Return known logical value sizes for read-budget planning."""
        self._reconcile_reads()
        fields = {field for names in required.values() for field in names}
        rows = self.catalog.index.query_rows(
            "live_field_values",
            columns=("obj_uid", "field", "size_bytes"),
            where={"obj_kind": OBJECT_KIND_BY_TABLE[table]},
            where_in={"obj_uid": list(required), "field": list(fields)},
        )
        return {
            (r["obj_uid"], r["field"]): max(1, r["size_bytes"])
            for r in rows
            if r["size_bytes"] is not None and r["field"] in required[r["obj_uid"]]
        }

    def list_fields(
        self, obj_uid: str | None = None, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """List the live fields in a scope."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.list_fields(obj_uid, table=table)
        return self.catalog.fields(OBJECT_KIND_BY_TABLE[table], obj_uid)

    def list_objs(self, *, table: str = DEFAULT_TABLE) -> list[str]:
        """Return owners with live fields."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.list_objs(table=table)
        return self.catalog.objects(OBJECT_KIND_BY_TABLE[table])

    def list_objs_has_field(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> list[str]:
        """Return owners with a live value for a field."""
        if table not in OBJECT_KIND_BY_TABLE:
            return self.backend.list_objs_has_field(field_name, table=table)
        return self.catalog.objects(OBJECT_KIND_BY_TABLE[table], field_name)

    def erase_selection(
        self,
        table: str,
        uids: Sequence[str],
        fields: Sequence[str] | None,
    ) -> list[tuple[str, str]]:
        """Tombstone a scoped selection and return fields that actually changed."""
        return self.catalog.tombstone_selection(
            OBJECT_KIND_BY_TABLE[table], uids, fields
        )

    def erase_field(
        self, obj_uid: str, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Tombstone a single cataloged field."""
        if table not in OBJECT_KIND_BY_TABLE:
            self.backend.erase_field(obj_uid, field_name, table=table)
            return
        self.catalog.tombstone(
            OBJECT_KIND_BY_TABLE[table], uid=obj_uid, field=field_name
        )

    def erase_result_groups(
        self, table: str, uids: Sequence[str], fields: Sequence[str] | None
    ) -> list[tuple[str, str]]:
        """Tombstone results together with their stored kernel siblings."""
        return self.catalog.tombstone_result_groups(
            OBJECT_KIND_BY_TABLE[table], uids, fields
        )

    def erase_obj(self, obj_uid: str, *, table: str = DEFAULT_TABLE) -> None:
        """Tombstone an owner's fields."""
        if table not in OBJECT_KIND_BY_TABLE:
            self.backend.erase_obj(obj_uid, table=table)
            return
        self.catalog.tombstone(OBJECT_KIND_BY_TABLE[table], uid=obj_uid)

    def erase_field_for_all(
        self, field_name: str, *, table: str = DEFAULT_TABLE
    ) -> None:
        """Tombstone a field across all owners of one kind."""
        if table not in OBJECT_KIND_BY_TABLE:
            self.backend.erase_field_for_all(field_name, table=table)
            return
        self.catalog.tombstone(OBJECT_KIND_BY_TABLE[table], field=field_name)

    def clear(self, *, table: str | None = None) -> None:
        """Tombstone all values in the selected entity table."""
        if table is None:
            with self.catalog.index:
                for store in self._stores.values():
                    store.clear_raw_tables({"", *OBJECT_KIND_BY_TABLE})
                for kind in OBJECT_KIND_BY_TABLE.values():
                    self.catalog.tombstone(kind)
        elif table in OBJECT_KIND_BY_TABLE:
            self.catalog.tombstone(OBJECT_KIND_BY_TABLE[table])
        else:
            self.backend.clear(table=table)
