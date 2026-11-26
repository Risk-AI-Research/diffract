"""Value pointers kept in the metadata index."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from diffract.core.constants import TABLE_AGGREGATES, TABLE_PARAMETERS
from diffract.core.data.metadata.runs import Generation
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.utils.hashing import new_entity_uid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffract.core.data.metadata.interface import IMetadataIndex

type ObjectKind = Literal["parameter", "aggregate"]

OBJECT_KIND_BY_TABLE: dict[str, ObjectKind] = {
    TABLE_PARAMETERS: "parameter",
    TABLE_AGGREGATES: "aggregate",
}


@dataclass(frozen=True, slots=True)
class ValuePointer:
    """Physical location and encoding version of a single field."""

    backend: str
    storage_key: str
    size_bytes: int | None = None
    format_version: int = 1
    revision: str = field(default_factory=new_entity_uid)
    generation: Generation | None = None


class ValueCatalog:
    """Resolve stored fields and track their logical deletion."""

    def __init__(
        self, index: IMetadataIndex, *, supports_generations: bool = False
    ) -> None:
        self.index = index
        self.supports_generations = supports_generations

    def get(self, kind: ObjectKind, uid: str, field: str) -> ValuePointer:
        """Look up a live pointer, raising KeyError if the field is missing."""
        pointer = self.find(kind, uid, field)
        if pointer is None:
            raise KeyError((uid, field))
        return pointer

    def find(self, kind: ObjectKind, uid: str, field: str) -> ValuePointer | None:
        """Return the live pointer when the field exists."""
        if type(self.index) is SQLiteMetadataIndex:
            row = self.index.read_live_field(
                uid, field, kind, include_generation=self.supports_generations
            )
            if row is None:
                return None
            generation = (
                Generation(*row[5:])
                if self.supports_generations and row[5] is not None
                else None
            )
            return ValuePointer(*row[:5], generation=generation)
        columns = ("backend", "storage_key", "size_bytes", "format_version", "revision")
        if self.supports_generations:
            columns += ("run_id", "output_name", "built_at_seq", "changed_at_seq")
        key = {"obj_kind": kind, "obj_uid": uid, "field": field}
        point_lookup = getattr(type(self.index), "find_row", None)
        if point_lookup is not None:
            row = point_lookup(
                self.index, "live_field_values", columns=columns, where=key
            )
        else:
            rows = self.index.query_rows(
                "live_field_values", columns=columns, where=key
            )
            row = rows[0] if rows else None
        if row is None:
            return None
        generation = None
        if self.supports_generations:
            row = dict(row)
            fields = {
                name: row.pop(name)
                for name in ("run_id", "output_name", "built_at_seq", "changed_at_seq")
            }
            if fields["run_id"] is not None:
                generation = Generation(**fields)
        return ValuePointer(**row, generation=generation)

    def bind(
        self,
        kind: ObjectKind,
        uid: str,
        field: str,
        pointer: ValuePointer,
        *,
        insert_only: bool = False,
    ) -> bool:
        """Bind a pointer and report whether it was written."""
        if type(self.index) is SQLiteMetadataIndex:
            return self.index.write_field_pointer(
                uid,
                field,
                kind,
                pointer.backend,
                pointer.storage_key,
                pointer.size_bytes,
                pointer.format_version,
                pointer.revision,
                generation=pointer.generation,
                insert_only=insert_only,
            )
        if insert_only:
            return False
        key = {"obj_uid": uid, "field": field}
        values = {
            "obj_kind": kind,
            "backend": pointer.backend,
            "storage_key": pointer.storage_key,
            "size_bytes": pointer.size_bytes,
            "format_version": pointer.format_version,
            "revision": pointer.revision,
            "tombstoned_at": None,
        }
        if pointer.generation is not None:
            generation = pointer.generation
            values.update(
                run_id=generation.run_id,
                output_name=generation.output_name,
                built_at_seq=generation.built_at_seq,
                changed_at_seq=generation.changed_at_seq,
            )
        upsert = getattr(type(self.index), "upsert_row", None)
        if upsert is not None:
            upsert(self.index, "field_values", key=key, values=values)
            return True
        with self.index:
            if not self.index.update_rows("field_values", where=key, values=values):
                self.index.insert_rows("field_values", [key | values])
        return True

    def discard_binding(self, uid: str, field: str, revision: str) -> None:
        """Remove an unpublished binding for the given revision."""
        self.index.delete_rows(
            "field_values", where={"obj_uid": uid, "field": field, "revision": revision}
        )

    def fields(self, kind: ObjectKind, uid: str | None = None) -> list[str]:
        """List live field names for one owner or an entire object kind."""
        where = {"obj_kind": kind}
        if uid is not None:
            where["obj_uid"] = uid
        return [
            row["field"]
            for row in self.index.query_rows(
                "live_field_values",
                columns=("field",),
                where=where,
                distinct=True,
                order_by=("field",),
            )
        ]

    def objects(self, kind: ObjectKind, field: str | None = None) -> list[str]:
        """List owners with at least one live field matching the selection."""
        where = {"obj_kind": kind}
        if field is not None:
            where["field"] = field
        return [
            row["obj_uid"]
            for row in self.index.query_rows(
                "live_field_values",
                columns=("obj_uid",),
                where=where,
                distinct=True,
                order_by=("obj_uid",),
            )
        ]

    def tombstone(
        self, kind: ObjectKind, *, uid: str | None = None, field: str | None = None
    ) -> int:
        """Mark matching live values for collection and return their count."""
        where = {"obj_kind": kind}
        if uid is not None:
            where["obj_uid"] = uid
        if field is not None:
            where["field"] = field
        return self.index.update_rows(
            "field_values",
            where=where,
            where_null=("tombstoned_at",),
            values={"tombstoned_at": datetime.now(UTC).isoformat()},
        )

    def tombstone_selection(
        self,
        kind: ObjectKind,
        uids: Sequence[str],
        fields: Sequence[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Tombstone live fields and return the exact owner and field pairs removed."""
        scope = {"obj_uid": uids}
        if fields is not None:
            scope["field"] = fields
        rows = self.index.update_rows_returning(
            "field_values",
            where={"obj_kind": kind},
            where_in=scope,
            where_null=("tombstoned_at",),
            values={"tombstoned_at": datetime.now(UTC).isoformat()},
            columns=("obj_uid", "field"),
        )
        return [(row["obj_uid"], row["field"]) for row in rows]

    def tombstone_result_groups(
        self,
        kind: ObjectKind,
        uids: Sequence[str],
        fields: Sequence[str] | None,
    ) -> list[tuple[str, str]]:
        """Erase the selected kernel output groups and independent ingested values."""
        if not uids:
            return []
        if not self.supports_generations:
            return self.tombstone_selection(kind, uids, fields)

        with self.index:
            scope = {"obj_uid": uids}
            if fields is not None:
                scope["field"] = fields
            columns = ("obj_uid", "field", "run_id")
            selected = self.index.query_rows(
                "live_field_values",
                columns=columns,
                where={"obj_kind": kind},
                where_in=scope,
            )
            runs = self.index.query_rows(
                "runs",
                columns=("run_id",),
                where={"kind": "kernel"},
                where_in={
                    "run_id": tuple(
                        {row["run_id"] for row in selected if row["run_id"]}
                    )
                },
            )
            kernel_runs = {row["run_id"] for row in runs}
            groups = {
                (row["obj_uid"], row["run_id"])
                for row in selected
                if row["run_id"] in kernel_runs
            }
            owners_by_field: dict[str, set[str]] = defaultdict(set)
            for row in selected:
                owners_by_field[row["field"]].add(row["obj_uid"])
            if groups:
                siblings_scope = {"run_id": tuple(kernel_runs)}
                if kind == "parameter":
                    siblings_scope["obj_uid"] = tuple({uid for uid, _ in groups})
                siblings = self.index.query_rows(
                    "live_field_values",
                    columns=columns,
                    where={"obj_kind": kind},
                    where_in=siblings_scope,
                )
                for row in siblings:
                    if kind == "aggregate" or (row["obj_uid"], row["run_id"]) in groups:
                        owners_by_field[row["field"]].add(row["obj_uid"])

            return [
                pair
                for field, owners in owners_by_field.items()
                for pair in self.tombstone_selection(kind, tuple(owners), (field,))
            ]
