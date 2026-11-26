"""Persistent model identities backed by the metadata index."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Self

from diffract.core.constants import REGEX_PREFIX, TABLE_PARAMETERS
from diffract.core.data.nn.extractors.report import ExtractionItem, ExtractionReport
from diffract.core.data.utils import build_matcher
from diffract.core.utils.hashing import new_entity_uid

from .metadata import InstanceMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from diffract.core.data.metadata.interface import IMetadataIndex


class InstanceRepository:
    """Resolve model names to stable instance identifiers."""

    TABLE = "instances"
    EXTRACTIONS = "checkpoint_extractions"

    def __init__(self, metadata_index: IMetadataIndex) -> None:
        self._metadata_index = metadata_index

    @classmethod
    def initialize(cls, metadata_index: IMetadataIndex) -> Self:
        """Bind the repository to the index that owns the instance schema."""
        metadata_index.define_table(cls.EXTRACTIONS, {})
        return cls(metadata_index)

    @staticmethod
    def _extraction_key(instance_uid: str, step: int) -> str:
        return f"{instance_uid}/{step}"

    def set_extraction(
        self, instance_uid: str, step: int, report: ExtractionReport
    ) -> None:
        """Record the source inputs observed at checkpoint admission."""
        self._metadata_index.upsert(
            self.EXTRACTIONS,
            self._extraction_key(instance_uid, step),
            items=[asdict(item) for item in report.items],
        )

    def extraction(self, instance_uid: str, step: int) -> ExtractionReport | None:
        """Read the admission report when the checkpoint has a recorded extraction."""
        row = self._metadata_index.get(
            self.EXTRACTIONS, self._extraction_key(instance_uid, step)
        )
        if row is None:
            return None
        return ExtractionReport(tuple(ExtractionItem(**item) for item in row["items"]))

    def erase_extraction(self, instance_uid: str, step: int) -> None:
        """Discard the admission record when its checkpoint is erased."""
        self._metadata_index.delete(
            self.EXTRACTIONS, self._extraction_key(instance_uid, step)
        )

    def prune_empty(
        self, coordinates: Iterable[tuple[str, int]], instances: Iterable[str]
    ) -> None:
        """Remove labels, reports and identities left without parameters by erasure."""
        with self._metadata_index as index:
            for uid, step in coordinates:
                coordinate = {"instance_uid": uid, "step": step}
                if not index.query(TABLE_PARAMETERS, where=coordinate, limit=1):
                    index.delete_rows("step_labels", where=coordinate)
                    self.erase_extraction(uid, step)
            for uid in instances:
                owner = {"instance_uid": uid}
                if not index.query(TABLE_PARAMETERS, where=owner, limit=1):
                    index.delete_rows("step_labels", where=owner)
                    index.delete_rows("instances", where=owner)

    def get_or_create(self, name: str) -> str:
        """Return the identity for a name, creating it on first use."""
        return self.get_or_create_instance(name).instance_uid

    def get_or_create_instance(self, name: str) -> InstanceMetadata:
        """Reserve a name and return its persistent record in one transaction."""
        with self._metadata_index as index:
            index.insert_rows(
                self.TABLE,
                [
                    {
                        "instance_uid": new_entity_uid(),
                        "name": name,
                        "created_at": int(time.time()),
                    }
                ],
                on_conflict="ignore",
                conflict_target=("name",),
            )
            (row,) = index.query_rows(self.TABLE, where={"name": name})
            return self._metadata(row)

    @staticmethod
    def _metadata(row: dict[str, Any]) -> InstanceMetadata:
        return InstanceMetadata(
            instance_uid=row["instance_uid"],
            name=row["name"],
            display_label=row["display_label"],
            created_at=row["created_at"],
            json_data=json.loads(row["json_data"]),
        )

    def _find(self, **where: str) -> InstanceMetadata | None:
        rows = self._metadata_index.query_rows(self.TABLE, where=where)
        return self._metadata(rows[0]) if rows else None

    def get(self, instance_uid: str) -> InstanceMetadata | None:
        """Look up an instance by its persistent identifier."""
        return self._find(instance_uid=instance_uid)

    def get_by_name(self, name: str) -> InstanceMetadata | None:
        """Look up an instance by its case-sensitive name."""
        return self._find(name=name)

    def rename(self, instance_uid: str, new_name: str) -> None:
        """Change an instance name, raising KeyError for an unknown identifier."""
        count = self._metadata_index.update_rows(
            self.TABLE, where={"instance_uid": instance_uid}, values={"name": new_name}
        )
        if count != 1:
            raise KeyError(instance_uid)

    def resolve(self, names: tuple[str, ...], *, inverse: bool = False) -> list[str]:
        """Resolve exact handles and regular expressions to store-local identities."""
        if inverse or any(name.startswith(REGEX_PREFIX) for name in names):
            matches = build_matcher(names)
            rows = self._metadata_index.query_rows(
                self.TABLE, columns=("instance_uid", "name")
            )
            return [
                row["instance_uid"] for row in rows if matches(row["name"]) != inverse
            ]
        rows = self._metadata_index.query_rows(
            self.TABLE, columns=("instance_uid",), where_in={"name": names}
        )
        return [row["instance_uid"] for row in rows]

    def step_labels(self) -> dict[tuple[str, int], str]:
        """Read sparse display labels by model and step."""
        return {
            (row["instance_uid"], row["step"]): row["label"]
            for row in self._metadata_index.query_rows("step_labels")
        }

    def set_step_label(
        self, instance_uid: str, step: int, label: str, *, overwrite: bool
    ) -> None:
        """Store a checkpoint label, optionally replacing its current value."""
        key = {"instance_uid": instance_uid, "step": step}
        with self._metadata_index as index:
            if overwrite and index.update_rows(
                "step_labels", where=key, values={"label": label}
            ):
                return
            index.insert_rows(
                "step_labels",
                [{**key, "label": label}],
                on_conflict="ignore",
                conflict_target=("instance_uid", "step"),
            )

    def names(self, uids: list[str] | None = None) -> list[str]:
        """Read sorted handles for all instances or a selected set."""
        rows = self._metadata_index.query_rows(
            self.TABLE,
            columns=("name",),
            where_in={"instance_uid": uids} if uids is not None else None,
            order_by=("name",),
        )
        return [row["name"] for row in rows]

    def list(self) -> list[InstanceMetadata]:
        """Return all instances in name order."""
        return [
            self._metadata(row)
            for row in self._metadata_index.query_rows(self.TABLE, order_by=("name",))
        ]
