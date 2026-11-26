"""Persistence and membership of aggregates over stored parameters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from diffract.core.constants import TABLE_AGGREGATES, TABLE_PARAMETERS
from diffract.core.data.metadata.interface import JoinSpec
from diffract.core.data.repository import DataRepository

from .identity import AggregateIdentity, aggregate_identity
from .metadata import AggregateMetadata
from .proxy import AggregateProxy
from .view import AggregateView

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class AggregateRepository(DataRepository[AggregateMetadata, AggregateProxy]):
    """Store aggregate fields and resolve their context through parameter membership."""

    METADATA_CLASS = AggregateMetadata
    PROXY_CLASS = AggregateProxy
    VIEW_CLASS = AggregateView
    TABLE = TABLE_AGGREGATES
    MEMBERS = "aggregate_members"
    METADATA_COLUMNS: ClassVar[dict[str, type]] = {"field_name": str}
    METADATA_INDEXES: ClassVar[list[str]] = []

    def resolve_members(
        self,
        models: tuple[str, ...],
        params: tuple[str, ...],
        *,
        parameter_uids: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Resolve context names to members of an optional parameter scope."""
        filters = {}
        if models:
            filters["j.name"] = models
        if params:
            filters["base.name"] = params
        if parameter_uids is not None:
            filters["base.uid"] = parameter_uids
        if not (models or params or parameter_uids):
            return ()
        rows = self.metadata_index.query_joined_rows(
            TABLE_PARAMETERS,
            JoinSpec("instances", "instance_uid", "instance_uid"),
            columns=("base.uid", "base.name", "j.name"),
            where_in=filters,
        )
        missing_models = set(models) - {r["j.name"] for r in rows}
        missing_params = set(params) - {r["base.name"] for r in rows}
        if missing_models or missing_params:
            raise ValueError(
                f"Unresolved aggregate members: models={sorted(missing_models)!r}, "
                f"parameters={sorted(missing_params)!r}."
            )
        return tuple(row["base.uid"] for row in rows)

    def get_or_create(
        self,
        field_name: str,
        context_models: tuple[str, ...] = (),
        context_params: tuple[str, ...] = (),
        *,
        member_uids: Iterable[str] | None = None,
        directed: bool = False,
    ) -> AggregateProxy:
        """Get or create an aggregate over concrete parameter members."""
        with self.metadata_index as index:
            if member_uids is None:
                if directed:
                    raise ValueError(
                        "Directed membership requires ordered parameter UIDs"
                    )
                member_uids = self.resolve_members(context_models, context_params)
            identity = aggregate_identity(field_name, member_uids, directed=directed)
            if self.has_uid(identity.uid):
                return self.get_proxy(identity.uid)
            requested = {uid for _, uid in identity.members}
            rows = (
                index.query_joined_rows(
                    TABLE_PARAMETERS,
                    JoinSpec("instances", "instance_uid", "instance_uid"),
                    columns=(
                        "base.uid",
                        "base.name",
                        "base.step",
                        "base.instance_uid",
                        "j.name",
                    ),
                    where_in={"base.uid": sorted(requested)},
                )
                if requested
                else []
            )
            by_uid = {r["base.uid"]: r for r in rows}
            missing = requested.difference(by_uid)
            if missing:
                raise ValueError(f"Unknown aggregate member UIDs: {sorted(missing)!r}")
            index.insert_rows(
                self.TABLE, [{"uid": identity.uid, "field_name": field_name}]
            )
            self._write_members(identity)
            ordered = tuple(uid for _, uid in identity.members)
            meta = AggregateMetadata(
                uid=identity.uid,
                field_name=field_name,
                context_models=tuple(sorted({r["j.name"] for r in rows})),
                context_params=tuple(sorted({r["base.name"] for r in rows})),
                member_uids=ordered,
                member_names=tuple(by_uid[u]["base.name"] for u in ordered),
                member_instances=tuple(by_uid[u]["base.instance_uid"] for u in ordered),
                member_coordinates=tuple(
                    (by_uid[u]["j.name"], by_uid[u]["base.step"]) for u in ordered
                ),
                directed=directed,
            )
            proxy = self.PROXY_CLASS(meta=meta, _repository=self)
            self.append(proxy)
            return proxy

    def _write_members(self, identity: AggregateIdentity) -> None:
        self.metadata_index.insert_rows(
            self.MEMBERS,
            [
                {
                    "aggregate_uid": identity.uid,
                    "member_uid": uid,
                    "role": role,
                    "ordinal": ordinal if role != "member" else None,
                }
                for ordinal, (role, uid) in enumerate(identity.members)
            ],
        )

    def member_rows(self, aggregate_uid: str) -> list[dict[str, Any]]:
        """Join an aggregate to its stored parameter identities and coordinates."""
        return self.metadata_index.query_joined_rows(
            self.MEMBERS,
            JoinSpec(TABLE_PARAMETERS, "member_uid", "uid"),
            columns=(
                "base.member_uid",
                "base.role",
                "base.ordinal",
                "base.json_data",
                "j.instance_uid",
                "j.step",
                "j.name",
            ),
            where={"base.aggregate_uid": aggregate_uid},
            order_by=("base.ordinal", "base.member_uid"),
        )

    def _decode_metadata(self, row: dict[str, Any]) -> AggregateMetadata:
        members = self.member_rows(row["uid"])
        instances = self.metadata_index.query_rows(
            "instances",
            columns=("instance_uid", "name"),
            where_in={
                "instance_uid": list({member["j.instance_uid"] for member in members})
            },
            order_by=("name",),
        )
        names = {r["instance_uid"]: r["name"] for r in instances}
        return AggregateMetadata(
            uid=row["uid"],
            field_name=row["field_name"],
            context_models=tuple(instance["name"] for instance in instances),
            context_params=tuple(sorted({member["j.name"] for member in members})),
            member_uids=tuple(member["base.member_uid"] for member in members),
            member_coordinates=tuple(
                (names[r["j.instance_uid"]], r["j.step"]) for r in members
            ),
            member_names=tuple(r["j.name"] for r in members),
            member_instances=tuple(r["j.instance_uid"] for r in members),
            direction_unknown=bool(row.get("direction_unknown"))
            or any(
                json.loads(r["base.json_data"] or "{}").get("direction_unknown")
                for r in members
                if r["base.json_data"] not in (None, "{}")
            ),
            directed=any(r["base.role"] == "source" for r in members),
        )

    def refresh_names(self) -> None:
        """Refresh membership projections after a model name changes."""
        self._proxy_cache.clear()

    def remove_membership(self, uids: Sequence[str]) -> int:
        """Remove aggregate members and their owning records together."""
        with self.metadata_index as index:
            index.delete_rows(self.MEMBERS, where_in={"aggregate_uid": uids})
            return super().remove_membership(uids)

    def referencing(self, member_uids: Sequence[str]) -> list[str]:
        """Return aggregate identifiers that reference any of the given parameters."""
        rows = self.metadata_index.query_rows(
            self.MEMBERS,
            columns=("aggregate_uid",),
            where_in={"member_uid": member_uids},
            distinct=True,
        )
        return [row["aggregate_uid"] for row in rows]

    def create_view(self) -> AggregateView:
        """Create a view of all aggregates."""
        return AggregateView(repository=self, uids=None)
