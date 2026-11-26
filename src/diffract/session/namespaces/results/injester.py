from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diffract.core.data.nn.aggregates.identity import aggregate_identity
from diffract.session._identifiers import check_identifier

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.params.interface import IParameterProxy, IParameterView

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class FieldIngestionError(Exception):
    """Error during field ingestion."""


class FieldIngester:
    """Validate conflicts and write precomputed parameter fields."""

    def ingest(
        self,
        *,
        fields_by_uid: dict[str, dict[str, Any]],
        parameters: IParameterView,
        force: bool = False,
    ) -> None:
        """Ingest precomputed fields with validation.

        Args:
            fields_by_uid: Mapping of parameter uid -> {field_name: value}.
            parameters: Parameter view containing target parameters.
            force: If True, overwrite existing field values.

        Raises:
            FieldIngestionError: If unknown uids or conflicts are detected.
        """
        if not fields_by_uid:
            logger.debug("ingest called with empty input, skipping")
            return

        resolved, unknown_uids = self._resolve_parameters(fields_by_uid, parameters)

        if unknown_uids:
            preview = ", ".join(sorted(unknown_uids)[:_ERROR_PREVIEW_LIMIT])
            suffix = "..." if len(unknown_uids) > _ERROR_PREVIEW_LIMIT else ""
            raise FieldIngestionError(
                f"Unknown parameter UIDs (not found in session): {preview}{suffix}"
            )

        conflicts = self._validate_fields(fields_by_uid, resolved, force)

        if conflicts:
            preview = ", ".join(
                f"{uid}:{f}" for uid, f in conflicts[:_ERROR_PREVIEW_LIMIT]
            )
            suffix = "..." if len(conflicts) > _ERROR_PREVIEW_LIMIT else ""
            raise FieldIngestionError(
                f"Field conflicts (already exist) for {len(conflicts)} entries: "
                f"{preview}{suffix}"
            )

        self._write_fields(fields_by_uid, resolved, parameters)

        total_fields = sum(len(fm) for fm in fields_by_uid.values())
        logger.info(
            "Ingested %d fields for %d parameters", total_fields, len(fields_by_uid)
        )

    def _resolve_parameters(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        parameters: IParameterView,
    ) -> tuple[dict[str, IParameterProxy], list[str]]:
        """Resolve parameter proxies from UIDs."""
        resolved: dict[str, IParameterProxy] = {}
        unknown_uids: list[str] = []

        for uid in fields_by_uid:
            try:
                resolved[uid] = parameters[uid]
            except KeyError:
                unknown_uids.append(uid)

        return resolved, unknown_uids

    def _validate_fields(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        resolved: dict[str, IParameterProxy],
        force: bool,
    ) -> list[tuple[str, str]]:
        """Validate fields for conflict issues."""
        conflicts: list[tuple[str, str]] = []

        for uid, field_map in fields_by_uid.items():
            param = resolved[uid]
            conflicts.extend(
                (uid, field_name)
                for field_name in field_map
                if (not force) and param.has_field(field_name)
            )

        return conflicts

    def _write_fields(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        resolved: dict[str, IParameterProxy],
        parameters: IParameterView,
    ) -> None:
        """Write field values to parameters.

        Note:
            Uses batch write context from parameters view. If an error occurs
            mid-write, partial writes may be committed depending on storage
            backend. Callers should handle this as a potential partial failure.
        """
        with parameters:
            for uid, field_map in fields_by_uid.items():
                param = resolved[uid]
                for field_name, value in field_map.items():
                    param.set_field(field_name, value)


class AggregateIngestionError(Exception):
    """Error during aggregate ingestion."""


class AggregateIngester:
    """Utility for ingesting precomputed aggregate values.

    Validates and writes precomputed aggregate values into the aggregate repository.

    Example:
        >>> ingester = AggregateIngester()
        >>> ingester.ingest(
        ...     aggregates=[
        ...         {
        ...             "field_name": "l_overlap",
        ...             "context_models": ("m1", "m2"),
        ...             "value": arr,
        ...         }
        ...     ],
        ...     repository=aggregate_repository,
        ...     force=False,
        ... )
    """

    def ingest(
        self,
        *,
        aggregates: list[dict[str, Any]],
        repository: AggregateRepository,
        force: bool = False,
        parameter_uids: list[str] | None = None,
    ) -> None:
        """Ingest precomputed aggregate values with validation.

        Args:
            aggregates: List of aggregate dictionaries, each containing:
                - field_name: Base field name (e.g., "l_overlap").
                - context_models: Tuple/list of model IDs.
                - context_params: Tuple/list of parameter names (optional).
                - value: The computed value to store.
            repository: Target aggregate repository.
            force: If True, overwrite existing aggregate values.
            parameter_uids: Optional parameter scope for resolving names.

        Raises:
            AggregateIngestionError: If aggregate structure is invalid or
                conflicts are detected (force=False).
        """
        if not aggregates:
            logger.debug("ingest_aggregates called with empty input, skipping")
            return

        self._validate_structure(aggregates)

        with repository:
            resolved = []
            seen = set()
            for agg in aggregates:
                try:
                    members = repository.resolve_members(
                        tuple(agg["context_models"]),
                        tuple(agg.get("context_params", ())),
                        parameter_uids=parameter_uids,
                    )
                except ValueError as exc:
                    raise AggregateIngestionError(str(exc)) from exc
                identity = aggregate_identity(agg["field_name"], members)
                if not force and (
                    identity.uid in seen or repository.has_uid(identity.uid)
                ):
                    raise AggregateIngestionError(
                        f"Aggregate conflicts detected: {identity.uid}"
                    )
                seen.add(identity.uid)
                resolved.append((agg, members))
            for agg, members in resolved:
                proxy = repository.get_or_create(agg["field_name"], member_uids=members)
                proxy.set_field("value", agg["value"])
        logger.info("Ingested %d aggregates", len(aggregates))

    def _validate_structure(self, aggregates: list[dict[str, Any]]) -> None:
        """Validate aggregate structure."""
        required_keys = ("field_name", "context_models", "value")

        for i, agg in enumerate(aggregates):
            if not isinstance(agg, dict):
                raise AggregateIngestionError(
                    f"Aggregate at index {i} must be a mapping"
                )
            for key in required_keys:
                if key not in agg:
                    raise AggregateIngestionError(
                        f"Aggregate at index {i} missing '{key}'"
                    )
            if not isinstance(agg["field_name"], str):
                raise AggregateIngestionError(
                    f"Aggregate at index {i} requires a field name"
                )
            check_identifier(agg["field_name"], kind="aggregate field name")
            for key, kind in (
                ("context_models", "model id"),
                ("context_params", "parameter name"),
            ):
                names = agg.get(key, ())
                if not isinstance(names, (list, tuple)) or not all(
                    isinstance(name, str) for name in names
                ):
                    raise AggregateIngestionError(
                        f"Aggregate at index {i} requires a sequence of names for {key}"
                    )
                for name in names:
                    check_identifier(name, kind=kind)
