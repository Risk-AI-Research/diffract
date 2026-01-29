"""Field and aggregate ingestion utilities for importing precomputed data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.proxy import ParameterDataProxy

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class FieldIngestionError(Exception):
    """Error during field ingestion."""


class FieldIngester:
    """Utility for ingesting precomputed fields into parameters.

    Validates and writes precomputed field values into parameters.

    Example:
        >>> ingester = FieldIngester()
        >>> ingester.ingest(
        ...     fields_by_uid={"param_uid": {"metric": 0.95, "loss": 0.05}},
        ...     parameters=session._get_view(),
        ...     force=False,
        ... )
    """

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
            FieldIngestionError: If unknown uids, forbidden fields,
                or conflicts are detected.
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

        forbidden, conflicts = self._validate_fields(fields_by_uid, resolved, force)

        if forbidden:
            preview = ", ".join(
                f"{uid}:{f}" for uid, f in forbidden[:_ERROR_PREVIEW_LIMIT]
            )
            suffix = "..." if len(forbidden) > _ERROR_PREVIEW_LIMIT else ""
            raise FieldIngestionError(
                f"Forbidden fields requested for {len(forbidden)} entries: "
                f"{preview}{suffix}"
            )

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
    ) -> tuple[dict[str, ParameterDataProxy], list[str]]:
        """Resolve parameter proxies from UIDs."""
        resolved: dict[str, ParameterDataProxy] = {}
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
        resolved: dict[str, ParameterDataProxy],
        force: bool,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Validate fields for forbidden and conflict issues."""
        forbidden: list[tuple[str, str]] = []
        conflicts: list[tuple[str, str]] = []

        for uid, field_map in fields_by_uid.items():
            param = resolved[uid]
            for field_name in field_map:
                if (not force) and param.has_field(field_name):
                    conflicts.append((uid, field_name))

        return forbidden, conflicts

    def _write_fields(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        resolved: dict[str, ParameterDataProxy],
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
        ...         {"field_name": "l_overlap", "context_models": ("m1", "m2"), "value": arr}
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

        Raises:
            AggregateIngestionError: If aggregate structure is invalid or
                conflicts are detected (force=False).
        """
        if not aggregates:
            logger.debug("ingest_aggregates called with empty input, skipping")
            return

        self._validate_structure(aggregates)

        if not force:
            conflicts = self._check_conflicts(aggregates, repository)
            if conflicts:
                preview = ", ".join(conflicts[:_ERROR_PREVIEW_LIMIT])
                suffix = "..." if len(conflicts) > _ERROR_PREVIEW_LIMIT else ""
                raise AggregateIngestionError(
                    f"Aggregate conflicts detected: {preview}{suffix}"
                )

        self._write_aggregates(aggregates, repository)
        logger.info("Ingested %d aggregates", len(aggregates))

    def _validate_structure(self, aggregates: list[dict[str, Any]]) -> None:
        """Validate aggregate structure."""
        required_keys = ("field_name", "context_models", "value")

        for i, agg in enumerate(aggregates):
            for key in required_keys:
                if key not in agg:
                    raise AggregateIngestionError(
                        f"Aggregate at index {i} missing '{key}'"
                    )

    def _check_conflicts(
        self,
        aggregates: list[dict[str, Any]],
        repository: AggregateRepository,
    ) -> list[str]:
        """Check for existing aggregates that would conflict."""
        from diffract.core.data.nn.aggregates.metadata import AggregateMetadata

        existing_uids = set(repository.create_view().list_uids())
        conflicts: list[str] = []

        for agg in aggregates:
            uid = AggregateMetadata.create_uid_from_context(
                field_name=agg["field_name"],
                context_models=tuple(agg["context_models"]),
                context_params=tuple(agg.get("context_params", [])),
            )
            if uid in existing_uids:
                conflicts.append(uid)

        return conflicts

    def _write_aggregates(
        self,
        aggregates: list[dict[str, Any]],
        repository: AggregateRepository,
    ) -> None:
        """Write aggregate values to repository."""
        with repository.storage_manager:
            for agg in aggregates:
                aggregate = repository.get_or_create(
                    field_name=agg["field_name"],
                    context_models=tuple(agg["context_models"]),
                    context_params=tuple(agg.get("context_params", [])),
                )
                aggregate.set_field("value", agg["value"])
