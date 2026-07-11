"""Exporter implementation for collecting and formatting results.

The ResultExporter gathers requested fields from both parameter and aggregate
repositories, then delegates formatting to a provided IResultFormatter implementation.

Behavior:
    - Requires at least one field name to export.
    - Best-effort prefetch; continues if prefetch fails.
    - Returns formatter-specific output with separate scalars and aggregates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diffract.core.parallel import ParallelContext
from diffract.core.utils.exceptions import format_exception_message

from .interface import AggregateData, IResultExporter, IResultFormatter, ResultData

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.session.field_cache import SessionFieldCache

logger = logging.getLogger(__name__)


class SessionExportError(Exception):
    """Base exception for session export errors."""


class ResultExporter(IResultExporter):
    """Main implementation of result export functionality."""

    def export_results(
        self,
        *fields: tuple[str, ...],
        parameters: IParameterView,
        aggregates: AggregateView | None = None,
        formatter: IResultFormatter,
        parallel: ParallelContext | None = None,
        field_cache: SessionFieldCache | None = None,
    ) -> Any:
        """Export requested fields from parameter and aggregate repositories.

        Args:
            *fields: Field names to export.
            parameters: Parameter collection to read scalar values from.
            aggregates: Aggregate view to read contextual/aggregation values from.
            formatter: Formatter used to convert results to target format.
            parallel: Optional per-method parallel context for collection operations.
            field_cache: Optional session-level cache for field availability.
                If provided and valid, skips expensive list_fields_by_uid() call.

        Returns:
            Formatter-specific output object (typically StructuredExportResult).

        Raises:
            ValueError: If no fields are specified.
        """
        if not fields:
            msg = "At least one field must be specified"
            raise ValueError(msg)

        param_results = self._collect_parameter_results(
            *fields, parameters=parameters, parallel=parallel, field_cache=field_cache
        )
        aggregate_results = self._collect_aggregate_results(
            *fields, aggregates=aggregates, parallel=parallel
        )

        return formatter.format_results(param_results, aggregate_results, fields)

    def _collect_parameter_results(
        self,
        *fields: tuple[str, ...],
        parameters: IParameterView,
        parallel: ParallelContext | None,
        field_cache: SessionFieldCache | None = None,
    ) -> ResultData:
        """Collect results from parameter repository.

        Args:
            *fields: Field names to collect (exact match only, no regex patterns).
            parameters: Parameter collection to iterate over.
            parallel: Optional per-method parallel context for collection operations.
            field_cache: Optional session-level cache for field availability.
                If provided and valid, skips expensive list_fields_by_uid() call.

        Returns:
            Nested mapping with metadata and field values per parameter.
        """
        results: ResultData = {}
        fields_set = set(fields)

        if field_cache is not None and field_cache.is_valid:
            cached_fields_by_uid = field_cache.get()
            required_fields = set(parameters.list_uids())
            to_receive = tuple(set(required_fields) - set(cached_fields_by_uid.keys()))
            received = parameters[to_receive].list_fields_by_uid(parallel=parallel)
            field_cache.update(received)
            cached_fields_by_uid = field_cache.get()
            fields_by_uid = {uid: cached_fields_by_uid[uid] for uid in required_fields}
        else:
            fields_by_uid = parameters.list_fields_by_uid(parallel=parallel)
            if field_cache is not None:
                field_cache.set(fields_by_uid)

        to_receive: dict[str, list[str]] = {}
        for uid, fields_available in fields_by_uid.items():
            to_receive[uid] = [f for f in fields_available if f in fields_set]

        try:
            parameters.prefetch_fields(fields_by_uid=to_receive, parallel=parallel)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "Prefetch failed during export: %s",
                format_exception_message(e),
                exc_info=True,
            )

        for param in parameters:
            param_results: dict[str, Any] = {}
            for field in to_receive[param.meta.uid]:
                param_results[field] = param.get_field(field, auto_prefetch=False)

            results[param.meta.uid] = {
                "metadata": {
                    "name": param.meta.name,
                    "model_id": param.meta.model_id,
                    "parameter_type": param.meta.ptype.name,
                    **param.meta.other_meta,
                },
                "fields": param_results,
            }

        return results

    def _collect_aggregate_results(
        self,
        *fields: tuple[str, ...],
        aggregates: AggregateView | None,
        parallel: ParallelContext | None,
    ) -> AggregateData:
        """Collect results from aggregate repository.

        Args:
            *fields: Base field names to collect.
            aggregates: Aggregate view to iterate over.
            parallel: Optional per-method parallel context (unused currently).

        Returns:
            List of aggregate records with field name, context, and value.
        """
        if aggregates is None or not aggregates:
            return []

        fields_set = set(fields)
        results: AggregateData = []

        # Filter aggregates by requested field names
        filtered_view = aggregates.filter_by_field_name(*fields)

        # Prefetch values
        try:
            filtered_view.prefetch_fields(fields=["value"], parallel=parallel)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "Prefetch failed during aggregate export: %s",
                format_exception_message(e),
                exc_info=True,
            )

        for aggregate in filtered_view:
            if aggregate.meta.field_name not in fields_set:
                continue
            if not aggregate.has_field("value"):
                continue

            results.append(
                {
                    "field": aggregate.meta.field_name,
                    "context_models": aggregate.meta.context_models,
                    "context_params": aggregate.meta.context_params,
                    "value": aggregate.get_field("value"),
                }
            )

        return results
