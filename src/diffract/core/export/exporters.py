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

from diffract.core.data.nn.aggregates.identity import PAIR_SIZE
from diffract.core.parallel import ParallelContext
from diffract.core.utils.exceptions import format_exception_message

from .interface import AggregateData, IResultExporter, IResultFormatter, ResultData

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.metadata import ParameterMetadata
    from diffract.session.field_cache import SessionFieldCache

logger = logging.getLogger(__name__)


class SessionExportError(Exception):
    """Base exception for session export errors."""


class ResultExporter(IResultExporter):
    """Main implementation of result export functionality."""

    def export_results(
        self,
        *fields: str,
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
        *fields: str,
        parameters: IParameterView,
        parallel: ParallelContext | None,
        field_cache: SessionFieldCache | None = None,
        step_labels: dict[tuple[str, int], str] | None = None,
        accept: Callable[[ParameterMetadata, str], bool] | None = None,
        run_ids: dict[tuple[str, str], str] | None = None,
    ) -> ResultData:
        """Collect results from parameter repository.

        Args:
            *fields: Exact storage field names.
            accept: Optional predicate over parameter metadata and a stored field.
            step_labels: Display text keyed by instance and checkpoint.
            run_ids: Populate this mapping from the pointers used for value reads.
            parameters: Parameter collection to iterate over.
            parallel: Optional per-method parallel context for collection operations.
            field_cache: Optional session-level cache for field availability.
                If provided and valid, skips expensive list_fields_by_uid() call.

        Returns:
            Nested mapping with metadata and field values per parameter.
        """
        results: ResultData = {}
        read_catalog = getattr(type(parameters), "read_catalog_fields", None)
        values = (
            read_catalog(
                parameters,
                fields,
                accept=accept,
                **({"run_ids": run_ids} if run_ids is not None else {}),
            )
            if read_catalog
            else None
        )
        if values is None:
            to_receive = self._prepare_parameter_fields(
                parameters, fields, parallel, field_cache, accept
            )

        labels = step_labels or {}
        for param in sorted(
            parameters, key=lambda p: (p.meta.model_id, p.meta.step, p.meta.name)
        ):
            if values is None:
                param_results = {
                    field: param.get_field(field, auto_prefetch=False)
                    for field in to_receive[param.meta.uid]
                }
            else:
                param_results = values[param.meta.uid]

            results[param.meta.uid] = {
                "metadata": {
                    **param.meta.other_meta,
                    "name": param.meta.name,
                    "model_id": param.meta.model_id,
                    "parameter_type": param.meta.ptype.name,
                    "instance_uid": param.meta.instance_uid,
                    "step": param.meta.step,
                    "step_label": labels.get(
                        (param.meta.instance_uid, param.meta.step)
                    ),
                },
                "fields": param_results,
            }

        return results

    @staticmethod
    def _prepare_parameter_fields(
        parameters: IParameterView,
        fields: tuple[str, ...],
        parallel: ParallelContext | None,
        field_cache: SessionFieldCache | None,
        accept: Callable[[ParameterMetadata, str], bool] | None,
    ) -> dict[str, list[str]]:
        fields_set = set(fields)

        cached = field_cache.get() if field_cache is not None else None
        if cached is not None and field_cache is not None:
            required_uids = parameters.list_uids()
            fields_by_uid = {uid: cached[uid] for uid in required_uids if uid in cached}
            missing_uids = [uid for uid in required_uids if uid not in cached]
            if missing_uids:
                received = parameters[missing_uids].list_fields_by_uid(
                    parallel=parallel
                )
                field_cache.update(received)
                fields_by_uid.update(received)
        else:
            fields_by_uid = parameters.list_fields_by_uid(parallel=parallel)
            if field_cache is not None:
                field_cache.set(fields_by_uid)

        to_receive: dict[str, list[str]] = {}
        for uid, fields_available in fields_by_uid.items():
            to_receive[uid] = [f for f in fields_available if f in fields_set]

        if accept is not None:
            for param in parameters:
                uid = param.meta.uid
                to_receive[uid] = [f for f in to_receive[uid] if accept(param.meta, f)]
        try:
            parameters.prefetch_fields(fields_by_uid=to_receive, parallel=parallel)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "Prefetch failed during export: %s",
                format_exception_message(e),
                exc_info=True,
            )

        return to_receive

    def _collect_aggregate_results(
        self,
        *fields: str,
        aggregates: AggregateView | None,
        parallel: ParallelContext | None,
        include_members: bool = False,
        symmetric_fields: set[str] | None = None,
        step_labels: dict[tuple[str, int], str] | None = None,
        accept: Callable[[dict[str, Any]], bool] | None = None,
        include_runs: bool = False,
    ) -> AggregateData:
        """Collect results from aggregate repository.

        Args:
            accept: Optional predicate over a record before fetching the value.
            step_labels: Display text keyed by instance and checkpoint.
            include_members: Include member UIDs for an internal parameter join.
            include_runs: Include the run ID from the pointer used to read the value.
            symmetric_fields: Pair fields to emit in both operand orders.
            *fields: Base field names to collect.
            aggregates: Aggregate view to iterate over.
            parallel: Optional per-method parallel context (unused currently).

        Returns:
            List of aggregate records with field name, context, and value.
        """
        if aggregates is None:
            return []

        labels = step_labels or {}
        records: list[tuple[str, dict[str, Any]]] = []
        filtered = aggregates.filter_by_field_name(*fields)
        for aggregate in filtered:
            meta = aggregate.meta
            members = [
                {
                    "model_id": name,
                    "step": step,
                    "name": param,
                    "instance_uid": instance,
                    "step_label": labels.get((instance, step)),
                }
                for (name, step), param, instance in zip(
                    meta.member_coordinates,
                    meta.member_names,
                    meta.member_instances,
                    strict=True,
                )
            ]
            record = {
                "field": meta.field_name,
                "context_models": meta.context_models,
                "context_params": meta.context_params,
                "members": members,
            }
            if include_members:
                record["member_uids"] = meta.member_uids
            if meta.direction_unknown:
                record["direction_unknown"] = True
            mirror = not meta.direction_unknown and meta.field_name in (
                symmetric_fields or ()
            )
            if len(members) == PAIR_SIZE and (meta.directed or mirror):
                source, target = members
                for first, second in (
                    ((source, target), (target, source))
                    if mirror
                    else ((source, target),)
                ):
                    pair = record | {
                        "symmetric": bool(mirror),
                        "members": [first, second],
                        "model_source": first["model_id"],
                        "step_source": first["step"],
                        "model_target": second["model_id"],
                        "step_target": second["step"],
                    }
                    if include_members and first is target:
                        pair["member_uids"] = meta.member_uids[::-1]
                    if accept is None or accept(pair):
                        records.append((meta.uid, pair))
            elif accept is None or accept(record):
                records.append((meta.uid, record))

        if not records:
            return []
        uids = list(dict.fromkeys(uid for uid, _ in records))
        selected = filtered[uids]
        values = selected.read_catalog_values() if include_runs else None
        if values is None:
            try:
                selected.prefetch_fields(fields=["value"], parallel=parallel)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Aggregate prefetch failed: %s",
                    format_exception_message(e),
                    exc_info=True,
                )
            values = {
                agg.meta.uid: (agg.get_field("value"), None)
                for agg in selected
                if agg.has_field("value")
            }

        def order(item: tuple[str, dict[str, Any]]) -> tuple:
            uid, row = item
            coordinates = [
                (m["model_id"], m["step"], m["name"]) for m in row["members"]
            ]
            if "model_source" not in row:
                coordinates.sort()
            return row["field"], coordinates, uid

        return [
            row
            | {"aggregate_uid": uid, "value": values[uid][0]}
            | ({"run_id": values[uid][1]} if include_runs else {})
            for uid, row in sorted(records, key=order)
            if uid in values
        ]
