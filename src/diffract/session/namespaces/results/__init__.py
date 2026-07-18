"""Results namespace for Session."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.export.formatters.registry import get_formatter
from diffract.core.parallel import ParallelContext
from diffract.session._identifiers import check_field_name, check_identifier
from diffract.session.errors import KernelNotFoundError, SessionError
from diffract.session.namespaces.results.eraser import ResultsEraser, ResultsEraserError
from diffract.session.namespaces.results.injester import (
    AggregateIngester,
    AggregateIngestionError,
    FieldIngester,
    FieldIngestionError,
)
from diffract.session.namespaces.results.validation import (
    check_export_fields,
    found_aggregate_fields,
    found_metric_fields,
)
from diffract.session.resolver import FieldSelector, render_label
from diffract.session.session import Session, SessionContext
from diffract.session.summaries import EraseSummary

if TYPE_CHECKING:
    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.export.exporters import ResultExporter
    from diffract.core.export.interface import AggregateData, ResultData

logger = logging.getLogger(__name__)

# Used when a configuration carries no [export] default_export_format.
FALLBACK_EXPORT_FORMAT = "dict"


class ResultsNamespace:
    """Results export and ingestion API for Session."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        aggregate_repository: AggregateRepository = Provide["nn.aggregate_repository"],
        kernel_registry: KernelRegistry = Provide["compute.kernel_registry"],
        parallel_context_factory: Callable[[], ParallelContext] = Provide[
            "parallel_singleton.thread_pool_context.provider"
        ],
        results_exporter_factory: Callable[
            [tuple[str, ...], IParameterView, str], Any
        ] = Provide["export.result_exporter.provider"],
        default_export_format: str | None = Provide[
            "export.config.default_export_format"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository
        self.__kernel_registry = kernel_registry
        self.__parallel_context_factory = parallel_context_factory
        self.__results_exporter_factory = results_exporter_factory
        self.__default_export_format = default_export_format

    def _resolve_export_format(self, export_format: str | None) -> str:
        """Resolve the export format for a call.

        An explicit argument wins over the configured default.

        Args:
            export_format: Format named by the caller, or None to take the
                configured default.

        Returns:
            The format to export in. A configuration carrying no
            ``[export] default_export_format`` yields FALLBACK_EXPORT_FORMAT.
        """
        if export_format is not None:
            return export_format
        if self.__default_export_format is None:
            return FALLBACK_EXPORT_FORMAT
        return self.__default_export_format

    def _params_in_scope(self) -> IParameterView:
        """Parameter view honoring the active session scope."""
        if isinstance(self.__session_or_context, Session):
            return self.__param_repo.create_view()
        return self.__session_or_context._param_filter_context

    def _aggs_in_scope(self) -> AggregateView:
        """Aggregate view honoring the active session scope."""
        if isinstance(self.__session_or_context, Session):
            return self.__agg_repo.create_view()
        return self.__session_or_context._agg_filter_context

    def _collect_metrics(
        self, fields: tuple[str, ...], params: IParameterView
    ) -> ResultData:
        """Collect per-parameter values for the requested fields."""
        exporter: ResultExporter = self.__results_exporter_factory()
        return exporter._collect_parameter_results(
            *fields,
            parameters=params,
            parallel=self.__parallel_context_factory(),
            field_cache=self.__session_or_context._field_cache,
        )

    def _collect_aggregates(
        self, fields: tuple[str, ...], aggs: AggregateView
    ) -> AggregateData:
        """Collect aggregate values for the requested fields."""
        exporter: ResultExporter = self.__results_exporter_factory()
        return exporter._collect_aggregate_results(
            *fields,
            aggregates=aggs,
            parallel=self.__parallel_context_factory(),
        )

    def _check_export_fields(
        self,
        fields: tuple[str, ...],
        *,
        found: set[str],
        params: IParameterView | None = None,
        aggs: AggregateView | None = None,
        searched_metrics: bool,
        searched_aggregates: bool,
    ) -> None:
        """Reject misspelled fields; warn for known fields with no values.

        Views already built for collection are passed through; a side not
        collected from is provided lazily, so the happy path builds nothing.
        """
        check_export_fields(
            fields,
            found=found,
            registry=self.__kernel_registry,
            params=(lambda: params) if params is not None else self._params_in_scope,
            aggregates=(lambda: aggs) if aggs is not None else self._aggs_in_scope,
            field_cache=self.__session_or_context._field_cache,
            searched_metrics=searched_metrics,
            searched_aggregates=searched_aggregates,
        )

    def _format_metrics(
        self,
        param_results: ResultData,
        fields: tuple[str, ...],
        export_format: str,
    ) -> Any:
        """Format collected per-parameter results in the requested format."""
        if export_format == "json":
            return self._dump_json(param_results)

        formatter = get_formatter(export_format)
        formatted = formatter.format_results(param_results, [], fields)
        return formatted.scalars

    def _format_aggregates(
        self,
        aggregate_results: AggregateData,
        fields: tuple[str, ...],
        export_format: str,
    ) -> Any:
        """Format collected aggregate results in the requested format."""
        if export_format == "json":
            return self._dump_json(aggregate_results)

        formatter = get_formatter(export_format)
        formatted = formatter.format_results({}, aggregate_results, fields)
        return formatted.aggregates

    def export_metrics(
        self,
        *fields: str,
        export_format: str | None = None,
    ) -> Any:
        """Retrieve computation results for specified fields (metrics only).

        Serves ``PARAMETER``-level fields; ``IN_MODEL`` and ``CROSS_MODEL``
        fields are retrieved with ``export_aggregates``.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.

        Args:
            *fields: Field names to retrieve.
            export_format: Output format - "dict", "json", "pandas", "polars", "list".
                Defaults to the configured ``[export] default_export_format``.

        Returns:
            Metric results in the specified format.

        Raises:
            KernelNotFoundError: If a requested field is neither producible
                by a registered kernel nor stored in the current scope.
        """
        export_format = self._resolve_export_format(export_format)
        with self.__session_or_context:
            if not fields:
                raise ValueError("At least one field must be specified")

            params = self._params_in_scope()
            param_results = self._collect_metrics(fields, params)
            self._check_export_fields(
                fields,
                found=found_metric_fields(param_results),
                params=params,
                searched_metrics=True,
                searched_aggregates=False,
            )
            return self._format_metrics(param_results, fields, export_format)

    def export_aggregates(
        self,
        *fields: str,
        export_format: str | None = None,
    ) -> Any:
        """Retrieve aggregation results for specified fields (aggregates only).

        Serves ``IN_MODEL`` and ``CROSS_MODEL``-level fields; ``PARAMETER``
        fields are retrieved with ``export_metrics``.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.

        Args:
            *fields: Field names to retrieve.
            export_format: Output format - "dict", "json", "pandas", "polars", "list".
                Defaults to the configured ``[export] default_export_format``.

        Returns:
            Aggregation results in the specified format.

        Raises:
            KernelNotFoundError: If a requested field is neither producible
                by a registered kernel nor stored in the current scope.
        """
        export_format = self._resolve_export_format(export_format)
        with self.__session_or_context:
            if not fields:
                raise ValueError("At least one field must be specified")

            aggs = self._aggs_in_scope()
            aggregate_results = self._collect_aggregates(fields, aggs)
            self._check_export_fields(
                fields,
                found=found_aggregate_fields(aggregate_results),
                aggs=aggs,
                searched_metrics=False,
                searched_aggregates=True,
            )
            return self._format_aggregates(aggregate_results, fields, export_format)

    def ingest_metrics(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        """Ingest precomputed fields into the session via a uid->field mapping.

        Field names are stored exactly as provided (including contextual suffixes
        like "metric@ctx"). By default, this method is strict and raises if any
        target field already exists; set force=True to overwrite.

        Args:
            fields_by_uid: Mapping of parameter uid -> {field_name: value}.
            force: If False (default), raise on any existing field conflict.
                If True, overwrite existing field values.

        Raises:
            InvalidIdentifierError: If a field name carries storage-unsafe
                characters.
            SessionError: If unknown uids are provided or conflicts are
                detected (force=False).
        """
        with self.__session_or_context:
            if not fields_by_uid:
                return

            for field_map in fields_by_uid.values():
                for field_name in field_map:
                    check_field_name(field_name)

            params = self.__param_repo.create_view()
            filtered_params = params[fields_by_uid.keys()]

            ingester = FieldIngester()

            try:
                ingester.ingest(
                    fields_by_uid=fields_by_uid,
                    parameters=filtered_params,
                    force=force,
                )
            except FieldIngestionError as e:
                raise SessionError(str(e)) from e

            all_fields: set[str] = set()
            for field_dict in fields_by_uid.values():
                all_fields.update(field_dict.keys())

            self.__session_or_context._field_cache.add_computed_fields(
                affected_uids=fields_by_uid.keys(),
                new_fields=all_fields,
            )

    def ingest_aggregates(
        self,
        aggregates: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        """Ingest precomputed aggregate values into the session.

        Each aggregate is identified by (field_name, context_models, context_params).
        Useful for importing precomputed aggregation results.

        Args:
            aggregates: List of aggregate dictionaries, each containing:
                - field_name: Base field name (e.g., "l_overlap").
                - context_models: Tuple/list of model IDs.
                - context_params: Tuple/list of parameter names (optional).
                - value: The computed value to store.
            force: If False (default), raise on existing aggregate conflict.
                If True, overwrite existing values.

        Raises:
            InvalidIdentifierError: If a field name or context member is not
                over the accepted identifier alphabet.
            SessionError: If aggregate structure is invalid or conflicts are
                detected (force=False).
        """
        with self.__session_or_context:
            if not aggregates:
                return

            self._validate_aggregate_identifiers(aggregates)

            ingester = AggregateIngester()

            try:
                ingester.ingest(
                    aggregates=aggregates,
                    repository=self.__agg_repo,
                    force=force,
                )
            except AggregateIngestionError as e:
                raise SessionError(str(e)) from e

    @staticmethod
    def _validate_aggregate_identifiers(aggregates: list[dict[str, Any]]) -> None:
        """Reject exotic ids in each aggregate's field name and context members.

        Missing keys are left for the ingester's structure check.
        """
        for agg in aggregates:
            field_name = agg.get("field_name")
            if isinstance(field_name, str):
                check_identifier(field_name, kind="aggregate field name")
            for model_id in agg.get("context_models", ()) or ():
                check_identifier(model_id, kind="model id")
            for param_name in agg.get("context_params", ()) or ():
                check_identifier(param_name, kind="parameter name")

    def erase(
        self,
        *fields_to_erase: str,
        erase_dependent_also: bool = False,
        erase_all: bool = False,
    ) -> EraseSummary:
        """Erase computation results for specified fields.

        Removes computed field data from parameters while preserving the
        parameters themselves. Optionally erases dependent fields that
        rely on the specified fields as inputs.

        Args:
            *fields_to_erase: Field names to erase. Required if erase_all is False.
            erase_dependent_also: If True, also erases fields that depend on
                the specified fields.
            erase_all: If True, erases all computed fields. fields_to_erase
                must be empty.

        Returns:
            An EraseSummary naming the erased fields and the number of
            parameter entries that held at least one of them.

        Raises:
            ValueError: If neither fields_to_erase nor erase_all is specified,
                or if both are specified.
            KernelNotFoundError: If any specified field cannot be produced
                by registered kernels.
        """
        with self.__session_or_context:
            if (not fields_to_erase) ^ erase_all:
                if not fields_to_erase:
                    msg = "No fields_to_erase provided and erase_all=False"
                else:
                    msg = "Cannot specify both fields_to_erase and erase_all=True"
                raise ValueError(msg)

            eraser = ResultsEraser(kernel_registry=self.__kernel_registry)

            try:
                if erase_all:
                    fields = set(self.__kernel_registry.list_fields_can_produce())
                else:
                    fields = eraser.resolve_fields_to_erase(
                        fields_to_erase, erase_dependent_also
                    )
            except ResultsEraserError as e:
                raise KernelNotFoundError(str(e)) from e

            params = self._params_in_scope()
            aggs = self._aggs_in_scope()

            scope_uids = params.list_uids()
            fields_by_uid = params.list_fields_by_uid()
            erased_count = sum(
                1
                for uid in scope_uids
                if not fields.isdisjoint(fields_by_uid.get(uid, ()))
            )
            eraser.erase(parameters=params, aggregates=aggs, fields=fields)

            self.__session_or_context._field_cache.remove_fields_by_uids(
                scope_uids, fields
            )

            return EraseSummary(
                fields=tuple(sorted(fields)), affected_uids=erased_count
            )

    def export(
        self,
        *fields: str,
        sources: str = "all",
        export_format: str | None = None,
        expand_contextual: bool = True,
    ) -> Any:
        """Unified export of metrics and aggregates.

        This method provides a single interface for exporting both parameter
        metrics and aggregate values. When both sources are requested,
        aggregate values are merged into parameter entries as contextual fields.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.
        With sources="all", a field found in either source is served without
        complaint.

        Args:
            *fields: Field names to retrieve.
            sources: Data sources to include - "metrics", "aggregates", or "all".
            export_format: Output format - "dict", "json", "pandas", "polars", "list".
                Defaults to the configured ``[export] default_export_format``.
            expand_contextual: When True and sources="all", merge aggregate values
                into parameter entries as contextual field names.

        Returns:
            Results in the specified format.

        Raises:
            KernelNotFoundError: If a requested field is neither producible
                by a registered kernel nor stored in the current scope.
        """
        export_format = self._resolve_export_format(export_format)
        if sources not in ("metrics", "aggregates", "all"):
            raise ValueError(
                f"Invalid sources: {sources}. Use 'metrics', 'aggregates', or 'all'."
            )

        if sources == "metrics":
            return self.export_metrics(*fields, export_format=export_format)

        if sources == "aggregates":
            return self.export_aggregates(*fields, export_format=export_format)

        with self.__session_or_context:
            if not fields:
                raise ValueError("At least one field must be specified")

            params = self._params_in_scope()
            aggs = self._aggs_in_scope()
            metrics = self._collect_metrics(fields, params)
            aggregates = self._collect_aggregates(fields, aggs)
            self._check_export_fields(
                fields,
                found=found_metric_fields(metrics) | found_aggregate_fields(aggregates),
                params=params,
                aggs=aggs,
                searched_metrics=True,
                searched_aggregates=True,
            )

            if expand_contextual and aggregates:
                self._merge_aggregates_into_metrics(metrics, aggregates)

            return self._format_metrics(metrics, fields, export_format)

    def _merge_aggregates_into_metrics(
        self,
        metrics: dict[str, Any],
        aggregates: list[dict[str, Any]],
    ) -> None:
        """Merge aggregate values into parameter entries as contextual fields."""
        for agg in aggregates:
            self._merge_single_aggregate(metrics, agg)

    @staticmethod
    def _merge_single_aggregate(
        metrics: dict[str, Any],
        agg: dict[str, Any],
    ) -> None:
        """Merge a single aggregate into matching parameter entries."""
        context_params = tuple(agg.get("context_params", ()) or ())
        context_models = tuple(agg.get("context_models", ()) or ())

        field = agg.get("field")
        value = agg.get("value")
        if not isinstance(field, str) or not context_params or value is None:
            return

        contextual_name = render_label(
            FieldSelector(
                field=field,
                models=context_models or None,
                params=context_params or None,
            )
        )

        for uid, entry in metrics.items():
            meta = entry.get("metadata", {})
            param_name = meta.get("name")
            model_id = meta.get("model_id")

            if not param_name or not model_id:
                continue

            if param_name in context_params and model_id in context_models:
                metrics[uid].setdefault("fields", {})[contextual_name] = value

    @staticmethod
    def _dump_json(data: Any) -> str:
        def _json_serializer(obj: object) -> object:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

        return json.dumps(data, indent=2, default=_json_serializer)
