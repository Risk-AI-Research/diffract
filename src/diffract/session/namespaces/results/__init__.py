"""Results namespace for Session."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from diffract.core.constants import TABLE_AGGREGATES, TABLE_PARAMETERS
from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.export.formatters.json_formatter import dump_json
from diffract.core.export.formatters.long import long_result_rows
from diffract.core.export.formatters.registry import get_formatter, get_long_formatter
from diffract.core.export.interface import StructuredExportResult
from diffract.core.parallel import ParallelContext
from diffract.core.storage.catalog_manager import storage_activity
from diffract.core.utils.exceptions import format_exception_message
from diffract.session._identifiers import check_field_name
from diffract.session.errors import (
    FieldNotFoundError,
    KernelNotFoundError,
    SessionError,
)
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
from diffract.session.resolver import (
    FieldAddress,
    FieldSelector,
    aggregate_selector,
    render_label,
    resolve,
)
from diffract.session.session import Session, SessionContext
from diffract.session.summaries import EraseSummary
from diffract.session.utils import did_you_mean

if TYPE_CHECKING:
    from diffract.core.compute.execution.executor import KernelExecutor
    from diffract.core.compute.execution.traces import FieldStatus
    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.metadata import ParameterMetadata
    from diffract.core.export.exporters import ResultExporter
    from diffract.core.export.interface import AggregateData, ResultData
    from diffract.session.field_cache import SessionFieldCache

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
        results_exporter_factory: Callable[[], ResultExporter] = Provide[
            "export.result_exporter.provider"
        ],
        default_export_format: str | None = Provide[
            "export.config.default_export_format"
        ],
        kernel_executor_factory: Callable[[], KernelExecutor] = Provide[
            "compute.kernel_executor.provider"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository
        self.__kernel_registry = kernel_registry
        self.__parallel_context_factory = parallel_context_factory
        self.__results_exporter_factory = results_exporter_factory
        self.__default_export_format = default_export_format
        self.__kernel_executor_factory = kernel_executor_factory

    @property
    def generation(self) -> int:
        """Revision of the session's cached result metadata."""
        return self.__session_or_context._field_cache.generation

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

    def describe_run(
        self, run_id: str | None, *, parameter_uid: str | None = None
    ) -> dict[str, Any]:
        """Look up requested settings and per-entity arithmetic by exported run ID.

        Args:
            run_id: The run_id from a long result row.
            parameter_uid: Limit observations to one parameter of a batched run.

        Returns:
            Run metadata, the requested configuration and recorded arithmetic.
            The arithmetic status is "unknown" for runs without observations.

        Raises:
            SessionError: If the run is unknown or outside the selected scope.
        """
        with self.__session_or_context:
            journal = getattr(self.__param_repo.storage_manager, "journal", None)
            if journal is None:
                raise SessionError("The store does not provide a run journal.")
            try:
                record = journal.describe(run_id, parameter_uid=parameter_uid)
            except KeyError as exc:
                raise SessionError(f"Unknown run {run_id!r}.") from exc
            except ValueError as exc:
                raise SessionError(str(exc)) from exc
            facts = record["arithmetic"]
            if not isinstance(self.__session_or_context, Session):
                owners = {
                    ("parameter", uid) for uid in self._params_in_scope().list_uids()
                }
                if parameter_uid is None:
                    owners.update(
                        ("aggregate", uid) for uid in self._aggs_in_scope().list_uids()
                    )
                else:
                    owners.intersection_update({("parameter", parameter_uid)})
                facts = [
                    f for f in facts if (f["entity_kind"], f["owner_uid"]) in owners
                ]
                if not facts and (
                    record["arithmetic_status"] == "recorded"
                    or owners.isdisjoint(journal.live_owners(run_id))
                ):
                    raise SessionError(f"Run {run_id!r} is outside the selected scope.")
            return record | {"arithmetic": facts}

    def status(self, *fields: str) -> tuple[FieldStatus, ...]:
        """Report freshness against stored inputs and the effective configuration.

        Returns one entry per requested field and in-scope identity. An entry's
        status is fresh, stale, or never computed. Inherited values eligible for
        verification have a pre-provenance reason. This call reads metadata
        without executing kernels or updating stored values.

        Args:
            *fields: Kernel-produced fields to inspect.

        Raises:
            KernelNotFoundError: If no kernel produces a requested field.
            SessionError: If the store cannot supply a freshness report.
        """
        with self.__session_or_context:
            for field in fields:
                if not self.__kernel_registry.can_produce_field(field):
                    available = self.__kernel_registry.list_fields_can_produce()
                    hint = did_you_mean(field, available)
                    raise KernelNotFoundError(
                        f"Cannot inspect '{field}': no registered kernel produces it."
                        f"{hint} Use session.compute.list_available_metrics() to "
                        "see all fields."
                    )
            if not fields:
                return ()
            try:
                parameters = self._params_in_scope()
                with self.__kernel_executor_factory() as executor:
                    return tuple(
                        row
                        for field in dict.fromkeys(fields)
                        for row in executor.inspect(field, parameters)
                    )
            except SessionError:
                raise
            except Exception as exc:
                raise SessionError(
                    f"Failed to inspect result status: {format_exception_message(exc)}"
                ) from exc

    def _aggs_in_scope(self) -> AggregateView:
        """Aggregate view honoring the active session scope."""
        if isinstance(self.__session_or_context, Session):
            return self.__agg_repo.create_view()
        return self.__session_or_context._agg_filter_context

    def _export_field_cache(self) -> SessionFieldCache | None:
        storage = self.__param_repo.storage_manager
        if getattr(storage, "uses_value_revisions", False) is True:
            return None
        return self.__session_or_context._field_cache

    def _collect_metrics(
        self,
        fields: tuple[FieldAddress, ...],
        params: IParameterView,
        *,
        run_ids: dict[tuple[str, str], str] | None = None,
    ) -> ResultData:
        """Collect per-parameter values for the requested fields."""
        exporter: ResultExporter = self.__results_exporter_factory()
        selectors = tuple(resolve(f) for f in fields)
        literals = {f for f in fields if isinstance(f, str)}
        names = literals | {sel.field for sel in selectors}

        def accept(meta: ParameterMetadata, name: str) -> bool:
            if name in literals:
                return True
            candidate = FieldSelector(
                name, (meta.model_id,), (meta.name,), (meta.step,)
            )
            return any(sel.matches(candidate) for sel in selectors)

        return exporter._collect_parameter_results(
            *names,
            accept=accept if any(sel.is_contextual for sel in selectors) else None,
            parameters=params,
            step_labels=self.__param_repo.instances.step_labels(),
            parallel=self.__parallel_context_factory(),
            field_cache=self._export_field_cache(),
            run_ids=run_ids,
        )

    def _collect_aggregates(
        self,
        fields: tuple[FieldAddress, ...],
        aggs: AggregateView,
        *,
        include_members: bool = False,
        include_runs: bool = False,
    ) -> AggregateData:
        """Collect aggregate values for the requested fields."""
        exporter: ResultExporter = self.__results_exporter_factory()
        selectors = tuple(resolve(f) for f in fields)
        names = {sel.field for sel in selectors}

        def accept(record: dict[str, Any]) -> bool:
            candidate = aggregate_selector(record)
            return any(sel.matches(candidate) for sel in selectors)

        return exporter._collect_aggregate_results(
            *names,
            accept=accept if any(sel.is_contextual for sel in selectors) else None,
            step_labels=self.__param_repo.instances.step_labels(),
            aggregates=aggs,
            parallel=self.__parallel_context_factory(),
            include_members=include_members,
            include_runs=include_runs,
            symmetric_fields={
                field
                for field in names
                if self.__kernel_registry.can_produce_field(field)
                and self.__kernel_registry.get_kernel_symmetry(
                    self.__kernel_registry.get_kernel_producing_field(field)
                )
                is True
            },
        )

    def _check_export_fields(
        self,
        fields: tuple[FieldAddress, ...],
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
            field_cache=self._export_field_cache(),
            searched_metrics=searched_metrics,
            searched_aggregates=searched_aggregates,
        )

    def _format_metrics(
        self,
        param_results: ResultData,
        fields: tuple[FieldAddress, ...],
        export_format: str,
    ) -> Any:
        """Format collected per-parameter results in the requested format."""
        if export_format == "json":
            return dump_json(param_results)

        formatter = get_formatter(export_format)
        formatted = formatter.format_results(
            param_results, [], tuple(render_label(resolve(f)) for f in fields)
        )
        if not isinstance(formatted, StructuredExportResult):
            raise TypeError(
                f"Formatter {export_format!r} must return StructuredExportResult."
            )
        return formatted.scalars

    def _format_aggregates(
        self,
        aggregate_results: AggregateData,
        fields: tuple[FieldAddress, ...],
        export_format: str,
    ) -> Any:
        """Format collected aggregate results in the requested format."""
        if export_format == "json":
            return dump_json(aggregate_results)

        formatter = get_formatter(export_format)
        formatted = formatter.format_results(
            {}, aggregate_results, tuple(render_label(resolve(f)) for f in fields)
        )
        if not isinstance(formatted, StructuredExportResult):
            raise TypeError(
                f"Formatter {export_format!r} must return StructuredExportResult."
            )
        return formatted.aggregates

    def export_metrics(
        self,
        *fields: FieldAddress,
        export_format: str | None = None,
    ) -> Any:
        """Retrieve computation results for specified fields (metrics only).

        Serves ``PARAMETER``-level fields; ``IN_MODEL`` and ``CROSS_MODEL``
        fields are retrieved with ``export_aggregates``.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.

        Args:
            *fields: Field names, structured selectors, or references to either.
            export_format: Output format - "dict", "json", "pandas", "polars", "list".
                Defaults to the configured ``[export] default_export_format``.

        Returns:
            Metric results in the specified format. Edits to exported arrays
            affect only the exported data.

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
                found=found_metric_fields(param_results, fields),
                params=params,
                searched_metrics=True,
                searched_aggregates=False,
            )
            return self._format_metrics(param_results, fields, export_format)

    def export_aggregates(
        self,
        *fields: FieldAddress,
        export_format: str | None = None,
    ) -> Any:
        """Retrieve aggregation results for specified fields (aggregates only).

        Serves ``IN_MODEL`` and ``CROSS_MODEL``-level fields; ``PARAMETER``
        fields are retrieved with ``export_metrics``.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.

        Args:
            *fields: Field names, structured selectors, or references to either.
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
                found=found_aggregate_fields(aggregate_results, fields),
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

            ingester = FieldIngester()

            try:
                with storage_activity(self.__param_repo.storage_manager, "ingest"):
                    ingester.ingest(
                        fields_by_uid=fields_by_uid,
                        parameters=params,
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

        Model and parameter names select existing members in the current scope.
        The field and selected parameter identities identify each stored value.

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
            SessionError: If an entry is malformed, a named member is missing
                from the scope, or a value already exists and force is false.
        """
        with self.__session_or_context:
            if not aggregates:
                return

            ingester = AggregateIngester()

            try:
                with storage_activity(self.__agg_repo.storage_manager, "ingest"):
                    ingester.ingest(
                        aggregates=aggregates,
                        parameter_uids=(
                            None
                            if isinstance(self.__session_or_context, Session)
                            else self._params_in_scope().list_uids()
                        ),
                        repository=self.__agg_repo,
                        force=force,
                    )
            except AggregateIngestionError as e:
                raise SessionError(str(e)) from e

    def erase(
        self,
        *fields_to_erase: str,
        erase_dependent_also: bool = False,
        erase_all: bool = False,
    ) -> EraseSummary:
        """Erase stored fields and the siblings from their producing kernel run.

        Removes computed and ingested values while preserving parameters.
        Kernel outputs from the same execution group are erased together.

        Args:
            *fields_to_erase: Field names to erase. Required if erase_all is False.
            erase_dependent_also: If True, also erases fields that depend on
                the specified fields.
            erase_all: If True, erases all stored fields in scope. fields_to_erase
                must be empty.

        Returns:
            An EraseSummary naming the erased fields and the number of
            parameter entries that held at least one of them.

        Raises:
            ValueError: If neither fields_to_erase nor erase_all is specified,
                or if both are specified.
            FieldNotFoundError: If a specified field is neither stored nor
                produced by a registered kernel.
        """
        with self.__session_or_context:
            if (not fields_to_erase) ^ erase_all:
                if not fields_to_erase:
                    msg = "No fields_to_erase provided and erase_all=False"
                else:
                    msg = "Cannot specify both fields_to_erase and erase_all=True"
                raise ValueError(msg)

            index = self.__param_repo.metadata_index
            with index:
                fields = set()
                if not erase_all:
                    known = set(
                        self.__param_repo.storage_manager.list_fields(
                            table=TABLE_PARAMETERS
                        )
                    )
                    known.update(
                        row["field_name"]
                        for row in index.query_rows(
                            TABLE_AGGREGATES, columns=("field_name",), distinct=True
                        )
                    )
                    eraser = ResultsEraser(kernel_registry=self.__kernel_registry)
                    try:
                        fields = eraser.resolve_fields_to_erase(
                            fields_to_erase, erase_dependent_also, stored_fields=known
                        )
                    except ResultsEraserError as exc:
                        field = str(exc)
                        candidates = known.union(
                            self.__kernel_registry.list_fields_can_produce()
                        )
                        raise FieldNotFoundError(
                            f"Unknown field {field!r}."
                            + did_you_mean(field, candidates)
                        ) from exc

                params = self._params_in_scope().list_uids()
                aggregates = self._aggs_in_scope().list_uids()
                if not erase_all:
                    aggregates = index.query(
                        TABLE_AGGREGATES,
                        where_in={"uid": aggregates, "field_name": sorted(fields)},
                    )
                removed = self.__param_repo.erase_result_groups(
                    params, None if erase_all else tuple(fields)
                )
                removed_aggregates = self.__agg_repo.erase_result_groups(aggregates)
                aggregate_uids = set(aggregates)
                aggregate_uids.update(uid for uid, _ in removed_aggregates)
                if aggregate_uids:
                    ids = tuple(aggregate_uids)
                    fields.update(
                        row["field_name"]
                        for row in index.query_rows(
                            TABLE_AGGREGATES,
                            columns=("field_name",),
                            where_in={"uid": ids},
                        )
                    )
                    self.__agg_repo.remove_membership(ids)

            erased_by_owner: dict[str, list[str]] = defaultdict(list)
            for uid, field in removed:
                erased_by_owner[uid].append(field)
                fields.add(field)
            for uid, erased in erased_by_owner.items():
                self.__session_or_context._field_cache.remove_fields_by_uids(
                    (uid,), erased
                )
            return EraseSummary(
                fields=tuple(sorted(fields)), affected_uids=len(erased_by_owner)
            )

    def export(
        self,
        *fields: FieldAddress,
        sources: str = "all",
        export_format: str | None = None,
        expand_contextual: bool = True,
        shape: str = "wide",
    ) -> Any:
        """Unified export of metrics and aggregates.

        This method provides a single interface for exporting both parameter
        metrics and aggregate values. Wide export can merge aggregate values
        into parameter entries as contextual fields. Long export emits one
        row per stored parameter field or aggregate member with join keys,
        run IDs when available, and separate scalar and array value columns.

        A field name that neither a registered kernel produces nor any stored
        field matches raises; a known field with no values in the current
        scope is reported with a warning instead of being silently omitted.
        With sources="all", a field found in either source is served without
        complaint.

        Args:
            *fields: Field names, structured selectors, or references to either.
            sources: Data sources to include - "metrics", "aggregates", or "all".
            export_format: Output format - "dict", "json", "pandas", "polars", "list".
                Defaults to the configured ``[export] default_export_format``.
            expand_contextual: When True and sources="all", merge aggregate values
                into parameter entries as contextual field names in wide export.
            shape: "wide" for parameter entries or "long" for tidy value rows.
                Long dictionary export maps column names to lists.

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
        if shape == "long":
            return self._export_long(
                fields, sources=sources, export_format=export_format
            )
        if shape != "wide":
            raise ValueError(f"Invalid shape: {shape!r}. Use 'wide' or 'long'.")

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
            aggregates = self._collect_aggregates(fields, aggs, include_members=True)
            self._check_export_fields(
                fields,
                found=found_metric_fields(metrics, fields)
                | found_aggregate_fields(aggregates, fields),
                params=params,
                aggs=aggs,
                searched_metrics=True,
                searched_aggregates=True,
            )

            if expand_contextual and aggregates:
                self._merge_aggregates_into_metrics(metrics, aggregates)

            return self._format_metrics(metrics, fields, export_format)

    def _export_long(
        self,
        fields: tuple[FieldAddress, ...],
        *,
        sources: str,
        export_format: str,
    ) -> Any:
        if not fields:
            raise ValueError("At least one field must be specified")
        format_rows = get_long_formatter(export_format)

        with self.__session_or_context:
            params = self._params_in_scope() if sources != "aggregates" else None
            aggs = self._aggs_in_scope() if sources != "metrics" else None
            run_ids: dict[tuple[str, str], str] = {}
            metrics = (
                self._collect_metrics(fields, params, run_ids=run_ids)
                if params is not None
                else {}
            )
            aggregates = (
                self._collect_aggregates(
                    fields, aggs, include_members=True, include_runs=True
                )
                if aggs is not None
                else []
            )
            self._check_export_fields(
                fields,
                found=found_metric_fields(metrics, fields)
                | found_aggregate_fields(aggregates, fields),
                params=params,
                aggs=aggs,
                searched_metrics=params is not None,
                searched_aggregates=aggs is not None,
            )
            return format_rows(long_result_rows(metrics, aggregates, run_ids))

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

        field = agg.get("field")
        value = agg.get("value")
        if not isinstance(field, str) or not context_params or value is None:
            return

        selector = aggregate_selector(agg)
        members = tuple(
            (m["model_id"], m["step"], m["name"], m["step_label"])
            for m in agg.get("members", ())
        )
        contextual_name = render_label(selector, members=members)
        context = {
            "symmetric": agg.get("symmetric", False),
            "field": selector.field,
            "models": selector.models,
            "params": selector.params,
            "roles": selector.roles,
            "steps": selector.steps,
        }

        for uid in agg["member_uids"]:
            if uid in metrics:
                fields = metrics[uid].setdefault("fields", {})
                if contextual_name in fields:
                    raise SessionError(
                        f"Wide export has multiple values for {contextual_name!r} on "
                        f"parameter {uid!r}. Use shape='long' to keep both."
                    )
                fields[contextual_name] = value
                metrics[uid].setdefault("field_contexts", {})[contextual_name] = context
