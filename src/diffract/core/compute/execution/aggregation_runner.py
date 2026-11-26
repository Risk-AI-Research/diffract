"""Aggregation-level kernel execution logic."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING, Any

from diffract.core.compute.config import KernelResult
from diffract.core.constants import TABLE_AGGREGATES
from diffract.core.data.metadata.runs import RunJournal
from diffract.core.data.nn.aggregates.identity import PAIR_SIZE, aggregate_identity

from ._types import (
    MAX_RESIDENT_ARGUMENTS as _MAX_RESIDENT_ARGUMENTS,
    AggregateInput,
    PendingGroup,
    RequiredInputs,
)
from .aggregation import AggregationContext, aggregate_groups
from .enums import KernelApplyLevel, KernelExecutionProtocol
from .restrictions import apply_restrictions_filter
from .strategy import create_execution_strategy
from .traces import StepDecision, comparison_for_scope, decide_step, describe_step

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import ProcessPoolExecutor

    from diffract.core.compute.config import KernelConfigIdentity, OutputComparison
    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.metadata.runs import Run, ValueTrace
    from diffract.core.data.nn.aggregates import AggregateRepository
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.parallel import ParallelContext
    from diffract.core.storage.catalog import ValuePointer
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    from .aggregation import CoordinatePairs
    from .traces import FieldStatus

logger = logging.getLogger(__name__)

type OutputKeys = tuple[tuple[str, str], ...]


class AggregationKernelRunner:
    """Executes aggregation-level kernels (IN_MODEL and CROSS_MODEL).

    Handles execution of kernels that aggregate across parameters within
    a model or across models, with support for memory-aware batching.
    """

    def __init__(
        self,
        registry: KernelRegistry,
        process_pool: ProcessPoolExecutor | None,
        parallel: ParallelContext | None,
        aggregate_repository: AggregateRepository | None,
    ) -> None:
        """Initialize the aggregation kernel runner.

        Args:
            registry: Kernel registry for metadata lookup.
            process_pool: Optional process pool for parallel execution.
            parallel: Optional parallel context for view operations.
            aggregate_repository: Repository for storing aggregation results.
        """
        self._registry = registry
        self._process_pool = process_pool
        self._parallel = parallel
        self._aggregate_repository = aggregate_repository
        self._group_scope: IParameterView | None = None
        self._group_pairs: CoordinatePairs | None = None
        self._groups: dict[
            KernelApplyLevel, list[tuple[AggregationContext, IParameterView]]
        ] = {}
        self._aggregate_uids: dict[tuple[str, tuple[str, ...], bool], str] = {}

    def clear_execution_state(self) -> None:
        """Release the grouping snapshot at the end of an apply call."""
        self._group_scope = None
        self._group_pairs = None
        self._groups.clear()
        self._aggregate_uids.clear()

    def _aggregate_uid(
        self, field: str, members: tuple[str, ...], directed: bool
    ) -> str:
        key = field, members, directed
        uid = self._aggregate_uids.get(key)
        if uid is None:
            uid = aggregate_identity(field, members, directed=directed).uid
            self._aggregate_uids[key] = uid
        return uid

    def run(
        self,
        kernel_name: str,
        parameters: IParameterView,
        *,
        force: bool = False,
        pairs: CoordinatePairs | None = None,
    ) -> set[str]:
        """Execute kernel with aggregation across models or parameters.

        Args:
            kernel_name: Name of the registered kernel.
            parameters: Parameter collection to process.
            force: Execute the kernel even for groups with fresh outputs.
            pairs: Ordered cross-model coordinates selected by the caller.

        Returns:
            The set of field names actually written to the aggregate
            repository (empty when every group was already computed, dropped by
            a restriction, or there is no repository configured).
        """
        storage = getattr(self._aggregate_repository, "storage_manager", None)
        journal = getattr(storage, "journal", None)
        if isinstance(journal, RunJournal):
            config, implementation = self._registry.capture_kernel_execution(
                kernel_name
            )
            fields = self._registry.get_fields_kernel_produce(kernel_name)
            if not force and storage.has_aggregate_records(fields):
                candidates = self._trace_candidates(
                    kernel_name, parameters, pairs=pairs
                )
                pending_groups, inherited, _ = self._partition_traced_groups(
                    storage, config.config_hash, candidates
                )
                if inherited:
                    self._adopt_groups(storage, journal, config, inherited)
            else:
                pending_groups = self._collect_pending_groups(
                    kernel_name, parameters, check_presence=False, pairs=pairs
                )
        else:
            pending_groups = self._collect_pending_groups(
                kernel_name, parameters, check_presence=not force, pairs=pairs
            )
        if not pending_groups:
            logger.debug(
                "Skip execution of kernel '%s': no pending groups", kernel_name
            )
            return set()

        logger.info(
            "Executing kernel '%s' (%d groups)", kernel_name, len(pending_groups)
        )

        written: set[str] = set()
        resident = getattr(type(parameters), "kernel_inputs_are_resident", None)
        if (
            self._registry.get_kernel_execution_protocol(kernel_name)
            is KernelExecutionProtocol.SEQUENTIAL
            and resident is not None
            and resident(parameters)
            and sum(
                1
                + len(group.context.member_uids)
                * len(group.required_inputs.parameter_fields)
                + len(group.required_inputs.aggregate_inputs)
                for group in pending_groups
            )
            <= _MAX_RESIDENT_ARGUMENTS
        ):
            batches = [pending_groups]
        else:
            batches = self._batch_groups_by_budget(pending_groups, parameters)
        if not isinstance(journal, RunJournal):
            for batch in batches:
                pointers = self._prepare_batch(batch, parameters)
                written |= self._execute_batch(kernel_name, batch, pointers)
            return written

        compare = comparison_for_scope(
            self._registry.get_output_comparison(kernel_name), parameters
        )
        for batch in batches:
            with journal.activities("kernel", len(batch), config) as runs:
                assignments = dict(
                    zip((group.context for group in batch), runs, strict=True)
                )
                pointers = self._prepare_batch(batch, parameters)
                written |= self._execute_batch(
                    kernel_name,
                    batch,
                    pointers,
                    implementation=implementation,
                    journal=journal,
                    runs=assignments,
                    compare=compare,
                )
        return written

    def _collect_pending_groups(
        self,
        kernel_name: str,
        parameters: IParameterView,
        *,
        check_presence: bool = True,
        pairs: CoordinatePairs | None = None,
    ) -> list[PendingGroup]:
        """Collect groups that need kernel execution."""
        apply_level = self._registry.get_kernel_apply_level(kernel_name)

        target_fields = self._registry.get_fields_kernel_produce(kernel_name)
        pending: list[PendingGroup] = []

        symmetry = self._registry.get_kernel_symmetry(kernel_name)
        parameter_fields, aggregate_fields = self._required_fields(kernel_name)
        parameter_inputs = RequiredInputs(parameter_fields, ())
        if self._group_scope is not parameters or self._group_pairs != pairs:
            self.clear_execution_state()
            self._group_scope = parameters
            self._group_pairs = pairs
        if apply_level not in self._groups:
            self._groups[apply_level] = list(
                aggregate_groups(parameters, apply_level, pairs=pairs)
            )
        seen_members: set[frozenset[str]] = set()
        for context, group in self._groups[apply_level]:
            contexts = (context,)
            if apply_level == KernelApplyLevel.CROSS_MODEL and pairs is not None:
                if symmetry is False:
                    contexts = (replace(context, directed=True),)
                else:
                    members = frozenset(context.member_uids)
                    if members in seen_members:
                        continue
                    seen_members.add(members)
            elif apply_level == KernelApplyLevel.CROSS_MODEL and symmetry is not None:
                if len(context.coordinates) != PAIR_SIZE:
                    logger.warning(
                        "Skipping kernel '%s': a pair needs two endpoints, found %s",
                        kernel_name,
                        context.coordinates,
                    )
                    continue
                if not symmetry:
                    contexts = (
                        replace(context, directed=True),
                        AggregationContext(
                            context.coordinates[::-1],
                            context.member_uids[::-1],
                            directed=True,
                        ),
                    )
            for task_context in contexts:
                if check_presence and self._all_fields_computed(
                    target_fields, task_context
                ):
                    continue
                inputs = parameter_inputs
                if aggregate_fields:
                    aggregate_inputs = tuple(
                        AggregateInput(
                            field_name=field,
                            uid=self._aggregate_uid(
                                field, task_context.member_uids, directed
                            ),
                        )
                        for field, directed in aggregate_fields
                    )
                    inputs = RequiredInputs(parameter_fields, aggregate_inputs)
                pending.append(
                    PendingGroup(
                        context=task_context,
                        view=group,
                        required_inputs=inputs,
                    )
                )

        return pending

    def _trace_candidates(
        self,
        kernel_name: str,
        parameters: IParameterView,
        *,
        pairs: CoordinatePairs | None = None,
    ) -> list[tuple[PendingGroup, OutputKeys]]:
        groups = self._collect_pending_groups(
            kernel_name, parameters, check_presence=False, pairs=pairs
        )
        fields = self._registry.get_fields_kernel_produce(kernel_name)
        return [
            (
                group,
                tuple(
                    (
                        self._aggregate_uid(
                            field,
                            group.context.member_uids,
                            group.context.directed,
                        ),
                        field,
                    )
                    for field in fields
                ),
            )
            for group in groups
        ]

    @staticmethod
    def _read_traces(
        storage: CatalogStorageManager,
        candidates: list[tuple[PendingGroup, OutputKeys]],
        *,
        inspect_only: bool = False,
    ) -> dict[tuple[str, str, str], ValueTrace]:
        parameter_uids: dict[str, None] = {}
        parameter_fields: dict[str, None] = {}
        aggregate_uids: dict[str, None] = {}
        for group, outputs in candidates:
            fields = group.required_inputs.parameter_fields
            if fields:
                parameter_uids.update(dict.fromkeys(group.context.member_uids))
                parameter_fields.update(dict.fromkeys(fields))
            aggregate_uids.update(
                dict.fromkeys(
                    item.uid for item in group.required_inputs.aggregate_inputs
                )
            )
            aggregate_uids.update(dict.fromkeys(uid for uid, _ in outputs))

        return storage.read_value_traces(
            {
                "parameter": (tuple(parameter_uids), tuple(parameter_fields)),
                "aggregate": (tuple(aggregate_uids), ("value",)),
            },
            inspect_only=inspect_only,
        )

    @staticmethod
    def _group_traces(
        group: PendingGroup,
        outputs: OutputKeys,
        values: dict[tuple[str, str, str], ValueTrace],
    ) -> tuple[tuple[ValueTrace | None, ...], list[ValueTrace | None]]:
        inputs = [
            values.get(("parameter", uid, field))
            for uid in group.context.member_uids
            for field in group.required_inputs.parameter_fields
        ]
        inputs.extend(
            values.get(("aggregate", item.uid, "value"))
            for item in group.required_inputs.aggregate_inputs
        )
        return tuple(
            values.get(("aggregate", uid, "value")) for uid, _ in outputs
        ), inputs

    def _partition_traced_groups(
        self,
        storage: CatalogStorageManager,
        config_hash: str,
        candidates: list[tuple[PendingGroup, OutputKeys]],
    ) -> tuple[
        list[PendingGroup],
        list[tuple[PendingGroup, OutputKeys]],
        dict[tuple[str, str, str], ValueTrace],
    ]:
        """Select runnable and inherited groups from batch metadata."""
        values = self._read_traces(storage, candidates)
        pending = []
        inherited = []
        for group, outputs in candidates:
            produced, inputs = self._group_traces(group, outputs, values)
            decision = decide_step(produced, inputs, config_hash)
            if decision is StepDecision.COMPUTE:
                pending.append(group)
            elif decision is StepDecision.ADOPT:
                inherited.append((group, outputs))
        return pending, inherited, values

    def inspect(
        self, kernel_name: str, field: str, parameters: IParameterView
    ) -> tuple[FieldStatus, ...]:
        """Describe the stored aggregates in the current scope."""
        storage = getattr(self._aggregate_repository, "storage_manager", None)
        if not isinstance(getattr(storage, "journal", None), RunJournal):
            raise TypeError("Results status requires a store with value traces")
        config = self._registry.get_kernel_config_identity(kernel_name)
        candidates = self._trace_candidates(kernel_name, parameters)
        values = self._read_traces(storage, candidates, inspect_only=True)
        report: list[FieldStatus] = []
        for group, outputs in candidates:
            uid = next(uid for uid, name in outputs if name == field)
            produced, inputs = self._group_traces(group, outputs, values)
            report.append(
                describe_step(
                    field,
                    uid,
                    values.get(("aggregate", uid, "value")),
                    produced,
                    inputs,
                    config,
                )
            )
        return tuple(report)

    def _adopt_groups(
        self,
        storage: CatalogStorageManager,
        journal: RunJournal,
        config: KernelConfigIdentity,
        inherited: list[tuple[PendingGroup, OutputKeys]],
    ) -> None:
        with (
            journal.activities("kernel", len(inherited), config) as runs,
            storage.catalog.index,
        ):
            _, eligible, traces = self._partition_traced_groups(
                storage, config.config_hash, inherited
            )
            if len(eligible) != len(inherited):
                raise RuntimeError(
                    "An input or output changed during verification. "
                    "Retry compute.apply."
                )
            for (_, outputs), run in zip(eligible, runs, strict=True):
                for uid, field in outputs:
                    storage.adopt_value(
                        uid,
                        "value",
                        traces["aggregate", uid, "value"],
                        run,
                        table=TABLE_AGGREGATES,
                        output_name=field,
                    )

    def _all_fields_computed(
        self, field_names: tuple[str, ...], context: AggregationContext
    ) -> bool:
        """Check if all target fields are already computed for the context."""
        if self._aggregate_repository is None:
            return False

        for field in field_names:
            uid = self._aggregate_uid(field, context.member_uids, context.directed)
            try:
                proxy = self._aggregate_repository.get_proxy(uid)
            except KeyError:
                return False

            if not proxy.has_field("value"):
                return False

        return True

    def _required_fields(
        self, kernel_name: str
    ) -> tuple[tuple[str, ...], list[tuple[str, bool]]]:
        """Determine each required field's owner kind and aggregate direction."""
        parameter_fields: list[str] = []
        aggregate_fields: list[tuple[str, bool]] = []

        for field_name in self._registry.get_fields_kernel_require(kernel_name):
            if not self._registry.can_produce_field(field_name):
                parameter_fields.append(field_name)
                continue

            producer = self._registry.get_kernel_producing_field(field_name)
            apply_level = self._registry.get_kernel_apply_level(producer)

            if apply_level == KernelApplyLevel.PARAMETER:
                parameter_fields.append(field_name)
            else:
                aggregate_fields.append(
                    (field_name, self._registry.get_kernel_symmetry(producer) is False)
                )

        return tuple(parameter_fields), aggregate_fields

    def _batch_groups_by_budget(
        self, groups: list[PendingGroup], base_view: IParameterView
    ) -> list[list[PendingGroup]]:
        """Batch groups by memory budget using iter_chunks_by_read_budget."""
        if not groups:
            return []

        required_fields_by_uid: dict[str, list[str]] = {}
        for group in groups:
            fields = list(group.required_inputs.parameter_fields)
            required_fields_by_uid.update(
                dict.fromkeys(group.context.member_uids, fields)
            )
        chunks = base_view[list(required_fields_by_uid)].iter_chunks_by_read_budget(
            required_fields_by_uid=required_fields_by_uid,
            parallel=self._parallel,
        )
        chunk_by_uid = {
            uid: i for i, chunk in enumerate(chunks) for uid in chunk.list_uids()
        }
        return [
            list(batch)
            for _, batch in groupby(
                groups, key=lambda g: chunk_by_uid[g.context.member_uids[0]]
            )
        ]

    def _prepare_batch(
        self, batch: list[PendingGroup], base_view: IParameterView
    ) -> dict[tuple[str, str], ValuePointer] | None:
        """Resolve uncached inputs or prefetch through the configured cache."""
        fields_by_uid: dict[str, dict[str, None]] = {}
        for group in batch:
            names = dict.fromkeys(group.required_inputs.parameter_fields)
            for uid in group.context.member_uids:
                fields_by_uid.setdefault(uid, {}).update(names)

        pointers = None
        if fields_by_uid:
            view = base_view[list(fields_by_uid)]
            fields = tuple(
                dict.fromkeys(
                    field for names in fields_by_uid.values() for field in names
                )
            )
            resolve = getattr(type(view), "resolve_kernel_inputs", None)
            if resolve is not None:
                pointers = resolve(view, fields)
            if pointers is None:
                view.prefetch_fields(
                    fields_by_uid={
                        uid: list(names) for uid, names in fields_by_uid.items()
                    },
                    parallel=self._parallel,
                )

        aggregate_uids = self._collect_aggregate_uids(batch)
        if aggregate_uids and self._aggregate_repository is not None:
            aggregate_view = self._aggregate_repository.create_view()
            aggregate_view[aggregate_uids].prefetch_fields(
                fields=["value"],
                parallel=self._parallel,
            )
        return pointers

    def _collect_aggregate_uids(self, batch: list[PendingGroup]) -> list[str]:
        """Collect all aggregate UIDs needed for a batch."""
        aggregate_uids: list[str] = []
        for group in batch:
            aggregate_uids.extend(
                item.uid for item in group.required_inputs.aggregate_inputs
            )
        return aggregate_uids

    def _execute_batch(
        self,
        kernel_name: str,
        batch: list[PendingGroup],
        pointers: dict[tuple[str, str], ValuePointer] | None,
        *,
        implementation: Callable[..., Any] | None = None,
        journal: RunJournal | None = None,
        runs: dict[AggregationContext, Run] | None = None,
        compare: OutputComparison | None = None,
    ) -> set[str]:
        """Execute aggregation-level kernel on a batch with streaming writes.

        Returns:
            The set of field names written for this batch.
        """
        required_args = self._registry.get_fields_kernel_require(kernel_name)
        tasks: dict[AggregationContext, tuple[Any, ...]] = {}

        for group in batch:
            task_args = self._build_task_args(required_args, group, pointers)
            tasks[group.context] = tuple(task_args)

        apply_restrictions_filter(
            kernel_name,
            tasks,
            self._registry.get_kernel_restrictions(kernel_name),
        )
        if not tasks:
            return set()

        if self._aggregate_repository is None:
            msg = "AggregateRepository required for aggregation kernels"
            raise RuntimeError(msg)

        written: set[str] = set()
        with self._aggregate_repository:
            for context, result in self._stream_results(
                kernel_name, tasks, implementation
            ):
                logger.debug(
                    "Set results of kernel '%s' for group '%s'",
                    kernel_name,
                    context.coordinates,
                )
                with (
                    journal.activate(runs[context], compare=compare)
                    if journal is not None and runs is not None
                    else nullcontext()
                ):
                    written |= self._store_result(kernel_name, context, result)
        return written

    def _build_task_args(
        self,
        required_args: tuple[str, ...],
        group: PendingGroup,
        pointers: dict[tuple[str, str], ValuePointer] | None,
    ) -> list[Any]:
        """Build argument list for a single aggregation task."""
        task_args: list[Any] = []

        dependencies = {
            item.field_name: item.uid for item in group.required_inputs.aggregate_inputs
        }
        ordered = []
        if group.required_inputs.parameter_fields:
            members = {param.meta.uid: param for param in group.view}
            ordered = [members[uid] for uid in group.context.member_uids]
        for field in required_args:
            if field in dependencies:
                task_args.append(self._read_aggregate_value(dependencies[field]))
            else:
                values = []
                for member in ordered:
                    pointer = (
                        pointers.get((member.meta.uid, field))
                        if pointers is not None
                        else None
                    )
                    values.append(
                        member.get_field(field)
                        if pointer is None
                        else member.read_resolved_field(
                            field, pointer, auto_prefetch=False, kernel_input=True
                        )
                    )
                task_args.append(values)

        return task_args

    def _read_aggregate_value(self, uid: str) -> Any:
        """Read the value of a resolved aggregate dependency."""
        if self._aggregate_repository is None:
            msg = f"AggregateRepository not configured, cannot read '{uid}'"
            raise KeyError(msg)

        try:
            aggregate = self._aggregate_repository.get_proxy(uid)
        except KeyError:
            msg = f"Missing contextual dependency '{uid}'"
            raise KeyError(msg) from None

        if not aggregate.has_field("value"):
            msg = f"Aggregate '{uid}' exists but has no value"
            raise KeyError(msg)

        return aggregate.get_field("value", kernel_input=True)

    def _store_result(
        self, kernel_name: str, context: AggregationContext, result: Any
    ) -> set[str]:
        """Store kernel result in aggregate repository.

        Returns:
            The set of field names written for this result.
        """
        normalized = self._registry.normalize_kernel_result(kernel_name, result)
        for field_name, value in normalized.items():
            aggregate = self._aggregate_repository.get_or_create(
                field_name=field_name,
                member_uids=context.member_uids,
                directed=context.directed,
            )
            aggregate.set_field("value", value)
            if isinstance(result, KernelResult):
                journal = getattr(
                    self._aggregate_repository.storage_manager, "journal", None
                )
                if journal is not None:
                    journal.record_arithmetic(
                        "aggregate",
                        aggregate.meta.uid,
                        result.arithmetic(),
                        observed=self._registry.is_builtin(kernel_name),
                    )
        return set(normalized.keys())

    def _stream_results(
        self,
        kernel_name: str,
        tasks: dict[Any, tuple[Any, ...]],
        implementation: Callable[..., Any] | None = None,
    ) -> Iterator[tuple[Any, Any]]:
        """Execute kernel tasks and yield results as they become available."""
        strategy = create_execution_strategy(
            kernel_name,
            self._registry,
            self._process_pool,
        )
        if implementation is None:
            implementation = self._registry.get_kernel_implementation(kernel_name)
        yield from strategy.execute_tasks(kernel_name, tasks, implementation)
