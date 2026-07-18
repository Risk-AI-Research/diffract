"""Aggregation-level kernel execution logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diffract.core.data.nn.aggregates.metadata import AggregateMetadata

from ._types import AggregateInput, PendingGroup, RequiredInputs
from .aggregation import AggregationContext, aggregate_models, aggregate_parameters
from .enums import KernelApplyLevel
from .restrictions import apply_restrictions_filter
from .strategy import create_execution_strategy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import ProcessPoolExecutor

    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates import AggregateRepository
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.parallel import ParallelContext

logger = logging.getLogger(__name__)


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

    def run(self, kernel_name: str, parameters: IParameterView) -> set[str]:
        """Execute kernel with aggregation across models or parameters.

        Args:
            kernel_name: Name of the registered kernel.
            parameters: Parameter collection to process.

        Returns:
            The set of field names actually written to the aggregate
            repository (empty when every group was already computed, dropped by
            a restriction, or there is no repository configured).
        """
        pending_groups = self._collect_pending_groups(kernel_name, parameters)
        if not pending_groups:
            logger.debug(
                "Skip execution of kernel '%s': no pending groups", kernel_name
            )
            return set()

        logger.info(
            "Executing kernel '%s' (%d groups)", kernel_name, len(pending_groups)
        )

        written: set[str] = set()
        batches = self._batch_groups_by_budget(pending_groups, parameters)
        for batch in batches:
            self._prefetch_batch(batch, parameters)
            written |= self._execute_batch(kernel_name, batch)
        return written

    def _collect_pending_groups(
        self, kernel_name: str, parameters: IParameterView
    ) -> list[PendingGroup]:
        """Collect groups that need kernel execution."""
        apply_level = self._registry.get_kernel_apply_level(kernel_name)

        if apply_level == KernelApplyLevel.IN_MODEL:
            group_iter, build_context = aggregate_parameters(parameters)
        elif apply_level == KernelApplyLevel.CROSS_MODEL:
            group_iter, build_context = aggregate_models(parameters)
        else:
            msg = f"Unsupported level for grouping: {apply_level}"
            raise ValueError(msg)

        target_fields = self._registry.get_fields_kernel_produce(kernel_name)
        pending: list[PendingGroup] = []

        for group_id, group in group_iter:
            context = build_context(group_id, group)

            if self._all_fields_computed(target_fields, context):
                logger.debug(
                    "Skip execution of kernel '%s' for group '%s'",
                    kernel_name,
                    group_id,
                )
                continue

            required_inputs = self._get_required_inputs(kernel_name, context)
            pending.append(
                PendingGroup(
                    group_id=group_id,
                    context=context,
                    view=group,
                    required_inputs=required_inputs,
                )
            )

        return pending

    def _all_fields_computed(
        self, field_names: tuple[str, ...], context: AggregationContext
    ) -> bool:
        """Check if all target fields are already computed for the context."""
        if self._aggregate_repository is None:
            return False

        for field in field_names:
            uid = AggregateMetadata.create_uid_from_context(
                field_name=field,
                context_models=context.models or (),
                context_params=context.parameters or (),
            )
            try:
                proxy = self._aggregate_repository.get_proxy(uid)
            except KeyError:
                return False

            if not proxy.has_field("value"):
                return False

        return True

    def _get_required_inputs(
        self, kernel_name: str, context: AggregationContext
    ) -> RequiredInputs:
        """Determine required inputs for executing an aggregation kernel."""
        parameter_fields: list[str] = []
        aggregate_inputs: list[AggregateInput] = []

        for field_name in self._registry.get_fields_kernel_require(kernel_name):
            if not self._registry.can_produce_field(field_name):
                parameter_fields.append(field_name)
                continue

            producer = self._registry.get_kernel_producing_field(field_name)
            apply_level = self._registry.get_kernel_apply_level(producer)

            if apply_level == KernelApplyLevel.PARAMETER:
                parameter_fields.append(field_name)
            else:
                aggregate_inputs.append(
                    AggregateInput(
                        field_name=field_name,
                        context_models=context.models or (),
                        context_params=context.parameters or (),
                    )
                )

        return RequiredInputs(
            parameter_fields=tuple(parameter_fields),
            aggregate_inputs=tuple(aggregate_inputs),
        )

    def _batch_groups_by_budget(
        self, groups: list[PendingGroup], base_view: IParameterView
    ) -> list[list[PendingGroup]]:
        """Batch groups by memory budget using iter_chunks_by_read_budget."""
        if not groups:
            return []

        uid_to_group: dict[str, PendingGroup] = {}
        required_fields_by_uid: dict[str, list[str]] = {}

        for group in groups:
            for uid in group.view.list_uids():
                uid_to_group[uid] = group
                fields = list(group.required_inputs.parameter_fields)
                required_fields_by_uid[uid] = fields

        all_uids = list(uid_to_group.keys())
        if not all_uids:
            return []

        combined_view = base_view[all_uids]
        chunks = list(
            combined_view.iter_chunks_by_read_budget(
                required_fields_by_uid=required_fields_by_uid,
                parallel=self._parallel,
            )
        )

        batches: list[list[PendingGroup]] = []
        for chunk in chunks:
            seen_groups: set[str] = set()
            batch: list[PendingGroup] = []
            for uid in chunk.list_uids():
                group = uid_to_group[uid]
                if group.group_id not in seen_groups:
                    seen_groups.add(group.group_id)
                    batch.append(group)
            if batch:
                batches.append(batch)

        return batches

    def _prefetch_batch(
        self, batch: list[PendingGroup], base_view: IParameterView
    ) -> None:
        """Prefetch fields for a batch of groups."""
        # Prefetch parameter fields
        param_fields_by_uid: dict[str, list[str]] = {}
        for group in batch:
            for uid in group.view.list_uids():
                param_fields_by_uid[uid] = list(group.required_inputs.parameter_fields)

        if param_fields_by_uid:
            batch_uids = list(param_fields_by_uid.keys())
            base_view[batch_uids].prefetch_fields(
                fields_by_uid=param_fields_by_uid,
                parallel=self._parallel,
            )

        # Prefetch aggregate fields
        aggregate_uids = self._collect_aggregate_uids(batch)
        if aggregate_uids and self._aggregate_repository is not None:
            aggregate_view = self._aggregate_repository.create_view()
            aggregate_view[aggregate_uids].prefetch_fields(
                fields=["value"],
                parallel=self._parallel,
            )

    def _collect_aggregate_uids(self, batch: list[PendingGroup]) -> list[str]:
        """Collect all aggregate UIDs needed for a batch."""
        aggregate_uids: list[str] = []
        for group in batch:
            for agg_input in group.required_inputs.aggregate_inputs:
                uid = AggregateMetadata.create_uid_from_context(
                    field_name=agg_input.field_name,
                    context_models=agg_input.context_models,
                    context_params=agg_input.context_params,
                )
                aggregate_uids.append(uid)
        return aggregate_uids

    def _execute_batch(self, kernel_name: str, batch: list[PendingGroup]) -> set[str]:
        """Execute aggregation-level kernel on a batch with streaming writes.

        Returns:
            The set of field names written for this batch.
        """
        required_args = self._registry.get_fields_kernel_require(kernel_name)
        tasks: dict[tuple[str, AggregationContext], tuple[Any, ...]] = {}

        for group in batch:
            task_args = self._build_task_args(required_args, group)
            tasks[(group.group_id, group.context)] = tuple(task_args)

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
            for (group_id, context), result in self._stream_results(kernel_name, tasks):
                logger.debug(
                    "Set results of kernel '%s' for group '%s'",
                    kernel_name,
                    group_id,
                )
                written |= self._store_result(kernel_name, context, result)
        return written

    def _build_task_args(
        self, required_args: tuple[str, ...], group: PendingGroup
    ) -> list[Any]:
        """Build argument list for a single aggregation task."""
        task_args: list[Any] = []

        for required_arg in required_args:
            if not self._registry.can_produce_field(required_arg):
                # Raw field from parameters
                values = [p.get_field(required_arg) for p in group.view]
                task_args.append(values)
                continue

            dependency = self._registry.get_kernel_producing_field(required_arg)
            dependency_level = self._registry.get_kernel_apply_level(dependency)

            if dependency_level == KernelApplyLevel.PARAMETER:
                # Parameter-level computed field
                values = [p.get_field(required_arg) for p in group.view]
                task_args.append(values)
            else:
                # Aggregate-level computed field
                value = self._read_aggregate_value(required_arg, group.context)
                task_args.append(value)

        return task_args

    def _read_aggregate_value(
        self, field_name: str, context: AggregationContext
    ) -> Any:
        """Read an aggregate value from AggregateRepository."""
        uid = AggregateMetadata.create_uid_from_context(
            field_name=field_name,
            context_models=context.models or (),
            context_params=context.parameters or (),
        )

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

        return aggregate.get_field("value")

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
                context_models=context.models or (),
                context_params=context.parameters or (),
            )
            aggregate.set_field("value", value)
        return set(normalized.keys())

    def _stream_results(
        self, kernel_name: str, tasks: dict[Any, tuple[Any, ...]]
    ) -> Iterator[tuple[Any, Any]]:
        """Execute kernel tasks and yield results as they become available."""
        strategy = create_execution_strategy(
            kernel_name,
            self._registry,
            self._process_pool,
        )
        implementation = self._registry.get_kernel_implementation(kernel_name)
        yield from strategy.execute_tasks(kernel_name, tasks, implementation)
