"""Parameter-level kernel execution logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diffract.core.compute.config import KernelResult
from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.nn.params.proxy import ParameterDataProxy

from ._types import MAX_RESIDENT_ARGUMENTS
from .enums import KernelExecutionProtocol
from .restrictions import apply_restrictions_filter
from .strategy import create_execution_strategy
from .traces import StepDecision, comparison_for_scope, decide_step, describe_step
from .utils import execute_kernel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import ProcessPoolExecutor

    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.metadata.runs import RunJournal, ValueTrace
    from diffract.core.data.nn.params.interface import IParameterProxy, IParameterView
    from diffract.core.parallel import ParallelContext
    from diffract.core.storage.catalog import ValuePointer
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    from .traces import FieldStatus

logger = logging.getLogger(__name__)


class ParameterKernelRunner:
    """Executes parameter-level kernels with batching and streaming writes.

    Handles execution of kernels that operate on individual parameters,
    including memory-aware chunking, prefetching, and result streaming.
    """

    def __init__(
        self,
        registry: KernelRegistry,
        process_pool: ProcessPoolExecutor | None,
        parallel: ParallelContext | None,
    ) -> None:
        """Initialize the parameter kernel runner.

        Args:
            registry: Kernel registry for metadata lookup.
            process_pool: Optional process pool for parallel execution.
            parallel: Optional parallel context for view operations.
        """
        self._registry = registry
        self._process_pool = process_pool
        self._parallel = parallel

    def run(
        self, kernel_name: str, parameters: IParameterView, *, force: bool = False
    ) -> set[str]:
        """Execute kernel on individual parameters.

        Args:
            kernel_name: Name of the registered kernel.
            parameters: Parameter collection to process.
            force: Execute the kernel even for parameters with fresh outputs.

        Returns:
            The set of field names actually written to storage (empty when
            every parameter already had the target fields or nothing matched).
        """
        target_fields = self._registry.get_fields_kernel_produce(kernel_name)
        get_scope = getattr(type(parameters), "kernel_trace_scope", None)
        scope = None if get_scope is None else get_scope(parameters)
        if scope is not None:
            storage, uids, unfiltered = scope
            return self._run_traced(
                kernel_name,
                parameters,
                target_fields,
                storage,
                uids,
                unfiltered,
                force=force,
            )

        pending = (
            parameters if force else self._pending_outputs(parameters, target_fields)
        )
        return self._run_pending(kernel_name, pending) if pending else set()

    def _run_traced(
        self,
        kernel_name: str,
        parameters: IParameterView,
        target_fields: tuple[str, ...],
        storage: CatalogStorageManager,
        uids: list[str],
        unfiltered: bool,
        *,
        force: bool,
    ) -> set[str]:
        if not uids:
            return set()
        config, implementation = self._registry.capture_kernel_execution(kernel_name)
        inputs = self._registry.get_fields_kernel_require(kernel_name)
        unresolved = []
        all_outputs_absent = False
        if not force:
            verification = storage.unverified_parameter_owners(
                None if unfiltered else uids, inputs, target_fields, config.config_hash
            )
            if verification is None:
                unresolved = uids
            else:
                unresolved = verification.candidates
                all_outputs_absent = verification.all_outputs_absent
                if unfiltered and unresolved:
                    scope = set(uids)
                    unresolved = [uid for uid in unresolved if uid in scope]
        run_all = force or all_outputs_absent
        if not run_all and not unresolved:
            return set()

        fields = tuple(dict.fromkeys((*inputs, *target_fields)))
        traces = (
            storage.read_value_traces({"parameter": (unresolved, fields)})
            if unresolved
            else {}
        )

        def decide(
            uid: str, values: dict[tuple[str, str, str], ValueTrace]
        ) -> StepDecision:
            return decide_step(
                self._parameter_traces(values, uid, target_fields),
                self._parameter_traces(values, uid, inputs),
                config.config_hash,
                requires_kernel_trace=config.requires_kernel_trace,
            )

        pending = list(uids) if run_all else []
        inherited: list[str] = []
        for uid in unresolved:
            decision = decide(uid, traces)
            if decision is StepDecision.COMPUTE:
                pending.append(uid)
            elif decision is StepDecision.ADOPT:
                inherited.append(uid)
        if not pending and not inherited:
            return set()

        compare = comparison_for_scope(
            self._registry.get_output_comparison(kernel_name), parameters
        )
        with storage.journal.activity("kernel", config, compare=compare) as run:
            written = (
                self._run_pending(
                    kernel_name,
                    parameters[pending],
                    implementation,
                    outputs_absent=all_outputs_absent,
                    journal=storage.journal,
                )
                if pending
                else set()
            )
            if inherited:
                with storage.catalog.index:
                    latest = storage.read_value_traces(
                        {"parameter": (inherited, fields)}
                    )
                    for uid in inherited:
                        if decide(uid, latest) is not StepDecision.ADOPT:
                            raise RuntimeError(
                                "An input or output changed during verification. "
                                "Retry compute.apply."
                            )
                        for field in target_fields:
                            storage.adopt_value(
                                uid,
                                field,
                                latest["parameter", uid, field],
                                run,
                                table=TABLE_PARAMETERS,
                                output_name=field,
                            )
            return written

    @staticmethod
    def _parameter_traces(
        values: dict[tuple[str, str, str], ValueTrace],
        uid: str,
        fields: tuple[str, ...],
    ) -> tuple[ValueTrace | None, ...]:
        return tuple(values.get(("parameter", uid, field)) for field in fields)

    def inspect(
        self, kernel_name: str, field: str, parameters: IParameterView
    ) -> tuple[FieldStatus, ...]:
        """Describe the stored parameter results in the current scope."""
        get_scope = getattr(type(parameters), "kernel_trace_scope", None)
        scope = None if get_scope is None else get_scope(parameters)
        if scope is None:
            raise RuntimeError("Results status requires a store with value traces")
        storage, _, _ = scope
        uids = parameters.list_uids()
        inputs = self._registry.get_fields_kernel_require(kernel_name)
        outputs = self._registry.get_fields_kernel_produce(kernel_name)
        fields = tuple(dict.fromkeys((*inputs, *outputs)))
        config = self._registry.get_kernel_config_identity(kernel_name)
        traces = storage.read_value_traces(
            {"parameter": (uids, fields)}, inspect_only=True
        )
        return tuple(
            describe_step(
                field,
                uid,
                traces.get(("parameter", uid, field)),
                self._parameter_traces(traces, uid, outputs),
                self._parameter_traces(traces, uid, inputs),
                config,
            )
            for uid in uids
        )

    def _run_pending(
        self,
        kernel_name: str,
        pending: IParameterView,
        implementation: Callable[..., Any] | None = None,
        *,
        outputs_absent: bool = False,
        journal: RunJournal | None = None,
    ) -> set[str]:
        required_fields = list(self._registry.get_fields_kernel_require(kernel_name))
        resident = getattr(type(pending), "kernel_inputs_are_resident", None)
        if (
            self._registry.get_kernel_execution_protocol(kernel_name)
            is KernelExecutionProtocol.SEQUENTIAL
            and resident is not None
            and resident(pending)
            and len(pending) * (1 + len(required_fields)) <= MAX_RESIDENT_ARGUMENTS
        ):
            chunks = [pending]
        else:
            required_by_uid = {p.meta.uid: required_fields for p in pending}
            chunks = list(
                pending.iter_chunks_by_read_budget(
                    required_fields_by_uid=required_by_uid,
                    parallel=self._parallel,
                )
            )

        written: set[str] = set()
        for chunk in chunks:
            written |= self._execute_batch(
                kernel_name,
                chunk,
                implementation,
                outputs_absent=outputs_absent,
                journal=journal,
            )
        return written

    def _pending_outputs(
        self, parameters: IParameterView, target_fields: tuple[str, ...]
    ) -> IParameterView:
        lookup = getattr(type(parameters), "pending_outputs", None)
        if lookup is not None:
            pending = lookup(parameters, *target_fields)
            if pending is not None:
                return pending
        if len(target_fields) == 1:
            return parameters.filter_by_fields(
                *target_fields, inverse_mask=True, parallel=self._parallel
            )
        complete = parameters.filter_by_fields(*target_fields, parallel=self._parallel)
        complete_uids = set(complete.list_uids())
        return parameters[
            [uid for uid in parameters.list_uids() if uid not in complete_uids]
        ]

    def _execute_batch(
        self,
        kernel_name: str,
        batch: IParameterView,
        implementation: Callable[..., Any] | None = None,
        *,
        outputs_absent: bool = False,
        journal: RunJournal | None = None,
    ) -> set[str]:
        """Execute parameter-level kernel on a batch with streaming writes.

        Results are written immediately as they become available, reducing
        memory pressure for heavy computations (e.g., SVD matrices).

        Returns:
            The set of field names written for this batch.
        """
        required_args = self._registry.get_fields_kernel_require(kernel_name)
        stream_inputs = self._registry.streams_inputs(kernel_name)
        resolve = getattr(type(batch), "resolve_kernel_inputs", None)
        pointers = resolve(batch, required_args) if resolve is not None else None
        if pointers is None and not stream_inputs:
            batch.prefetch_fields(fields=list(required_args), parallel=self._parallel)

        param_lookup = {param.meta.uid: param for param in batch}
        if not param_lookup:
            return set()
        results: Iterator[tuple[str, Any]]
        if stream_inputs:
            if implementation is None:
                implementation = self._registry.get_kernel_implementation(kernel_name)
            results = (
                (
                    uid,
                    execute_kernel(
                        kernel_name,
                        implementation,
                        self._read_arguments(
                            param, required_args, pointers, auto_prefetch=False
                        ),
                    ),
                )
                for uid, param in param_lookup.items()
            )
        else:
            tasks = {
                uid: self._read_arguments(
                    param, required_args, pointers, auto_prefetch=True
                )
                for uid, param in param_lookup.items()
            }
            apply_restrictions_filter(
                kernel_name,
                tasks,
                self._registry.get_kernel_restrictions(kernel_name),
            )
            if not tasks:
                return set()
            results = self._stream_results(kernel_name, tasks, implementation)

        written: set[str] = set()
        with batch:
            for key, result in results:
                normalized = self._registry.normalize_kernel_result(kernel_name, result)
                param = param_lookup[key]
                if isinstance(result, KernelResult) and journal is not None:
                    journal.record_arithmetic(
                        "parameter",
                        key,
                        result.arithmetic()
                        | {
                            "parameter_name": param.meta.name,
                            "checkpoint_dtype": param.meta.other_meta.get(
                                "source_dtype"
                            ),
                        },
                        observed=self._registry.is_builtin(kernel_name),
                    )
                for field_name, value in normalized.items():
                    if outputs_absent and type(param) is ParameterDataProxy:
                        param.set_field(field_name, value, initial=True)
                    else:
                        param.set_field(field_name, value)
                    written.add(field_name)
        return written

    @staticmethod
    def _read_arguments(
        param: IParameterProxy,
        fields: tuple[str, ...],
        pointers: dict[tuple[str, str], ValuePointer] | None,
        *,
        auto_prefetch: bool,
    ) -> tuple[Any, ...]:
        args = []
        for field in fields:
            pointer = pointers.get((param.meta.uid, field)) if pointers else None
            if pointer is None:
                args.append(param.get_field(field, auto_prefetch=auto_prefetch))
            else:
                args.append(
                    param.read_resolved_field(
                        field, pointer, auto_prefetch=False, kernel_input=True
                    )
                )
        return tuple(args)

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
