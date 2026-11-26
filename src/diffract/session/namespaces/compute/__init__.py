"""Compute namespace for Session."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from diffract.core.compute import KernelConfig, KernelExecutor, KernelRegistry
from diffract.core.compute.execution import (
    KernelApplyLevel,
    KernelExecutionProtocol,
    KernelRestrictions,
)
from diffract.core.compute.execution.aggregation import aggregate_groups
from diffract.core.data.nn.aggregates.repository import AggregateRepository
from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.utils.exceptions import format_exception_message
from diffract.session.errors import (
    FieldNotFoundError,
    KernelNotFoundError,
    ScopeValidationError,
    SessionError,
)
from diffract.session.session import Session, SessionContext
from diffract.session.summaries import ApplySummary
from diffract.session.utils import did_you_mean

from .pairs import Checkpoint, resolve_pairs, shape_restrictions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from diffract.core.compute import OutputComparison
    from diffract.core.compute.execution.aggregation import CoordinatePairs
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.schema import ParameterType

    from .pairs import PairPlan, ResolvedPair

logger = logging.getLogger(__name__)

_BINARY_MODEL_COUNT = 2


def _binary_scope_message(
    field_name: str, kernel_name: str, endpoints: list[tuple[str, int]]
) -> str:
    """Actionable message for a binary cross-model kernel on a non-pair scope."""
    if kernel_name == field_name:
        clause = "it is a binary cross-model kernel that"
    else:
        clause = f"its dependency '{kernel_name}' is a binary cross-model kernel that"

    prefix = (
        f"Cannot produce '{field_name}': {clause} requires exactly two "
        f"(model, step) coordinates, but the scope contains {len(endpoints)}: "
        f"{endpoints}."
    )
    if len(endpoints) < _BINARY_MODEL_COUNT:
        return f"{prefix} Add a checkpoint or widen the session filter."
    chosen = next(
        (pair for pair in pairwise(endpoints) if pair[0][0] == pair[1][0]),
        endpoints[:_BINARY_MODEL_COUNT],
    )
    models = sorted({model for model, _ in chosen})
    steps = sorted({step for _, step in chosen})
    return (
        f"{prefix} Select a pair, e.g. "
        f"session.filter(model_ids={models}, steps={steps})"
        f".compute.apply('{field_name}')."
    )


class ComputeNamespace:
    """Kernel compute API for Session."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        aggregate_repository: AggregateRepository = Provide["nn.aggregate_repository"],
        kernel_registry: KernelRegistry = Provide["compute.kernel_registry"],
        kernel_executor_factory: Callable[[], KernelExecutor] = Provide[
            "compute.kernel_executor.provider"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository
        self.__kernel_registry = kernel_registry
        self.__kernel_executor_factory = kernel_executor_factory

    def apply(
        self,
        *fields_to_produce: str,
        recompute: bool = False,
        pairs: Iterable[tuple[Checkpoint, Checkpoint]] | None = None,
        return_coverage: bool = False,
        required_names: Iterable[str] = (),
    ) -> ApplySummary:
        """Compute the requested fields in dependency order within the active scope.

        Args:
            *fields_to_produce: Names of fields to compute using registered kernels.
            recompute: Run the kernels that produce the requested fields even
                when their output groups are fresh. Kernels needed only as
                dependencies still follow the freshness check.
            pairs: Ordered source and target checkpoints for cross-model kernels.
                A bare model name selects its latest step in the current scope.
                Parameters are matched by name within each selected pair.
            return_coverage: Report the join and measure shape-compatible weights
                for each requested cross-model field in explicit pairs.
            required_names: Stored parameter names required at both endpoints and
                compatible with all requested fields. Checked before execution.

        Returns:
            An ApplySummary grouping the written fields by apply level (which
            exporter now holds them) and listing any requested field that
            produced nothing. For explicit pairs, ``pair_mismatches`` lists
            parameter names without a counterpart. With ``return_coverage=True``,
            ``pair_coverage`` contains one report for each distinct ordered pair.

        Raises:
            KernelNotFoundError: If any specified field cannot be produced.
            ScopeValidationError: If an explicit pair is invalid or outside
                the active scope, or an implicit binary comparison has a scope
                with other than two checkpoints. Also raised for missing required
                names and incompatible shapes outside the requested coverage policy.
            FieldNotFoundError: If a custom input field has no producer and is
                missing from a participating parameter.
            SessionError: If kernel execution fails.
        """
        with self.__session_or_context:
            if not fields_to_produce:
                logger.warning("No fields specified for computation")
                return ApplySummary()

            for field_name in fields_to_produce:
                if not self.__kernel_registry.can_produce_field(field_name):
                    available = self.__kernel_registry.list_fields_can_produce()
                    hint = did_you_mean(field_name, available)
                    msg = (
                        f"Cannot produce '{field_name}': no registered kernel "
                        f"produces it.{hint} Use "
                        "session.compute.list_available_metrics() to see all fields."
                    )
                    raise KernelNotFoundError(msg)

            if isinstance(self.__session_or_context, Session):
                pending = self.__param_repo.create_view()
            else:
                pending = self.__session_or_context._param_filter_context

            required = frozenset(
                (required_names,) if isinstance(required_names, str) else required_names
            )
            detailed = return_coverage or bool(required)
            if detailed and (
                pairs is None
                or any(
                    self.__kernel_registry.get_kernel_apply_level(
                        self.__kernel_registry.get_kernel_producing_field(field)
                    )
                    != KernelApplyLevel.CROSS_MODEL
                    for field in fields_to_produce
                )
            ):
                raise ScopeValidationError(
                    "Pair coverage and required_names need explicit pairs "
                    "and cross-model fields."
                )
            resolved = ()
            coordinates = None
            if pairs is None:
                self._validate_scope(fields_to_produce, pending)
            else:
                pending, resolved = resolve_pairs(pending, pairs)
                coordinates = tuple(pair.coordinates for pair in resolved)

            forced_kernels = (
                frozenset(
                    self.__kernel_registry.get_kernel_producing_field(field)
                    for field in fields_to_produce
                )
                if recompute
                else frozenset()
            )
            try:
                plans = self._plan_pairs(
                    resolved, fields_to_produce, required, coverage=return_coverage
                )
                written: set[str] = set()
                batches = (
                    [
                        (fields, pending[pair.members(names)], (pair.coordinates,))
                        for pair, plan in plans
                        for names, fields in plan.batches().items()
                    ]
                    if return_coverage
                    else [(fields_to_produce, pending, coordinates)]
                )
                for fields, selection, endpoints in batches:
                    self._validate_inputs(fields, selection, endpoints)
                for fields, selection, endpoints in batches:
                    with self.__kernel_executor_factory() as executor:
                        for field in fields:
                            written |= executor.execute(
                                field_or_kernel_name=field,
                                parameters=selection,
                                forced_kernels=forced_kernels,
                                pairs=endpoints,
                            )
                coverage = tuple(
                    pair.report(plan, self.__kernel_registry, self.__agg_repo)
                    for pair, plan in plans
                    if return_coverage
                )

                produced = [f for f in fields_to_produce if f in written]
                if produced:
                    logger.info("Successfully produced fields: %s", ", ".join(produced))
                else:
                    logger.info(
                        "No new fields produced for: %s (already computed or not "
                        "applicable to the current scope)",
                        ", ".join(fields_to_produce),
                    )

                self.__session_or_context._field_cache.invalidate()

                summary = self._summarize_apply(
                    fields_to_produce, written, forced_kernels
                )
                return replace(
                    summary,
                    pair_coverage=coverage,
                    pair_mismatches=tuple(
                        pair.join
                        for pair in resolved
                        if pair.join.source_only or pair.join.target_only
                    ),
                )

            except SessionError:
                raise
            except Exception as e:
                msg = f"Failed to produce fields: {format_exception_message(e)}"
                raise SessionError(msg) from e

    def _plan_pairs(
        self,
        resolved: tuple[ResolvedPair, ...],
        fields: tuple[str, ...],
        required: frozenset[str],
        *,
        coverage: bool,
    ) -> list[tuple[ResolvedPair, PairPlan]]:
        if not resolved:
            return []
        restrictions = {
            field: shape_restrictions(self.__kernel_registry, field) for field in fields
        }
        if not (coverage or required or any(restrictions.values())):
            return []

        @cache
        def shape(uid: str) -> tuple[int, ...] | None:
            meta = self.__param_repo.get_proxy(uid).get_field_metadata("weights")
            return tuple(meta["shape"]) if meta and meta.get("shape") else None

        admissions = (
            cache(self.__param_repo.instances.extraction)
            if coverage or required
            else None
        )
        return [
            (
                pair,
                pair.plan(
                    restrictions,
                    required,
                    shape,
                    admissions,
                    skip_incompatible=coverage,
                ),
            )
            for pair in resolved
        ]

    def _summarize_apply(
        self,
        requested: tuple[str, ...],
        written: set[str],
        forced_kernels: frozenset[str],
    ) -> ApplySummary:
        """Group written fields by apply level; list requested fields skipped."""
        buckets: dict[KernelApplyLevel, list[str]] = {
            KernelApplyLevel.PARAMETER: [],
            KernelApplyLevel.IN_MODEL: [],
            KernelApplyLevel.CROSS_MODEL: [],
        }
        forced_fields = []
        for name in sorted(written):
            kernel = self.__kernel_registry.get_kernel_producing_field(name)
            buckets[self.__kernel_registry.get_kernel_apply_level(kernel)].append(name)
            if kernel in forced_kernels:
                forced_fields.append(name)
        skipped = tuple(
            (name, "no new field written (already computed or out of scope)")
            for name in requested
            if name not in written
        )
        return ApplySummary(
            parameter_fields=tuple(buckets[KernelApplyLevel.PARAMETER]),
            in_model_fields=tuple(buckets[KernelApplyLevel.IN_MODEL]),
            cross_model_fields=tuple(buckets[KernelApplyLevel.CROSS_MODEL]),
            skipped=skipped,
            forced_fields=tuple(forced_fields),
        )

    def _validate_inputs(
        self,
        fields: Iterable[str],
        selection: IParameterView,
        pairs: CoordinatePairs | None,
    ) -> None:
        registry = self.__kernel_registry
        kernels = dict.fromkeys(
            k for f in fields for k in registry.resolve_dependencies(f)
        )
        external_inputs = [
            (kernel, field)
            for kernel in kernels
            for field in registry.get_fields_kernel_require(kernel)
            if field != "weights" and not registry.can_produce_field(field)
        ]
        if not external_inputs:
            return
        stored = selection.list_fields_by_uid()
        for kernel, field in external_inputs:
            missing = {uid for uid, names in stored.items() if field not in names}
            if not missing:
                continue
            level = registry.get_kernel_apply_level(kernel)
            if level == KernelApplyLevel.CROSS_MODEL:
                members = {
                    uid
                    for group, _ in aggregate_groups(selection, level, pairs=pairs)
                    if pairs is not None
                    or registry.get_kernel_symmetry(kernel) is None
                    or len(group.coordinates) == _BINARY_MODEL_COUNT
                    for uid in group.member_uids
                }
                missing.intersection_update(members)
                if not missing:
                    continue
            field_lists = (stored[uid] for uid in missing)
            available = sorted(set(next(field_lists)).intersection(*field_lists))
            candidates = set(registry.list_fields_can_produce()) - set(
                registry.get_fields_kernel_produce(kernel)
            )
            hint = did_you_mean(field, candidates.union(available))
            raise FieldNotFoundError(
                f"Kernel '{kernel}' requires input '{field}'. "
                f"Selected parameters missing this input: {len(missing)}.{hint} "
                f"Available stored inputs on these parameters: "
                f"{', '.join(available) or '(none)'}. "
                "Use 'weights' for the parameter's weight matrix. "
                "For other inputs, register a producing kernel or attach values "
                "with session.results.ingest_metrics(). See "
                "session.compute.list_available_metrics() for computable fields."
            )

    def _validate_scope(
        self, fields_to_produce: tuple[str, ...], pending: IParameterView
    ) -> None:
        """Reject a scope incompatible with a required kernel's apply level.

        A field that depends on a binary kernel requires a scope containing
        exactly two model/checkpoint coordinates.
        """
        offenders = [
            (field_name, kernel_name)
            for field_name in fields_to_produce
            for kernel_name in self.__kernel_registry.resolve_dependencies(field_name)
            if self._is_binary_cross_model(kernel_name)
        ]
        if not offenders:
            return

        endpoints = sorted({(p.meta.model_id, p.meta.step) for p in pending})
        if len(endpoints) == _BINARY_MODEL_COUNT:
            return

        field_name, kernel_name = offenders[0]
        raise ScopeValidationError(
            _binary_scope_message(field_name, kernel_name, endpoints)
        )

    def _is_binary_cross_model(self, kernel_name: str) -> bool:
        """True if the kernel is a binary cross-model kernel."""
        if (
            self.__kernel_registry.get_kernel_apply_level(kernel_name)
            != KernelApplyLevel.CROSS_MODEL
        ):
            return False
        restrictions = self.__kernel_registry.get_kernel_restrictions(kernel_name)
        return bool(restrictions and (restrictions & KernelRestrictions.BINARY))

    def list_available_kernels(self, verbose: bool = False) -> list[str]:
        """List all available computational kernels.

        Args:
            verbose: If True, include detailed kernel information including
                dependencies and field requirements.

        Returns:
            List of kernel names or detailed kernel descriptions.
        """
        with self.__session_or_context:
            return self.__kernel_registry.list_kernels(verbose=verbose)

    def describe_field(self, name: str) -> dict[str, Any]:
        """Describe a registered field's kernel, dependencies, and export route."""
        with self.__session_or_context:
            registry = self.__kernel_registry
            if not registry.can_produce_field(name):
                hint = did_you_mean(name, registry.list_fields_can_produce())
                raise KernelNotFoundError(
                    f"No registered kernel produces {name!r}.{hint}"
                )
            kernel = registry.get_kernel_producing_field(name)
            level = registry.get_kernel_apply_level(kernel)
            return {
                "kernel": kernel,
                "level": level.name,
                "exporter": (
                    "export_metrics"
                    if level == KernelApplyLevel.PARAMETER
                    else "export_aggregates"
                ),
                "requires": registry.get_fields_kernel_require(kernel),
                "produces": registry.get_fields_kernel_produce(kernel),
                "config": registry.get_kernel_config(kernel),
            }

    def list_available_metrics(self, verbose: bool = False) -> list[str]:
        """List all fields that can be computed by registered kernels.

        Args:
            verbose: If True, include detailed field information including
                producing kernels and dependencies.

        Returns:
            List of computable field names or detailed field descriptions.
        """
        with self.__session_or_context:
            return self.__kernel_registry.list_fields_can_produce(verbose=verbose)

    def configure_kernel(self, kernel_name: str, **config: Any) -> KernelConfig:
        """Configure parameters for a specific kernel.

        Updates the configuration for a registered kernel, affecting
        subsequent computations using that kernel.

        Args:
            kernel_name: Name of kernel to configure.
            **config: Kernel configuration parameters as keyword arguments.

        Returns:
            The effective KernelConfig applied to the kernel.

        Raises:
            KernelNotFoundError: If kernel is not registered in the registry.
        """
        with self.__session_or_context:
            if not self.__kernel_registry.has_kernel(kernel_name):
                hint = did_you_mean(kernel_name, self.__kernel_registry.list_kernels())
                msg = f"Kernel '{kernel_name}' not found in registry.{hint}"
                raise KernelNotFoundError(msg)

            kernel_config = KernelConfig(**config)
            self.__kernel_registry.configure_kernel(kernel_name, kernel_config)

            logger.info("Configured kernel '%s' with: %s", kernel_name, config)

            return kernel_config

    def kernel(
        self,
        _func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        require_fields: tuple[str, ...] | None = None,
        produce_fields: tuple[str, ...] | None = None,
        apply_level: KernelApplyLevel = KernelApplyLevel.PARAMETER,
        execution_protocol: KernelExecutionProtocol = (
            KernelExecutionProtocol.SEQUENTIAL
        ),
        restrictions: KernelRestrictions | None = None,
        version: str = "1",
        output_comparison: OutputComparison | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering custom kernels in the session registry.

        Creates a kernel that can compute metrics on neural network parameters.
        Dependencies are automatically inferred from function arguments.

        Args:
            name: Custom kernel name (defaults to function name).
            require_fields: Input field names (inferred from signature if None).
            produce_fields: Output field names (defaults to kernel name).
            apply_level: Granularity at which the kernel is applied.
            execution_protocol: Execution protocol for the kernel.
            restrictions: Optional restrictions applied to kernel execution.
            version: Change when the kernel behavior or an existing default changes.
            output_comparison: Optional output equivalence and array comparison budget.

        Returns:
            Decorator function for registering the kernel.

        Example:
            >>> session = Session()
            >>> with session:
            ...
            ...     @session.compute.kernel()
            ...     def my_metric(frob_norm: float, *, scale: float = 1.0) -> float:
            ...         return frob_norm * scale
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            req_auto, cfg = self.__kernel_registry._split_signature(func)
            final_name = name or func.__name__
            final_require = require_fields or req_auto
            final_produce = produce_fields or (final_name,)

            self.__kernel_registry.register_kernel(
                name=final_name,
                require_fields=final_require,
                produce_fields=final_produce,
                implementation=func,
                apply_level=apply_level,
                execution_protocol=execution_protocol,
                restrictions=restrictions,
                version=version,
                output_comparison=output_comparison,
                config=cfg,
                info=None,
            )
            return func

        if _func is None:
            return decorator

        return decorator(_func)
