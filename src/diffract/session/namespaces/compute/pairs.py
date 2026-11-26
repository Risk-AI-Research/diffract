"""Resolve ordered checkpoint joins and their field-specific measurement scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diffract.core.compute.execution.enums import KernelRestrictions
from diffract.core.data.identity import training_step
from diffract.core.data.nn.aggregates.identity import PAIR_SIZE, aggregate_identity
from diffract.session.errors import ScopeValidationError
from diffract.session.summaries import PairCoverage, PairMismatch, ShapeMismatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from diffract.core.compute import KernelRegistry
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.extractors.report import ExtractionReport
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.metadata import ParameterMetadata

type Checkpoint = str | tuple[str, int]
type Coordinate = tuple[str, int]


def shape_restrictions(registry: KernelRegistry, field: str) -> KernelRestrictions:
    """Combine dimension requirements across the field's dependency chain."""
    flags = KernelRestrictions(0)
    for kernel in registry.resolve_dependencies(field):
        flags |= registry.get_kernel_restrictions(kernel) or KernelRestrictions(0)
    return flags & (
        KernelRestrictions.MATCH_ROWS
        | KernelRestrictions.MATCH_COLUMNS
        | KernelRestrictions.MATCH_LOWER_DIM
    )


def _incompatible(
    left: Sequence[int], right: Sequence[int], flags: KernelRestrictions
) -> bool:
    return (
        (bool(flags & KernelRestrictions.MATCH_ROWS) and left[0] != right[0])
        or (bool(flags & KernelRestrictions.MATCH_COLUMNS) and left[1] != right[1])
        or (
            bool(flags & KernelRestrictions.MATCH_LOWER_DIM) and min(left) != min(right)
        )
    )


@dataclass
class PairPlan:
    """Eligible names and admission outcomes for one ordered pair."""

    matched: tuple[str, ...]
    eligible: dict[str, tuple[str, ...]]
    incompatible: dict[str, tuple[ShapeMismatch, ...]]
    source_extraction: ExtractionReport | None
    target_extraction: ExtractionReport | None

    def batches(self) -> dict[tuple[str, ...], list[str]]:
        """Group fields that can share one execution scope."""
        groups: dict[tuple[str, ...], list[str]] = {}
        for field, names in self.eligible.items():
            if names:
                groups.setdefault(names, []).append(field)
        return groups


@dataclass
class ResolvedPair:
    """Parameter identities at two resolved checkpoints in the active scope."""

    coordinates: tuple[Coordinate, Coordinate]
    join: PairMismatch
    left: dict[str, ParameterMetadata]
    right: dict[str, ParameterMetadata]

    def members(self, names: Iterable[str]) -> list[str]:
        """List source and target parameter identities for the selected names."""
        return [side[name].uid for name in names for side in (self.left, self.right)]

    def _mismatches(
        self,
        names: Iterable[str],
        flags: KernelRestrictions,
        shape: Callable[[str], Sequence[int] | None],
    ) -> Iterable[ShapeMismatch]:
        if flags:
            for name in names:
                left, right = shape(self.left[name].uid), shape(self.right[name].uid)
                if left and right and _incompatible(left, right, flags):
                    yield ShapeMismatch(name, tuple(left), tuple(right))

    def plan(
        self,
        restrictions: dict[str, KernelRestrictions],
        required: frozenset[str],
        shape: Callable[[str], Sequence[int] | None],
        admissions: Callable[[str, int], ExtractionReport | None] | None,
        *,
        skip_incompatible: bool,
    ) -> PairPlan:
        """Validate required names and determine eligible members for each field."""
        source = admissions(*self.coordinates[0]) if admissions else None
        target = admissions(*self.coordinates[1]) if admissions else None
        for role, coordinate, side, report in (
            ("source", self.join.source, self.left, source),
            ("target", self.join.target, self.right, target),
        ):
            missing = sorted(required.difference(side))
            if missing:
                reasons = (
                    {i.name: i.reason for i in report.items if i.reason}
                    if report
                    else {}
                )
                detail = ", ".join(
                    f"{name!r} ({reasons.get(name, 'absent from the stored scope')})"
                    for name in missing
                )
                raise ScopeValidationError(
                    f"Required names missing from {role} {coordinate}: {detail}."
                )

        matched = tuple(sorted(self.left.keys() & self.right.keys()))
        eligible = {}
        mismatches = {}
        for field, flags in restrictions.items():
            items = tuple(self._mismatches(matched, flags, shape))
            for item in items:
                if not skip_incompatible or item.name in required:
                    raise ScopeValidationError(
                        f"{field!r} cannot compare {item.name!r}: "
                        f"source {self.join.source} has shape {item.source_shape}, "
                        f"target {self.join.target} "
                        f"has shape {item.target_shape}."
                    )
            excluded = {item.name for item in items}
            eligible[field] = tuple(name for name in matched if name not in excluded)
            mismatches[field] = items
        return PairPlan(matched, eligible, mismatches, source, target)

    def report(
        self, plan: PairPlan, registry: KernelRegistry, aggregates: AggregateRepository
    ) -> PairCoverage:
        """Resolve eligible names to stored measurements in the requested direction."""
        measured = {}
        for field, names in plan.eligible.items():
            kernel = registry.get_kernel_producing_field(field)
            directed = registry.get_kernel_symmetry(kernel) is False
            present = []
            for name in names:
                uid = aggregate_identity(
                    field, self.members((name,)), directed=directed
                ).uid
                if aggregates.has_uid(uid) and aggregates.get_proxy(uid).has_field(
                    "value"
                ):
                    present.append(name)
            measured[field] = tuple(present)
        return PairCoverage(
            source=self.join.source,
            target=self.join.target,
            source_only=self.join.source_only,
            target_only=self.join.target_only,
            matched=plan.matched,
            source_extraction=plan.source_extraction,
            target_extraction=plan.target_extraction,
            incompatible=plan.incompatible,
            measured=measured,
        )


def resolve_pairs(
    scope: IParameterView,
    pairs: Iterable[tuple[Checkpoint, Checkpoint]],
) -> tuple[IParameterView, tuple[ResolvedPair, ...]]:
    """Resolve and deduplicate ordered pairs within the active parameter view."""
    by_name: dict[str, dict[int, dict[str, ParameterMetadata]]] = {}
    for param in scope:
        meta = param.meta
        by_name.setdefault(meta.model_id, {}).setdefault(meta.step, {})[meta.name] = (
            meta
        )

    def endpoint(
        value: Checkpoint,
    ) -> tuple[str, int, dict[str, ParameterMetadata]]:
        if isinstance(value, str):
            if not (steps := by_name.get(value)):
                raise ScopeValidationError(
                    f"Unknown model in the current scope: {value!r}."
                )
            return value, max(steps), steps[max(steps)]
        if (
            not isinstance(value, tuple)
            or len(value) != PAIR_SIZE
            or not isinstance(value[0], str)
        ):
            raise ScopeValidationError(
                "Each endpoint must be a model name or a (model, step) tuple."
            )
        name, step = value[0], training_step(value[1])
        params = by_name.get(name, {}).get(step)
        if not params:
            raise ScopeValidationError(
                f"Checkpoint {(name, step)!r} is outside the current scope."
            )
        return name, step, params

    task_pairs = {}
    owner_uids: dict[str, None] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != PAIR_SIZE:
            raise ScopeValidationError("Each pair must be a (source, target) tuple.")
        source, source_step, left = endpoint(pair[0])
        target, target_step, right = endpoint(pair[1])
        left_key = (next(iter(left.values())).instance_uid, source_step)
        right_key = (next(iter(right.values())).instance_uid, target_step)
        if left_key == right_key:
            raise ScopeValidationError("A pair must contain two distinct checkpoints.")
        coordinates = left_key, right_key
        if coordinates in task_pairs:
            continue
        for member in (*left.values(), *right.values()):
            owner_uids[member.uid] = None
        source_only = tuple(sorted(left.keys() - right.keys()))
        target_only = tuple(sorted(right.keys() - left.keys()))
        join = PairMismatch(
            (source, source_step), (target, target_step), source_only, target_only
        )
        task_pairs[coordinates] = ResolvedPair(coordinates, join, left, right)
    if not task_pairs:
        raise ScopeValidationError("Supply at least one checkpoint pair.")
    return scope[list(owner_uids)], tuple(task_pairs.values())
