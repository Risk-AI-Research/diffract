"""Select reusable output groups from live value traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import partial
from math import isclose, isnan
from numbers import Integral, Real
from typing import TYPE_CHECKING

from diffract.core.compute.config import OutputComparison

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    from diffract.core.compute.config import KernelConfigIdentity
    from diffract.core.data.metadata.runs import ValueTrace
    from diffract.core.data.nn.params.interface import IParameterView

DEFAULT_OUTPUT_RTOL = 1e-7
DEFAULT_OUTPUT_ATOL = 1e-12


class StepDecision(Enum):
    """The work required for one output group."""

    COMPUTE = auto()
    REUSE = auto()
    ADOPT = auto()


@dataclass(frozen=True, slots=True)
class FieldStatus:
    """Freshness of one result at its persistent entity UID."""

    field: str
    uid: str
    status: Literal["fresh", "stale", "never computed"]
    reason: str | None = None


def equal_scalar_outputs(
    previous: object,
    current: object,
    *,
    rtol: float = DEFAULT_OUTPUT_RTOL,
    atol: float = DEFAULT_OUTPUT_ATOL,
) -> bool:
    """Compare real scalar outputs with exact integers and NaN equality."""
    if not isinstance(previous, Real) or not isinstance(current, Real):
        return False
    if isinstance(previous, bool) or isinstance(current, bool):
        return type(previous) is type(current) and previous == current
    if isinstance(previous, Integral) or isinstance(current, Integral):
        return (
            isinstance(previous, Integral)
            and isinstance(current, Integral)
            and (previous == current)
        )
    return (isnan(previous) and isnan(current)) or isclose(
        previous, current, rel_tol=rtol, abs_tol=atol
    )


DEFAULT_OUTPUT_COMPARISON = OutputComparison(equal_scalar_outputs)
EXACT_OUTPUT_COMPARISON = OutputComparison(
    partial(equal_scalar_outputs, rtol=0, atol=0)
)


def comparison_for_scope(
    requested: OutputComparison | None, parameters: IParameterView
) -> OutputComparison:
    """Limit optional array comparisons to the active view's read budget."""
    if requested is None:
        return DEFAULT_OUTPUT_COMPARISON
    if requested.max_array_bytes == 0:
        return requested
    get_budget = getattr(type(parameters), "kernel_comparison_budget", None)
    available = 0 if get_budget is None else get_budget(parameters)
    return replace(requested, max_array_bytes=min(requested.max_array_bytes, available))


def decide_step(
    outputs: Iterable[ValueTrace | None],
    inputs: Iterable[ValueTrace | None],
    config_hash: str,
    *,
    requires_kernel_trace: bool = False,
) -> StepDecision:
    """Require a complete, current generation or an untouched inherited group."""
    produced = tuple(outputs)
    consumed = tuple(inputs)
    if not produced or None in produced or None in consumed:
        return StepDecision.COMPUTE

    newest_input = max(
        (trace.changed_at_seq or 0 for trace in consumed if trace is not None),
        default=0,
    )
    traces = tuple(trace for trace in produced if trace is not None)
    if requires_kernel_trace and any(trace.predates_tracing for trace in traces):
        return StepDecision.COMPUTE
    if newest_input == 0 and all(trace.predates_tracing for trace in traces):
        return StepDecision.ADOPT

    kernel_run = None
    for trace in traces:
        if (
            trace.status != "COMPLETE"
            or trace.built_at_seq is None
            or trace.built_at_seq < newest_input
        ):
            return StepDecision.COMPUTE
        if trace.kind in {"ingest", "merge"}:
            continue
        if (
            trace.kind != "kernel"
            or trace.config_hash != config_hash
            or (kernel_run is not None and trace.run_id != kernel_run)
        ):
            return StepDecision.COMPUTE
        kernel_run = trace.run_id
    return StepDecision.REUSE


def describe_step(
    field: str,
    uid: str,
    target: ValueTrace | None,
    outputs: Iterable[ValueTrace | None],
    inputs: Iterable[ValueTrace | None],
    config: KernelConfigIdentity,
) -> FieldStatus:
    """Explain the apply predicate without executing or adopting a result."""
    if target is None:
        return FieldStatus(field, uid, "never computed")

    produced = tuple(outputs)
    consumed = tuple(inputs)
    decision = decide_step(
        produced,
        consumed,
        config.config_hash,
        requires_kernel_trace=config.requires_kernel_trace,
    )
    if decision is StepDecision.REUSE:
        return FieldStatus(field, uid, "fresh")
    if decision is StepDecision.ADOPT:
        return FieldStatus(field, uid, "fresh", "pre-provenance")

    for trace in produced:
        if (
            trace is not None
            and trace.kind == "kernel"
            and trace.config_hash != config.config_hash
            and trace.config_json is not None
        ):
            before = json.loads(trace.config_json)
            current = json.loads(config.config_json)
            changed = [
                key
                for key in sorted(before.keys() | current.keys())
                if json.dumps(before.get(key), sort_keys=True)
                != json.dumps(current.get(key), sort_keys=True)
                or (key in before) != (key in current)
            ]
            if not changed:
                reason = "kernel version or configuration identity changed"
            elif len(changed) == 1:
                reason = f"config key {changed[0]} changed"
            else:
                reason = f"config keys {', '.join(changed)} changed"
            return FieldStatus(field, uid, "stale", reason)

    newest_input = max(
        (trace.changed_at_seq or 0 for trace in consumed if trace is not None),
        default=0,
    )
    if None in consumed:
        reason = "required input is missing"
    elif None in produced:
        reason = "output group is incomplete"
    elif config.requires_kernel_trace and any(
        t.predates_tracing for t in produced if t is not None
    ):
        reason = "spectral execution settings are unknown"
    elif newest_input and any(
        (t.built_at_seq or 0) < newest_input for t in produced if t is not None
    ):
        reason = f"input changed at run {newest_input}"
    else:
        reason = "output group has no matching completed run"
    return FieldStatus(field, uid, "stale", reason)
