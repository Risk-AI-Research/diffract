"""Build aggregate tasks from persisted model and checkpoint coordinates."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

from .enums import KernelApplyLevel

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.metadata import ParameterMetadata

type CoordinatePairs = tuple[tuple[tuple[str, int], tuple[str, int]], ...]


@dataclass(frozen=True, slots=True)
class AggregationContext:
    """Parameter members in operand order and their distinct training coordinates."""

    coordinates: tuple[tuple[str, int], ...]
    member_uids: tuple[str, ...]
    directed: bool = False


def aggregate_groups(
    collection: IParameterView,
    level: KernelApplyLevel,
    *,
    pairs: CoordinatePairs | None = None,
) -> Iterator[tuple[AggregationContext, IParameterView]]:
    """Group each checkpoint's parameters or each tensor name across checkpoints."""

    def checkpoint_sort_key(meta: ParameterMetadata) -> tuple[int, str]:
        return meta.step, meta.instance_uid

    rows = [p.meta for p in collection]
    if level == KernelApplyLevel.CROSS_MODEL and pairs is not None:
        by_coordinate: dict[tuple[str, int], dict[str, str]] = {}
        for row in rows:
            by_coordinate.setdefault((row.instance_uid, row.step), {})[row.name] = (
                row.uid
            )
        for source, target in pairs:
            left, right = by_coordinate[source], by_coordinate[target]
            for name in sorted(left.keys() & right.keys()):
                pair_uids = (left[name], right[name])
                yield (
                    AggregationContext((source, target), pair_uids),
                    collection[list(pair_uids)],
                )
        return

    if level == KernelApplyLevel.IN_MODEL:
        rows.sort(key=lambda m: (checkpoint_sort_key(m), m.name, m.uid))
        groups: Iterator[tuple[object, Iterator[ParameterMetadata]]] = groupby(
            rows, checkpoint_sort_key
        )
    elif level == KernelApplyLevel.CROSS_MODEL:
        rows.sort(key=lambda m: (m.name, checkpoint_sort_key(m), m.uid))
        groups = groupby(rows, lambda m: m.name)
    else:
        raise ValueError(f"Unsupported aggregation level: {level}")

    for _, group in groups:
        members = list(group)
        endpoints = tuple(dict.fromkeys((m.instance_uid, m.step) for m in members))
        uids = tuple(m.uid for m in members)
        yield AggregationContext(endpoints, uids), collection[list(uids)]
