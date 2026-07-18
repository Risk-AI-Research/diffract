"""Aggregation and grouping helpers extracted from executor."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from diffract.core.data.nn.params.interface import IParameterView

type GroupsIterator = Iterable[tuple[str, IParameterView]]
type ContextBuilder = Callable[[str, IParameterView], AggregationContext]


@dataclass(frozen=True)
class AggregationContext:
    """Structured context of an aggregated computation.

    Attributes:
        models: Model identifiers participating in aggregation.
        parameters: Parameter names participating in aggregation.
    """

    models: tuple[str, ...] | None = None
    parameters: tuple[str, ...] | None = None


def aggregate_parameters(
    collection: IParameterView,
) -> tuple[GroupsIterator, ContextBuilder]:
    """Group parameters by model ID."""
    model_ids = sorted({p.meta.model_id for p in collection})

    def iter_groups() -> GroupsIterator:
        for model_id in model_ids:
            group = collection.filter_by_model_id(model_id)
            yield model_id, group

    def build_context(model_id: str, group: IParameterView) -> AggregationContext:
        names = tuple(sorted(p.meta.name for p in group))
        return AggregationContext(models=(model_id,), parameters=names)

    return iter_groups(), build_context


def aggregate_models(
    collection: IParameterView,
) -> tuple[GroupsIterator, ContextBuilder]:
    """Group parameters by parameter name."""
    param_names = sorted({p.meta.name for p in collection})

    def iter_groups() -> GroupsIterator:
        for param_name in param_names:
            grp = collection.filter_by_name(param_name)
            yield param_name, grp

    def build_context(param_name: str, group: IParameterView) -> AggregationContext:
        model_ids = tuple(sorted(p.meta.model_id for p in group))
        return AggregationContext(models=model_ids, parameters=(param_name,))

    return iter_groups(), build_context
