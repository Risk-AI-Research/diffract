"""Internal data types for kernel execution.

This module defines the data structures used internally by kernel runners
for tracking execution state, required inputs, and pending work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterView

    from .aggregation import AggregationContext


@dataclass(frozen=True, slots=True)
class AggregateInput:
    """Description of an aggregate input required by an aggregation kernel.

    Used to track which aggregate values need to be read from the
    AggregateRepository when executing aggregation-level kernels.

    Attributes:
        field_name: Name of the aggregate field to read.
        context_models: Model IDs participating in the aggregation.
        context_params: Parameter names participating in the aggregation.
    """

    field_name: str
    context_models: tuple[str, ...]
    context_params: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RequiredInputs:
    """Required inputs for kernel execution, categorized by source.

    Separates inputs into two categories:
    - parameter_fields: Read directly from parameter proxies
    - aggregate_inputs: Read from the aggregate repository

    Attributes:
        parameter_fields: Field names to read from parameters.
        aggregate_inputs: Aggregate field descriptors for repository lookups.
    """

    parameter_fields: tuple[str, ...]
    aggregate_inputs: tuple[AggregateInput, ...]


@dataclass(frozen=True, slots=True)
class PendingGroup:
    """A group of parameters pending aggregation kernel execution.

    Represents a unit of work for aggregation kernels, containing all
    information needed to execute the kernel for a specific group.

    Attributes:
        group_id: Unique identifier for this group.
        context: Aggregation context with model/parameter scope.
        view: Parameter view containing the group's parameters.
        required_inputs: Inputs needed to execute the kernel.
    """

    group_id: str
    context: AggregationContext
    view: IParameterView
    required_inputs: RequiredInputs
