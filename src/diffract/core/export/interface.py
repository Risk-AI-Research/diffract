"""Interfaces (protocols) for result export and formatting.

Defines:
    - IResultFormatter: Converts structured results to a target format.
    - IResultExporter: Extracts requested fields from parameters and aggregates,
      then delegates formatting to an IResultFormatter.

ResultData structure (parameters):
    {
        "<param_uid>": {
            "metadata": { ... },
            "fields": { "field": value, ... }
        },
        ...
    }

AggregateData structure:
    [
        {
            "field": "field_name",
            "context_models": ("model1", "model2"),
            "context_params": ("param1",),
            "value": ...,
        },
        ...
    ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView

ResultData = dict[str, dict[str, Any]]
AggregateData = list[dict[str, Any]]

T = TypeVar("T")


@dataclass
class StructuredExportResult(Generic[T]):
    """Container for structured export with separate scalar and aggregate data.

    Scalars are per-parameter fields (e.g., frob_norm, stable_rank).
    Aggregates are cross-entity fields from aggregation kernels (e.g., l_overlap).

    Attributes:
        scalars: DataFrame/dict with per-parameter metrics indexed by parameter.
        aggregates: DataFrame/dict with aggregation results including context info.
    """

    scalars: T
    aggregates: T


@runtime_checkable
class IResultFormatter(Protocol):
    """Protocol for converting raw results to specific formats."""

    def format_results(
        self,
        param_results: ResultData,
        aggregate_results: AggregateData,
        fields: tuple[str, ...],
    ) -> Any:
        """Convert raw results to the target format.

        Args:
            param_results: Parameter results dictionary with scalar fields.
            aggregate_results: Aggregate results list with contextual fields.
            fields: Field names requested for export.

        Returns:
            Formatted results in the target format (StructuredExportResult, dict, etc.).
        """
        ...


@runtime_checkable
class IResultExporter(Protocol):
    """Protocol for exporting results from parameter and aggregate repositories."""

    def export_results(
        self,
        *fields: tuple[str, ...],
        parameters: IParameterView,
        aggregates: AggregateView | None,
        formatter: IResultFormatter,
    ) -> Any:
        """Export results from parameters and aggregates using the specified formatter.

        Args:
            *fields: Field names to export.
            parameters: Parameter collection to export scalar fields from.
            aggregates: Aggregate view to export contextual fields from.
            formatter: Formatter to use for output conversion.

        Returns:
            Formatted results from the formatter.
        """
        ...
