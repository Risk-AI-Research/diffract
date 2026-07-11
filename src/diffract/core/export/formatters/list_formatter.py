"""List formatter for result export.

Returns structured results with scalar records as list-of-dicts and
aggregates as list-of-dicts.
"""

from __future__ import annotations

from diffract.core.export.interface import (
    AggregateData,
    IResultFormatter,
    ResultData,
    StructuredExportResult,
)

from .base import build_aggregate_records, build_scalar_records


class ListFormatter(IResultFormatter):
    """Return results as list-based records."""

    def format_results(
        self,
        param_results: ResultData,
        aggregate_results: AggregateData,
        _fields: tuple[str, ...],
    ) -> StructuredExportResult[list[dict]]:
        """Return results as lists of records.

        Args:
            param_results: Parameter results dictionary with scalar fields.
            aggregate_results: Aggregate results list with contextual fields.
            _fields: Field names (unused for list format).

        Returns:
            StructuredExportResult with scalars list and aggregates list.
        """
        return StructuredExportResult(
            scalars=build_scalar_records(param_results),
            aggregates=build_aggregate_records(aggregate_results),
        )
