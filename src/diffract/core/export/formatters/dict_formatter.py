"""Dictionary formatter for result export.

Returns structured results with separate scalars and aggregates dictionaries.
"""

from __future__ import annotations

from diffract.core.export.interface import (
    AggregateData,
    IResultFormatter,
    ResultData,
    StructuredExportResult,
)


class DictFormatter(IResultFormatter):
    """Return results as structured dictionaries."""

    def format_results(
        self,
        param_results: ResultData,
        aggregate_results: AggregateData,
        _fields: tuple[str, ...],
    ) -> StructuredExportResult[dict]:
        """Return results as structured dict with scalars and aggregates.

        Args:
            param_results: Parameter results dictionary with scalar fields.
            aggregate_results: Aggregate results list with contextual fields.
            _fields: Field names (unused for dict format).

        Returns:
            StructuredExportResult with scalars dict and aggregates list.
        """
        return StructuredExportResult(
            scalars=param_results,
            aggregates=aggregate_results,
        )
