"""Pandas DataFrame formatter for result export.

Creates structured DataFrames separating scalar per-parameter fields from
contextual/aggregation fields. If pandas is unavailable, raises ImportError.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import diffract.core.utils.imports as import_utils
from diffract.core.export.interface import (
    AggregateData,
    IResultFormatter,
    ResultData,
    StructuredExportResult,
)

from .base import (
    AGGREGATE_COLUMNS,
    SCALAR_COLUMNS,
    build_aggregate_records,
    build_scalar_records,
)

logger = logging.getLogger(__name__)


if not import_utils.is_available("pandas"):
    logger.debug("Failed to import pandas, disabling corresponding formatters")

    class PandasFormatter(IResultFormatter):
        """Stub formatter that raises when pandas is unavailable."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError because pandas is not installed."""
            msg = "pandas package not available"
            raise ImportError(msg)

else:
    pd = import_utils.require("pandas")

    class PandasFormatter(IResultFormatter):
        """Convert results to structured DataFrames separating scalars/aggregates."""

        def format_results(
            self,
            param_results: ResultData,
            aggregate_results: AggregateData,
            fields: tuple[str, ...],
        ) -> StructuredExportResult[pd.DataFrame]:
            """Convert results to structured pandas DataFrames.

            Args:
                param_results: Parameter results dictionary with scalar fields.
                aggregate_results: Aggregate results list with contextual fields.
                fields: Field names requested for export.

            Returns:
                StructuredExportResult with scalars and aggregates DataFrames.
            """
            if not param_results and not aggregate_results:
                return StructuredExportResult(
                    scalars=pd.DataFrame(columns=[*SCALAR_COLUMNS, *fields]),
                    aggregates=pd.DataFrame(columns=AGGREGATE_COLUMNS),
                )

            scalar_records = build_scalar_records(param_results)
            aggregate_records = build_aggregate_records(aggregate_results)

            scalars_df = (
                pd.DataFrame(scalar_records)
                if scalar_records
                else pd.DataFrame(columns=[*SCALAR_COLUMNS, *fields])
            )
            aggregates_df = (
                pd.DataFrame(aggregate_records)
                if aggregate_records
                else pd.DataFrame(columns=AGGREGATE_COLUMNS)
            )

            return StructuredExportResult(
                scalars=scalars_df,
                aggregates=aggregates_df,
            )
