"""Polars DataFrame formatter for result export.

Creates structured DataFrames separating scalar per-parameter fields from
contextual/aggregation fields. If polars is unavailable, raises ImportError.
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


if not import_utils.is_available("polars"):
    logger.debug("Failed to import polars, disabling corresponding formatters")

    class PolarsFormatter(IResultFormatter):
        """Stub formatter that raises when polars is unavailable."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            msg = "polars package not available"
            raise ImportError(msg)

else:
    pl = import_utils.require("polars")

    class PolarsFormatter(IResultFormatter):
        """Convert results to structured DataFrames separating scalars/aggregates."""

        def format_results(
            self,
            param_results: ResultData,
            aggregate_results: AggregateData,
            fields: tuple[str, ...],
        ) -> StructuredExportResult[pl.DataFrame]:
            """Convert results to structured polars DataFrames.

            Args:
                param_results: Parameter results dictionary with scalar fields.
                aggregate_results: Aggregate results list with contextual fields.
                fields: Field names requested for export.

            Returns:
                StructuredExportResult with scalars and aggregates DataFrames.
            """
            if not param_results and not aggregate_results:
                return StructuredExportResult(
                    scalars=pl.DataFrame({c: [] for c in [*SCALAR_COLUMNS, *fields]}),
                    aggregates=pl.DataFrame({c: [] for c in AGGREGATE_COLUMNS}),
                )

            scalar_records = build_scalar_records(param_results)
            aggregate_records = build_aggregate_records(aggregate_results)

            scalars_df = pl.DataFrame(scalar_records) if scalar_records else pl.DataFrame({c: [] for c in [*SCALAR_COLUMNS, *fields]})
            aggregates_df = pl.DataFrame(aggregate_records) if aggregate_records else pl.DataFrame({c: [] for c in AGGREGATE_COLUMNS})

            return StructuredExportResult(
                scalars=scalars_df,
                aggregates=aggregates_df,
            )
