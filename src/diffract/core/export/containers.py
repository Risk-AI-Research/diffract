"""Dependency injection container for exporting results.

Provides providers for selecting a formatter and exporting results using
ResultExporter. The default export format is read from config.default_export_format.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dependency_injector import containers, providers

from .exporters import ResultExporter
from .formatters.registry import get_formatter


def _export_with_formatter(
    *args: Any,
    exporter: ResultExporter,
    export_format: str,
    formatter_factory: Callable[[str], Any],
    **kwargs: Any,
) -> Any:
    """Export results using dynamically selected formatter."""
    return exporter.export_results(
        *args,
        formatter=formatter_factory(export_format),
        **kwargs,
    )


class ExportContainer(containers.DeclarativeContainer):
    """Container for export-related dependencies."""

    config = providers.Configuration()

    result_formatter = providers.Factory(get_formatter)

    result_exporter = providers.Singleton(ResultExporter)

    export_results = providers.Factory(
        _export_with_formatter,
        exporter=result_exporter,
        export_format=config.default_export_format,
        formatter_factory=result_formatter.provider,
    )
