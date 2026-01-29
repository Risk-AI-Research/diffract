"""Export and formatting utilities for computation results.

This package provides a unified interface to export results from parameter
collections into multiple output formats using pluggable formatters.

Features:
    - Single entry point via ResultExporter
    - Protocols for formatters and exporters
    - DI container wiring for default formatter selection

Example:
    >>> from diffract.core.export import ResultExporter
    >>> exporter = ResultExporter()
    >>> out = exporter.export_results(
    ...     "weights", parameters=collection, formatter=formatter
    ... )
"""

from .exporters import ResultExporter
from .interface import IResultExporter, IResultFormatter, StructuredExportResult

__all__ = [
    "IResultExporter",
    "IResultFormatter",
    "ResultExporter",
    "StructuredExportResult",
]
