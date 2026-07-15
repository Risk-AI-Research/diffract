"""Dependency injection container for exporting results.

Provides the ResultExporter singleton and the ``default_export_format`` config
value, which the results namespace applies when a call does not name a format.
"""

from __future__ import annotations

from dependency_injector import containers, providers

from .exporters import ResultExporter


class ExportContainer(containers.DeclarativeContainer):
    """Container for export-related dependencies."""

    config = providers.Configuration()

    result_exporter = providers.Singleton(ResultExporter)
