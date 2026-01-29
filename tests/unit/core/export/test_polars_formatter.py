"""Tests for the optional polars export formatter."""

from __future__ import annotations

import importlib

import pytest


def test_polars_formatter_does_not_require_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    pl = pytest.importorskip("polars")

    real_import_module = importlib.import_module

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas":
            raise ImportError("pandas import blocked")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    from diffract.core.export.formatters.polars_formatter import PolarsFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PolarsFormatter()

    param_results = {
        "uid-1": {
            "metadata": {
                "name": "layer.weight",
                "model_id": "model-1",
                "parameter_type": "WEIGHT",
            },
            "fields": {"mean": 0.5},
        },
    }
    aggregate_results: list = []

    export = formatter.format_results(param_results, aggregate_results, ("mean", "std"))
    assert isinstance(export, StructuredExportResult)
    assert isinstance(export.scalars, pl.DataFrame)
    assert isinstance(export.aggregates, pl.DataFrame)

    df = export.scalars
    assert df.height == 1
    assert set(df.columns) >= {
        "parameter_uid",
        "model_id",
        "parameter_name",
        "parameter_type",
        "mean",
    }
    assert df.select("parameter_uid").item() == "uid-1"
    assert df.select("mean").item() == 0.5

    assert export.aggregates.height == 0


def test_polars_formatter_empty_results_schema() -> None:
    pl = pytest.importorskip("polars")

    from diffract.core.export.formatters.polars_formatter import PolarsFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PolarsFormatter()
    export = formatter.format_results({}, [], ("mean",))

    assert isinstance(export, StructuredExportResult)
    assert isinstance(export.scalars, pl.DataFrame)
    assert export.scalars.height == 0
    assert set(export.scalars.columns) == {"model_id", "parameter_name", "parameter_uid", "parameter_type", "mean"}

    assert isinstance(export.aggregates, pl.DataFrame)
    assert export.aggregates.height == 0


def test_polars_formatter_with_aggregates() -> None:
    """Test that aggregates are correctly formatted into DataFrame."""
    pl = pytest.importorskip("polars")

    from diffract.core.export.formatters.polars_formatter import PolarsFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PolarsFormatter()

    param_results = {
        "uid-1": {
            "metadata": {
                "name": "layer.weight",
                "model_id": "model-A",
                "parameter_type": "WEIGHT",
            },
            "fields": {"frob_norm": 1.5},
        },
        "uid-2": {
            "metadata": {
                "name": "layer.weight",
                "model_id": "model-B",
                "parameter_type": "WEIGHT",
            },
            "fields": {"frob_norm": 2.0},
        },
    }

    aggregate_results = [
        {
            "field": "l_overlap",
            "context_models": ("model-A", "model-B"),
            "context_params": ("layer.weight",),
            "value": [[0.9, 0.1]],
        },
    ]

    export = formatter.format_results(param_results, aggregate_results, ("frob_norm", "l_overlap"))
    assert isinstance(export, StructuredExportResult)

    # Scalars should have frob_norm
    scalars = export.scalars
    assert scalars.height == 2
    assert "frob_norm" in scalars.columns

    # Aggregates should have l_overlap
    aggregates = export.aggregates
    assert aggregates.height == 1
    assert set(aggregates.columns) >= {
        "field",
        "context_models",
        "context_params",
        "value",
    }
    assert aggregates.select("field").item() == "l_overlap"
