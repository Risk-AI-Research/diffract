"""Tests for the optional pandas export formatter."""

from __future__ import annotations

import pytest


def test_pandas_formatter_scalar_fields() -> None:
    """Test that scalar fields are correctly formatted."""
    pd = pytest.importorskip("pandas")

    from diffract.core.export.formatters.pandas_formatter import PandasFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PandasFormatter()

    param_results = {
        "uid-1": {
            "metadata": {
                "name": "layer.weight",
                "model_id": "model-1",
                "parameter_type": "WEIGHT",
            },
            "fields": {"mean": 0.5, "std": 0.1},
        },
    }
    aggregate_results: list = []

    export = formatter.format_results(param_results, aggregate_results, ("mean", "std"))
    assert isinstance(export, StructuredExportResult)
    assert isinstance(export.scalars, pd.DataFrame)
    assert isinstance(export.aggregates, pd.DataFrame)

    df = export.scalars
    assert len(df) == 1
    assert set(df.columns) >= {
        "parameter_uid",
        "model_id",
        "parameter_name",
        "parameter_type",
        "mean",
        "std",
    }
    assert df["parameter_uid"].iloc[0] == "uid-1"
    assert df["mean"].iloc[0] == 0.5
    assert df["std"].iloc[0] == 0.1

    assert len(export.aggregates) == 0


def test_pandas_formatter_empty_results_schema() -> None:
    """Test that empty results return proper schema."""
    pd = pytest.importorskip("pandas")

    from diffract.core.export.formatters.pandas_formatter import PandasFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PandasFormatter()
    export = formatter.format_results({}, [], ("mean",))

    assert isinstance(export, StructuredExportResult)
    assert isinstance(export.scalars, pd.DataFrame)
    assert len(export.scalars) == 0
    assert "mean" in export.scalars.columns

    assert isinstance(export.aggregates, pd.DataFrame)
    assert len(export.aggregates) == 0


def test_pandas_formatter_with_aggregates() -> None:
    """Test that aggregates are correctly formatted into DataFrame."""
    pytest.importorskip("pandas")

    from diffract.core.export.formatters.pandas_formatter import PandasFormatter
    from diffract.core.export.interface import StructuredExportResult

    formatter = PandasFormatter()

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

    export = formatter.format_results(
        param_results, aggregate_results, ("frob_norm", "l_overlap")
    )
    assert isinstance(export, StructuredExportResult)

    # Scalars should have frob_norm
    scalars = export.scalars
    assert len(scalars) == 2
    assert "frob_norm" in scalars.columns

    # Aggregates should have l_overlap
    aggregates = export.aggregates
    assert len(aggregates) == 1
    assert set(aggregates.columns) >= {
        "field",
        "context_models",
        "context_params",
        "value",
    }
    assert aggregates["field"].iloc[0] == "l_overlap"
    assert tuple(aggregates["context_models"].iloc[0]) == ("model-A", "model-B")
    assert tuple(aggregates["context_params"].iloc[0]) == ("layer.weight",)


def test_pandas_formatter_multiple_aggregates() -> None:
    """Test multiple distinct aggregates."""
    pytest.importorskip("pandas")

    from diffract.core.export.formatters.pandas_formatter import PandasFormatter

    formatter = PandasFormatter()

    param_results: dict = {}
    aggregate_results = [
        {
            "field": "l_overlap",
            "context_models": ("model-A", "model-B"),
            "context_params": ("layer.weight",),
            "value": [[0.9]],
        },
        {
            "field": "l_overlap",
            "context_models": ("model-A", "model-C"),
            "context_params": ("layer.weight",),
            "value": [[0.8]],
        },
    ]

    export = formatter.format_results(param_results, aggregate_results, ("l_overlap",))

    # Should have 2 distinct aggregates
    aggregates = export.aggregates
    assert len(aggregates) == 2
    assert set(aggregates["field"]) == {"l_overlap"}
