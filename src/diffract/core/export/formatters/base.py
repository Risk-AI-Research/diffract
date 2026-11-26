"""Base utilities for DataFrame formatters.

Provides shared logic for building scalar and aggregate records.
"""

from __future__ import annotations

from typing import Any

from diffract.core.export.interface import AggregateData, ResultData

COORDINATE_COLUMNS = ("instance_uid", "step", "step_label")
SCALAR_COLUMNS = [
    "model_id",
    "parameter_name",
    "parameter_uid",
    "parameter_type",
    *COORDINATE_COLUMNS,
]
AGGREGATE_COLUMNS = [
    "aggregate_uid",
    "field",
    "context_models",
    "context_params",
    "value",
]


def value_column(field: str) -> str:
    """Escape values that overlap with identity or metadata columns."""
    if field in SCALAR_COLUMNS or field.startswith(("meta_", "field_")):
        return f"field_{field}"
    return field


def build_scalar_records(results: ResultData) -> list[dict[str, Any]]:
    """Build scalar records from parameter results.

    Args:
        results: Parameter results dictionary.

    Returns:
        List of scalar records for DataFrame creation.
    """
    scalar_records: list[dict[str, Any]] = []

    for param_uid, param_data in results.items():
        metadata = param_data["metadata"]
        field_values = param_data["fields"]

        base_meta = {
            "parameter_uid": param_uid,
            "model_id": metadata["model_id"],
            "parameter_name": metadata["name"],
            "parameter_type": metadata["parameter_type"],
            **{key: metadata[key] for key in COORDINATE_COLUMNS if key in metadata},
        }

        scalar_record: dict[str, Any] = {**base_meta}

        for key, value in metadata.items():
            if key not in {"model_id", "name", "parameter_type", *COORDINATE_COLUMNS}:
                scalar_record[f"meta_{key}"] = value

        scalar_record.update((value_column(k), v) for k, v in field_values.items())

        # Only add record if it has data beyond base metadata
        if len(scalar_record) > len(base_meta):
            scalar_records.append(scalar_record)

    return scalar_records


def build_aggregate_records(aggregates: AggregateData) -> list[dict[str, Any]]:
    """Build aggregate records from aggregate results.

    Args:
        aggregates: Aggregate results list.

    Returns:
        List of aggregate records for DataFrame creation.
    """
    # Aggregates are already in the right format, just return them
    return list(aggregates)


def scalar_frame_columns(
    results: ResultData, fields: tuple[str, ...]
) -> dict[str, list[Any]]:
    """Order identity, known metadata, and metric columns for either DataFrame."""
    records = build_scalar_records(results)
    metadata = sorted(
        {key for row in records for key in row if key.startswith("meta_")}
    )
    expanded = {
        context["field"]
        for entry in results.values()
        for context in entry.get("field_contexts", {}).values()
    }
    metric_columns = dict.fromkeys(value_column(f) for f in fields if f not in expanded)
    excluded = set(SCALAR_COLUMNS) | set(metadata)
    for row in records:
        metric_columns.update(dict.fromkeys(key for key in row if key not in excluded))
    return {
        key: [row.get(key) for row in records]
        for key in (*SCALAR_COLUMNS, *metadata, *metric_columns)
    }
