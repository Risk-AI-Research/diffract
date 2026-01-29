"""Base utilities for DataFrame formatters.

Provides shared logic for building scalar and aggregate records.
"""

from __future__ import annotations

from typing import Any

from diffract.core.export.interface import AggregateData, ResultData

SCALAR_COLUMNS = ["model_id", "parameter_name", "parameter_uid", "parameter_type"]
AGGREGATE_COLUMNS = [
    "field",
    "context_models",
    "context_params",
    "value",
]


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
        }

        scalar_record: dict[str, Any] = {**base_meta}

        for key, value in metadata.items():
            if key not in ["model_id", "name", "parameter_type"]:
                scalar_record[f"meta_{key}"] = value

        for field_name, value in field_values.items():
            scalar_record[field_name] = value

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
