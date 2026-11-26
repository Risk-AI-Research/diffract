"""JSON formatter for result export.

Serializes structured results to a JSON string with a custom serializer that
converts array-like objects (e.g., numpy arrays, tensors) via tolist()
when available, and falls back to string representation otherwise.
"""

from __future__ import annotations

import json
from typing import Any

from diffract.core.export.interface import (
    AggregateData,
    IResultFormatter,
    ResultData,
)


class JsonFormatter(IResultFormatter):
    """Convert results to a pretty-printed JSON string."""

    def format_results(
        self,
        param_results: ResultData,
        aggregate_results: AggregateData,
        fields: tuple[str, ...],  # noqa: ARG002 - formatter interface
    ) -> str:
        """Convert results to formatted JSON string.

        Args:
            param_results: Parameter results dictionary with scalar fields.
            aggregate_results: Aggregate results list with contextual fields.
            fields: Field names (unused for JSON format).

        Returns:
            JSON string representation with scalars and aggregates.
        """
        combined = {
            "scalars": param_results,
            "aggregates": aggregate_results,
        }
        return dump_json(combined)


def dump_json(data: Any) -> str:
    """Serialize result data and convert array values to nested lists."""

    def json_value(obj: Any) -> object:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    return json.dumps(data, indent=2, default=json_value)
