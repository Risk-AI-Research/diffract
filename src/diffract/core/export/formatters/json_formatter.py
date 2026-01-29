"""JSON formatter for result export.

Serializes structured results to a JSON string with a custom serializer that
converts array-like objects (e.g., numpy arrays, tensors) via tolist()
when available, and falls back to string representation otherwise.
"""

from __future__ import annotations

import json

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
        _fields: tuple[str, ...],
    ) -> str:
        """Convert results to formatted JSON string.

        Args:
            param_results: Parameter results dictionary with scalar fields.
            aggregate_results: Aggregate results list with contextual fields.
            _fields: Field names (unused for JSON format).

        Returns:
            JSON string representation with scalars and aggregates.
        """
        combined = {
            "scalars": param_results,
            "aggregates": aggregate_results,
        }
        return json.dumps(combined, indent=2, default=self._json_serializer)

    def _json_serializer(self, obj: object) -> object:
        """Custom JSON serializer for non-native types.

        Uses tolist() for array-like structures to preserve value structure.
        Falls back to str() for other objects.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of the object.
        """
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)
