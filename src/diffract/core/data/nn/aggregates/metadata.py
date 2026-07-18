"""Metadata for aggregate entities.

This module provides the AggregateMetadata class for representing
aggregated data between parameters/models computed by aggregation kernels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Self

from diffract.core.utils.hashing import get_unique_id


@dataclass(frozen=True, kw_only=True)
class AggregateMetadata:
    """Immutable metadata container for aggregate entities.

    An aggregate represents computed data involving multiple models or parameters,
    such as overlap metrics between two models or correlation between parameters.

    Attributes:
        uid: Unique identifier automatically generated if not provided.
        field_name: Base field name of the computed aggregate (e.g., "l_overlap").
        context_models: Tuple of model IDs participating in this aggregate.
        context_params: Tuple of parameter names participating in this aggregate.
    """

    uid: str = field(default_factory=get_unique_id)
    field_name: str
    context_models: tuple[str, ...] = ()
    context_params: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary.

        Returns:
            Dictionary representation suitable for storage/reconstruction.
        """
        return {
            "uid": self.uid,
            "field_name": self.field_name,
            "context_models": json.dumps(list(self.context_models)),
            "context_params": json.dumps(list(self.context_params)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize metadata from dictionary.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            AggregateMetadata instance.
        """
        # Handle both list format (from to_dict) and JSON string format (from index)
        context_models = data.get("context_models", [])
        if isinstance(context_models, str):
            context_models = json.loads(context_models)

        context_params = data.get("context_params", [])
        if isinstance(context_params, str):
            context_params = json.loads(context_params)

        return cls(
            uid=data["uid"],
            field_name=data["field_name"],
            context_models=tuple(context_models),
            context_params=tuple(context_params),
        )

    @classmethod
    def create_uid_from_context(
        cls,
        field_name: str,
        context_models: tuple[str, ...],
        context_params: tuple[str, ...],
    ) -> str:
        """Create the deterministic aggregate uid for a context.

        The single composer of the legacy uid grammar
        (``field@models[m1,m2]@params[p1]``, context members sorted); only
        the session resolver interprets the produced string.

        Args:
            field_name: Base field name of the aggregate.
            context_models: Tuple of model IDs.
            context_params: Tuple of parameter names.

        Returns:
            A deterministic UID string based on the context.
        """
        parts = [field_name]
        if context_models:
            parts.append(f"models[{','.join(sorted(context_models))}]")
        if context_params:
            parts.append(f"params[{','.join(sorted(context_params))}]")
        return "@".join(parts)
