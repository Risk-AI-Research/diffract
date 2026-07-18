"""Parameter metadata definition.

This module provides the ParameterMetadata class which serves as an immutable
container for parameter descriptive information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from diffract.core.data.identity import STORAGE_UNSAFE_PATTERN
from diffract.core.utils.hashing import get_unique_id

from .schema import ParameterType


@dataclass(frozen=True, kw_only=True)
class ParameterMetadata:
    """Immutable metadata container for neural network parameters.

    Contains all descriptive information about a parameter including
    identification, classification, and additional metadata. The frozen
    dataclass ensures immutability for safe sharing across components.

    Implements IMetadata protocol for compatibility with generic data layer.

    Attributes:
        uid: Unique identifier automatically generated if not provided.
        name: Human-readable parameter name (e.g., "conv1.weight").
        ptype: Parameter type classification for filtering.
        model_id: Identifier of the model this parameter belongs to.
        other_meta: Additional metadata as key-value pairs.
    """

    uid: str = field(default_factory=get_unique_id)
    name: str
    ptype: ParameterType
    model_id: str
    other_meta: dict[str, Any] = field(default_factory=dict)

    _FORBIDDEN_CHARS = STORAGE_UNSAFE_PATTERN

    def __post_init__(self) -> None:
        """Validate metadata fields contain no forbidden characters."""
        invalid_fields: list[str] = []

        for field_name, value in [
            ("uid", self.uid),
            ("name", self.name),
            ("ptype", self.ptype.name),
            ("model_id", self.model_id),
        ]:
            if self._FORBIDDEN_CHARS.search(value):
                invalid_fields.append(field_name)

        for key in self.other_meta:
            if not isinstance(key, str):
                invalid_fields.append(f"other_meta[{key!r}] (non-string key)")
            elif self._FORBIDDEN_CHARS.search(key):
                invalid_fields.append(f"other_meta[{key!r}]")

        if invalid_fields:
            raise ValueError(
                f"Fields contain invalid characters {self._FORBIDDEN_CHARS.pattern!r}: "
                f"{', '.join(invalid_fields)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary.

        Returns:
            Dictionary representation suitable for storage/reconstruction.
        """
        return {
            "uid": self.uid,
            "name": self.name,
            "ptype": self.ptype.name,  # Always serialize as string name
            "model_id": self.model_id,
            "other_meta": self.other_meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterMetadata:
        """Deserialize metadata from dictionary.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            ParameterMetadata instance.
        """
        ptype_value = data["ptype"]
        if isinstance(ptype_value, str):
            ptype = ParameterType.from_string(ptype_value)
        elif isinstance(ptype_value, ParameterType):
            ptype = ptype_value
        else:
            raise TypeError(
                f"Invalid or unsupported 'ptype' value '{ptype_value!r}' in metadata; "
                "expected str or ParameterType"
            )

        return cls(
            uid=data["uid"],
            name=data["name"],
            ptype=ptype,
            model_id=data["model_id"],
            other_meta=data.get("other_meta", {}),
        )
