from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypedDict

from .ordering import as_is

if TYPE_CHECKING:
    from .ordering import Ordering


class Entry(TypedDict, total=False):
    """Single data entry from session results."""

    fields: dict[str, Any]


class DataType(Enum):
    """Whether a field's values are numeric or categorical."""

    NUMERIC = auto()
    CATEGORICAL = auto()

    @classmethod
    def from_string(cls, name: str) -> DataType:
        """Return the DataType matching the given name (case-insensitive)."""
        match name.lower():
            case "numeric":
                return DataType.NUMERIC
            case "categorical":
                return DataType.CATEGORICAL
            case _:
                raise ValueError(f"Unknown data type: {name!r}")


class DataShape(Enum):
    """Whether a field holds a scalar or a vector per entry."""

    SCALAR = auto()
    VECTOR = auto()

    @classmethod
    def from_string(cls, name: str) -> DataShape:
        """Return the DataShape matching the given name (case-insensitive)."""
        match name.lower():
            case "scalar":
                return DataShape.SCALAR
            case "vector":
                return DataShape.VECTOR
            case _:
                raise ValueError(f"Unknown data shape: {name!r}")


@dataclass
class FieldRef:
    """Reference to a data field with optional ordering.

    Attributes:
        field: Name of the field to reference.
        ordering: How to order values of this field. Defaults to AS_IS (no reordering).
    """

    field: str
    data_type: DataType | None = field(default=None)
    data_shape: DataShape | None = field(default=None)
    ordering: Ordering = field(default_factory=as_is)


@dataclass
class FieldMeta:
    """Metadata about a field's type and shape."""

    data_type: DataType
    data_shape: DataShape


@dataclass
class EntryContext:
    """Entry fields plus extracted model and parameter context."""

    fields: dict[str, Any]
    model_id: Any
    parameter_name: Any

    @classmethod
    def from_entry(cls, entry: Entry) -> EntryContext:
        """Create EntryContext from a session entry."""
        entry_fields = entry.get("fields", {})
        if not isinstance(entry_fields, dict):
            raise TypeError(
                f"Entry 'fields' must be a mapping, got {type(entry_fields).__name__}"
            )
        return cls(
            fields=entry_fields,
            model_id=entry_fields.get("model_id"),
            parameter_name=entry_fields.get("name"),
        )
