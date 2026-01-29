"""Domain schema definitions for diffract.core.data.nn.

This module centralizes NN-domain type aliases and enumerations to keep
interfaces explicit and avoid ambiguous bare primitives like str/int in
public APIs. It defines the core domain schema including parameter types,
identifiers, and field names.
"""

from __future__ import annotations

from enum import IntFlag, auto

# Domain primitives.
type ParameterUID = str
"""Unique identifier of a parameter in storage."""

type ParameterIndex = int
"""Zero-based positional index within a view or list-like ordering."""

type FieldName = str
"""Field name for parameter-associated data stored in backends (e.g., "weights")."""

type ModelID = str
"""Unique identifier of a model containing parameters."""


class ParameterType(IntFlag):
    """Parameter type enumeration with extensible design.

    This enumeration supports bitwise operations for flexible parameter
    filtering and type combination. New parameter types can be added
    dynamically through string conversion, making it extensible for
    future parameter classifications.

    The IntFlag base allows combining types using bitwise OR operations
    and checking membership using bitwise AND operations.

    Attributes:
        UNKNOWN: Default type for unclassified parameters.
        DENSE: Dense/fully-connected layer parameters.
    """

    UNKNOWN = 0
    DENSE = auto()

    @classmethod
    def from_string(cls, type_str: str) -> ParameterType:
        """Create parameter type from string representation.

        Attempts to match existing parameter types first, then creates
        new dynamic types for unknown strings. This allows the system
        to adapt to new parameter types without code changes.

        Args:
            type_str: String representation of parameter type.

        Returns:
            Corresponding ParameterType enum value.
        """
        try:
            return cls[type_str.upper()]
        except KeyError:
            # Create dynamic parameter type for new types
            current_max = max(cls._value2member_map_.values())
            new_value = 1 << (current_max.bit_length())
            return cls._create_custom_type(type_str.upper(), new_value)

    @classmethod
    def _create_custom_type(cls, name: str, value: int) -> ParameterType:
        """Create a custom parameter type dynamically."""
        new_member = int.__new__(cls, value)
        new_member._name_ = name
        new_member._value_ = value

        cls._member_map_[name] = new_member
        cls._value2member_map_[value] = new_member

        return new_member

    def __repr__(self) -> str:
        """Return string representation of parameter type."""
        return self.name
