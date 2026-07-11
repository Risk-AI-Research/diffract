from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OrderMode(Enum):
    """How to order categorical values."""

    AS_IS = auto()
    LEXICOGRAPHIC = auto()
    NUMERIC = auto()
    MAP = auto()
    CUSTOM = auto()

    @classmethod
    def from_string(cls, name: str) -> OrderMode:
        """Return the OrderMode matching the given name (case-insensitive)."""
        match name.lower():
            case "as_is":
                return cls.AS_IS
            case "lexicographic":
                return cls.LEXICOGRAPHIC
            case "numeric":
                return cls.NUMERIC
            case "map":
                return cls.MAP
            case "custom":
                return cls.CUSTOM
            case _:
                raise ValueError(f"Unknown order mode: {name!r}")


@dataclass(kw_only=True)
class Ordering:
    """Defines how to order values.

    Returns indices (like np.argsort) so the same ordering
    can be applied to multiple related arrays.
    """

    mode: OrderMode = OrderMode.AS_IS
    descending: bool = False

    # For MAP mode: function to extract comparison key
    key: Callable[[Any], Any] | None = None

    # For CUSTOM mode: explicit order of values
    custom_order: Sequence[Any] | None = None

    def argsort(self, values: Sequence[Any]) -> NDArray[np.intp]:
        """Return indices that would sort unique values.

        Args:
            values: Sequence of values to order.

        Returns:
            Array of indices into unique values that produces the desired order.
            Use np.unique(..., return_inverse=True) to map back to original data.
        """
        unique = list(dict.fromkeys(values))  # preserve first occurrence order

        if self.mode == OrderMode.AS_IS:
            indices = np.arange(len(unique))

        elif self.mode == OrderMode.LEXICOGRAPHIC:
            indices = np.argsort([str(v) for v in unique])

        elif self.mode == OrderMode.NUMERIC:
            indices = np.argsort(
                [float(v) if v is not None else float("-inf") for v in unique]
            )

        elif self.mode == OrderMode.MAP:
            if self.key is None:
                indices = np.arange(len(unique))
            else:
                indices = np.argsort([self.key(v) for v in unique])

        elif True:
            if self.custom_order is None:
                indices = np.arange(len(unique))
            else:
                order_map = {v: i for i, v in enumerate(self.custom_order)}
                max_idx = len(order_map)
                # Values not in custom_order go to the end
                priorities = [order_map.get(v, max_idx) for v in unique]
                indices = np.argsort(priorities)

        else:
            indices = np.arange(len(unique))

        if self.descending:
            indices = indices[::-1].copy()

        return indices


# Convenience constructors


def as_is() -> Ordering:
    """Keep original order from data."""
    return Ordering(mode=OrderMode.AS_IS)


def lexicographic(descending: bool = False) -> Ordering:
    """Sort alphabetically by string representation."""
    return Ordering(mode=OrderMode.LEXICOGRAPHIC, descending=descending)


def numeric(descending: bool = False) -> Ordering:
    """Sort by numeric value."""
    return Ordering(mode=OrderMode.NUMERIC, descending=descending)


def by_key(key: Callable[[Any], Any], descending: bool = False) -> Ordering:
    """Sort by applying key function to each value."""
    return Ordering(mode=OrderMode.MAP, key=key, descending=descending)


def custom(order: Sequence[Any], descending: bool = False) -> Ordering:
    """Sort by explicit order. Values not in order go to the end."""
    return Ordering(mode=OrderMode.CUSTOM, custom_order=order, descending=descending)
