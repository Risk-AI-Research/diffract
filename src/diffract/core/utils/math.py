"""Math utilities."""

from __future__ import annotations

from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    """Return arithmetic mean, or 0.0 for empty iterables."""
    total = 0.0
    n = 0
    for v in values:
        total += float(v)
        n += 1
    return total / n if n > 0 else 0.0
