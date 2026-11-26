"""Coverage of the parameter inputs visited by an extractor."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ExtractionItem:
    """One source path and its extraction outcome, including dtype conversion."""

    name: str
    status: Literal["extracted", "excluded", "unsupported", "failed"]
    reason: str | None = None
    source_dtype: str | None = None
    storage_dtype: str | None = None
    stored_name: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Per-input outcomes for one checkpoint, without retaining its tensors."""

    items: tuple[ExtractionItem, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Count weight candidates and each outcome. Exclusions are non-candidates."""
        totals = Counter(item.status for item in self.items)
        return {
            "candidates": len(self.items) - totals["excluded"],
            **{
                s: totals[s] for s in ("extracted", "excluded", "unsupported", "failed")
            },
        }
