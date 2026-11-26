"""Structured results returned by the mutating session verbs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffract.core.data.nn.extractors.report import ExtractionReport


@dataclass(frozen=True)
class PairMismatch:
    """Resolved checkpoints and names that had no counterpart in the pair."""

    source: tuple[str, int]
    target: tuple[str, int]
    source_only: tuple[str, ...]
    target_only: tuple[str, ...]


@dataclass(frozen=True)
class ShapeMismatch:
    """Stored matrix shapes incompatible with a requested field."""

    name: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class PairCoverage(PairMismatch):
    """Name join in scope, original admission reports and results by field."""

    matched: tuple[str, ...]
    source_extraction: ExtractionReport | None
    target_extraction: ExtractionReport | None
    incompatible: dict[str, tuple[ShapeMismatch, ...]]
    measured: dict[str, tuple[str, ...]]

    @property
    def source_count(self) -> int:
        """Number of stored source names in the active scope."""
        return len(self.matched) + len(self.source_only)

    @property
    def target_count(self) -> int:
        """Number of stored target names in the active scope."""
        return len(self.matched) + len(self.target_only)

    @property
    def measurement_counts(self) -> dict[str, int]:
        """Count distinct measured aggregates for each field of this ordered pair."""
        return {field: len(names) for field, names in self.measured.items()}


@dataclass(frozen=True)
class ApplySummary:
    """What a ``compute.apply`` call produced, grouped by kernel apply level.

    The groups say which plane now holds each written field, and therefore which
    exporter retrieves it: ``parameter_fields`` come back from
    ``results.export_metrics``, while ``in_model_fields`` and
    ``cross_model_fields`` come back from ``results.export_aggregates``. Fields
    produced as dependencies along the way are included in their group.

    Attributes:
        parameter_fields: Written PARAMETER-level fields.
        in_model_fields: Written IN_MODEL-level fields.
        cross_model_fields: Written CROSS_MODEL-level fields.
        skipped: Requested fields that produced nothing, each paired with a
            short reason (already computed, or not applicable to the scope).
        forced_fields: Fields produced by a kernel run with ``recompute=True``.
        pair_mismatches: Parameters without a same-name counterpart in a pair.
        pair_coverage: One report per distinct ordered pair when requested.
    """

    parameter_fields: tuple[str, ...] = ()
    in_model_fields: tuple[str, ...] = ()
    cross_model_fields: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    forced_fields: tuple[str, ...] = ()
    pair_mismatches: tuple[PairMismatch, ...] = ()
    pair_coverage: tuple[PairCoverage, ...] = ()


@dataclass(frozen=True)
class EraseSummary:
    """What an erase call removed.

    ``models.erase`` populates ``models``; ``results.erase`` populates
    ``fields``. ``affected_uids`` counts the parameter entries that lost
    data: entries removed by ``models.erase``, entries holding at least one
    erased field for ``results.erase``.

    Attributes:
        models: Names of models affected by ``models.erase``.
        fields: Field names erased (``results.erase``).
        affected_uids: Number of parameter entries that lost data.
    """

    models: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    affected_uids: int = 0
