"""Structured results returned by the mutating session verbs."""

from __future__ import annotations

from dataclasses import dataclass


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
    """

    parameter_fields: tuple[str, ...] = ()
    in_model_fields: tuple[str, ...] = ()
    cross_model_fields: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EraseSummary:
    """What an erase call removed.

    ``models.erase`` populates ``models``; ``results.erase`` populates
    ``fields``. ``affected_uids`` counts the parameter entries that lost
    data: entries removed by ``models.erase``, entries holding at least one
    erased field for ``results.erase``.

    Attributes:
        models: Model ids erased (``models.erase``).
        fields: Field names erased (``results.erase``).
        affected_uids: Number of parameter entries that lost data.
    """

    models: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    affected_uids: int = 0
