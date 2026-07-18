"""Requested-field validation for result exports.

Separates the two silent-empty cases of an export request: a name nothing
can account for (a misspelling) raises with suggestions, while a known field
that simply has no values in the current scope is reported with an explicit
warning naming the exporter that serves it. An export never returns an
empty result without saying why.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from diffract.core.compute.execution import KernelApplyLevel
from diffract.session.errors import KernelNotFoundError
from diffract.session.resolver import resolve
from diffract.session.utils import did_you_mean

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.export.interface import AggregateData, ResultData
    from diffract.session.field_cache import SessionFieldCache

logger = logging.getLogger(__name__)


def found_metric_fields(param_results: ResultData) -> set[str]:
    """Requested field names that matched at least one in-scope parameter."""
    found: set[str] = set()
    for entry in param_results.values():
        found.update(entry["fields"])
    return found


def found_aggregate_fields(aggregate_results: AggregateData) -> set[str]:
    """Requested field names that matched at least one in-scope aggregate."""
    return {record["field"] for record in aggregate_results}


def check_export_fields(
    requested: tuple[str, ...],
    *,
    found: set[str],
    registry: KernelRegistry,
    params: Callable[[], IParameterView],
    aggregates: Callable[[], AggregateView | None],
    field_cache: SessionFieldCache | None,
    searched_metrics: bool,
    searched_aggregates: bool,
) -> None:
    """Raise for unexportable names; warn for known fields with no values.

    Runs after collection, so a request fully served by stored values costs
    nothing extra: the providers are only called when a requested name
    matched no value in the current scope.

    Args:
        requested: Field names as passed to the export call.
        found: Requested names that matched at least one stored value.
        registry: Kernel registry answering producibility and apply level.
        params: Lazy provider of the in-scope parameter view.
        aggregates: Lazy provider of the in-scope aggregate view.
        field_cache: Session field cache consulted before rescanning storage.
        searched_metrics: True if per-parameter values were collected.
        searched_aggregates: True if aggregate values were collected.

    Raises:
        KernelNotFoundError: If a requested name is neither producible by a
            registered kernel nor stored in the current scope.
    """
    missing = [name for name in dict.fromkeys(requested) if name not in found]
    if not missing:
        return

    stored_params = _stored_param_field_names(params(), field_cache)
    stored_aggs = _stored_aggregate_field_names(aggregates())

    for name in missing:
        base = resolve(name).field
        if not searched_aggregates and base in stored_aggs:
            logger.warning(
                "Requested field '%s' has aggregate values in the current "
                "scope, which export_metrics() does not return. Use "
                "export_aggregates('%s') or export('%s', sources='all').",
                name,
                base,
                base,
            )
        elif not searched_metrics and name in stored_params:
            logger.warning(
                "Requested field '%s' has per-parameter values in the current "
                "scope, which export_aggregates() does not return. Use "
                "export_metrics('%s') or export('%s', sources='all').",
                name,
                name,
                name,
            )
        elif registry.can_produce_field(base):
            _warn_known_field_without_values(
                name,
                base,
                registry=registry,
                searched_metrics=searched_metrics,
                searched_aggregates=searched_aggregates,
            )
        else:
            candidates = set(registry.list_fields_can_produce())
            candidates.update(stored_params)
            candidates.update(stored_aggs)
            hint = did_you_mean(name, candidates)
            msg = (
                f"Cannot export '{name}': no registered kernel produces it "
                f"and no stored field with that name exists in the current "
                f"scope.{hint} Use session.compute.list_available_metrics() "
                f"to see computable fields."
            )
            raise KernelNotFoundError(msg)


def _warn_known_field_without_values(
    name: str,
    base: str,
    *,
    registry: KernelRegistry,
    searched_metrics: bool,
    searched_aggregates: bool,
) -> None:
    """Warn for a producible field with no stored values in the current scope.

    When only one source was searched and the field's apply level routes its
    values to the other one, the warning names the exporter that serves it.
    """
    level = registry.get_kernel_apply_level(registry.get_kernel_producing_field(base))
    metrics_only = searched_metrics and not searched_aggregates
    aggregates_only = searched_aggregates and not searched_metrics

    if metrics_only and level != KernelApplyLevel.PARAMETER:
        logger.warning(
            "Requested field '%s' has no computed values in the current "
            "scope: it is produced at the %s level and exported as "
            "aggregates. Compute it with session.compute.apply('%s'), then "
            "use export_aggregates('%s') or export('%s', sources='all').",
            name,
            level.name,
            base,
            base,
            base,
        )
    elif aggregates_only and level == KernelApplyLevel.PARAMETER:
        logger.warning(
            "Requested field '%s' has no computed values in the current "
            "scope: it is produced per parameter and exported as metrics. "
            "Compute it with session.compute.apply('%s'), then use "
            "export_metrics('%s') or export('%s', sources='all').",
            name,
            base,
            name,
            name,
        )
    else:
        logger.warning(
            "Requested field '%s' has no computed values in the current "
            "scope. Compute it with session.compute.apply('%s'), or widen "
            "the session filter if it was computed under a narrower scope.",
            name,
            base,
        )


def _stored_param_field_names(
    params: IParameterView, field_cache: SessionFieldCache | None
) -> set[str]:
    """Field names stored on any in-scope parameter, cache-first."""
    cached = None
    if field_cache is not None and field_cache.is_valid:
        cached = field_cache.get()
    if cached is not None:
        uids = params.list_uids()
        if all(uid in cached for uid in uids):
            return {name for uid in uids for name in cached[uid]}
    return {name for names in params.list_fields_by_uid().values() for name in names}


def _stored_aggregate_field_names(aggregates: AggregateView | None) -> set[str]:
    """Field names stored on any in-scope aggregate."""
    if aggregates is None or not aggregates:
        return set()
    return {proxy.meta.field_name for proxy in aggregates}
