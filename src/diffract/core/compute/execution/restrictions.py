"""Kernel restriction checking utilities."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from .aggregation import AggregationContext
from .enums import KernelRestrictions

logger = logging.getLogger(__name__)

_BINARY_MODEL_COUNT = 2

# An aggregation task id is a (group_id, AggregationContext) pair; the
# arity is unrelated to the binary-kernel model count.
_AGGREGATION_TASK_ARITY = 2


def apply_restrictions_filter(
    kernel_name: str,
    tasks: dict[Hashable, tuple[Any, ...]],
    restrictions: KernelRestrictions | None,
) -> None:
    """Drop tasks that violate kernel restrictions (mutates tasks dict).

    The ``BINARY`` restriction marks a cross-model kernel that operates on a
    model pair. A group spanning any other number of models is dropped with a
    ``WARNING`` -- never silently -- so a mis-scoped binary kernel cannot no-op
    unnoticed (the caller pre-validates scope and raises before reaching here;
    this is the defense-in-depth backstop).

    Args:
        kernel_name: Name of the kernel for logging.
        tasks: Dictionary of task_id -> args to filter in place. Aggregation
            task ids are ``(group_id, AggregationContext)``; the model count is
            read from the context.
        restrictions: Optional kernel restrictions to check.
    """
    if not restrictions or not (restrictions & KernelRestrictions.BINARY):
        return

    to_remove: list[Hashable] = []
    for task_id in tasks:
        context = _task_context(task_id)
        if context is None:
            # BINARY is a cross-model restriction; there is nothing to enforce
            # off the aggregation path.
            continue

        model_count = len(context.models or ())
        if model_count != _BINARY_MODEL_COUNT:
            logger.warning(
                "Skipping kernel '%s' for group '%s': a binary cross-model kernel "
                "requires exactly two models, but the group spans %d: %s",
                kernel_name,
                _task_label(task_id),
                model_count,
                list(context.models or ()),
            )
            to_remove.append(task_id)

    for task_id in to_remove:
        del tasks[task_id]


def _task_context(task_id: Hashable) -> AggregationContext | None:
    """Extract the aggregation context from a task id, if present."""
    if isinstance(task_id, tuple) and len(task_id) == _AGGREGATION_TASK_ARITY:
        candidate = task_id[1]
        if isinstance(candidate, AggregationContext):
            return candidate
    return None


def _task_label(task_id: Hashable) -> Any:
    """Human-readable group label for logging."""
    if isinstance(task_id, tuple) and task_id:
        return task_id[0]
    return task_id
