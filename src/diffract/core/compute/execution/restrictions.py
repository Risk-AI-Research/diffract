"""Kernel restriction checking utilities."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from diffract.core.data.nn.aggregates.identity import PAIR_SIZE

from .aggregation import AggregationContext
from .enums import KernelRestrictions

logger = logging.getLogger(__name__)


def apply_restrictions_filter[Key: Hashable](
    kernel_name: str,
    tasks: dict[Key, tuple[Any, ...]],
    restrictions: KernelRestrictions | None,
) -> None:
    """Drop binary groups that span fewer or more than two training coordinates."""
    if not restrictions or not (restrictions & KernelRestrictions.BINARY):
        return

    for task in list(tasks):
        if isinstance(task, AggregationContext) and len(task.coordinates) != PAIR_SIZE:
            logger.warning(
                "Skipping kernel '%s': a binary cross-model kernel requires "
                "two coordinates, found %s",
                kernel_name,
                task.coordinates,
            )
            del tasks[task]
