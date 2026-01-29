"""Kernel restriction checking utilities."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from .enums import KernelRestrictions

logger = logging.getLogger(__name__)

_BINARY_ARG_COUNT = 2


def apply_restrictions_filter(
    kernel_name: str,
    tasks: dict[Hashable, tuple[Any, ...]],
    restrictions: KernelRestrictions | None,
) -> None:
    """Remove tasks that don't meet kernel restrictions (mutates tasks dict).

    Args:
        kernel_name: Name of the kernel for logging.
        tasks: Dictionary of task_id -> args to filter in place.
        restrictions: Optional kernel restrictions to check.
    """
    if not restrictions:
        return

    to_remove: list[Hashable] = []
    for task_id, args in tasks.items():
        if not _check_restrictions(restrictions, args):
            logger.debug(
                "Skip computation for kernel '%s', task '%s': restriction violated",
                kernel_name,
                task_id,
            )
            to_remove.append(task_id)

    for task_id in to_remove:
        del tasks[task_id]


def _check_restrictions(
    restrictions: KernelRestrictions, args: tuple[Any, ...]
) -> bool:
    """Check if arguments satisfy kernel restrictions.

    Args:
        restrictions: Kernel restrictions flags.
        args: Tuple of arguments to check.

    Returns:
        True if all restrictions are satisfied, False otherwise.
    """
    if restrictions & KernelRestrictions.BINARY:
        for arg in args:
            if hasattr(arg, "__len__") and len(arg) != _BINARY_ARG_COUNT:
                return False
    return True
