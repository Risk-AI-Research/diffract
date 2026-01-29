"""Kernel enumeration types and execution constants."""

from __future__ import annotations

from enum import Enum, Flag, auto


class KernelApplyLevel(Enum):
    """Defines the level at which kernel computation is applied.

    PARAMETER: Apply kernel to individual parameters.
    IN_MODEL: Apply kernel within a single model scope.
    CROSS_MODEL: Apply kernel across multiple models.
    """

    PARAMETER = auto()
    IN_MODEL = auto()
    CROSS_MODEL = auto()


class KernelExecutionProtocol(Enum):
    """Defines the execution protocol for kernel processing.

    SEQUENTIAL: Execute kernel tasks sequentially.
    PARALLEL: Execute kernel tasks in parallel.
    """

    SEQUENTIAL = auto()
    PARALLEL = auto()


class KernelRestrictions(Flag):
    """Flag-based restrictions that can be applied to kernel arguments.

    BINARY: Restrict kernel to binary operation (exactly two arguments).
    """

    BINARY = auto()
