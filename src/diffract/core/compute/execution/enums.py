"""Kernel enumeration types and execution constants."""

from __future__ import annotations

from enum import Enum, Flag, auto


class KernelApplyLevel(Enum):
    """Defines the level at which kernel computation is applied.

    PARAMETER: Apply kernel to individual parameters.
    IN_MODEL: Apply kernel within a single model/checkpoint coordinate.
    CROSS_MODEL: Apply kernel for a tensor name across selected coordinates.
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
    MATCH_ROWS: Require equal weight matrix row counts for a checkpoint pair.
    MATCH_COLUMNS: Require equal weight matrix column counts for a checkpoint pair.
    MATCH_LOWER_DIM: Require equal lengths of the two singular-value spectra.
    """

    BINARY = auto()
    MATCH_ROWS = auto()
    MATCH_COLUMNS = auto()
    MATCH_LOWER_DIM = auto()
