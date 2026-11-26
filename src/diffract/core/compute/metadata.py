"""Kernel metadata and information structures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import KernelConfig
from .execution import KernelApplyLevel, KernelExecutionProtocol, KernelRestrictions

if TYPE_CHECKING:
    from .config import OutputComparison


type ExecutionBinding = Callable[
    [dict[str, Any]], tuple[dict[str, str], Callable[..., Any]]
]


@dataclass(frozen=True)
class KernelInfo:
    """Optional human-readable kernel documentation fields.

    Contains optional metadata for kernel documentation and introspection.
    Used to provide additional context about kernel purpose and usage.

    Attributes:
        summary: Brief description of kernel functionality.
        notes: Additional implementation notes or usage guidelines.
        formula: Per-name display formula (LaTeX) for the metrics catalog,
            overriding the ``:math:`` role parsed from the kernel docstring.
            Set it on stacked registrations whose shared body needs a
            distinct formula per registered name.
    """

    summary: str | None = None
    notes: str | None = None
    formula: str | None = None


@dataclass
class KernelMetadata:
    """Complete metadata container for registered kernels.

    Stores all information required for kernel registration, execution,
    and dependency resolution within the kernel registry system.

    Attributes:
        name: Unique kernel identifier.
        require_fields: Tuple of field names required as input.
        produce_fields: Tuple of field names produced as output.
        implementation: Callable implementing the kernel logic.
        apply_level: Level at which kernel is applied (parameter/model/cross-model).
        execution_protocol: Execution strategy (sequential/parallel).
        restrictions: Optional execution restrictions.
        config: Configuration parameters with defaults.
        info: Optional documentation metadata.
        version: Explicit revision of the kernel behavior.
        output_comparison: Optional override of scalar output equivalence.
        stream_inputs: Load one parameter's arguments at a time within a batch.
    """

    name: str
    require_fields: tuple[str, ...]
    produce_fields: tuple[str, ...]
    implementation: Callable[..., Any]
    apply_level: KernelApplyLevel
    execution_protocol: KernelExecutionProtocol | None
    restrictions: KernelRestrictions | None
    config: KernelConfig
    info: KernelInfo
    symmetric: bool | None = None
    version: str = "1"
    output_comparison: OutputComparison | None = None
    stream_inputs: bool = False
    bind_execution: ExecutionBinding | None = None
    builtin: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        """Return string representation for debugging and logging."""
        conf = ", ".join(f"{k}={v}" for k, v in self.config.as_dict().items())
        restrictions = f",{self.restrictions}" if self.restrictions else ""
        return (
            f"{self.name} ({', '.join(self.require_fields)}; {conf}): "
            f"{self.apply_level}{restrictions}"
        )
