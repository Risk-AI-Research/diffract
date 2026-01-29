"""Kernel metadata and information structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import KernelConfig
from .execution import KernelApplyLevel, KernelExecutionProtocol, KernelRestrictions

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class KernelInfo:
    """Optional human-readable kernel documentation fields.

    Contains optional metadata for kernel documentation and introspection.
    Used to provide additional context about kernel purpose and usage.

    Attributes:
        summary: Brief description of kernel functionality.
        notes: Additional implementation notes or usage guidelines.
    """

    summary: str | None = None
    notes: str | None = None


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

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        """Return string representation for debugging and logging."""
        conf = ", ".join(f"{k}={v}" for k, v in self.config.as_dict().items())
        restrictions = f",{self.restrictions}" if self.restrictions else ""
        return (
            f"{self.name} ({', '.join(self.require_fields)}; {conf}): "
            f"{self.apply_level}{restrictions}"
        )
