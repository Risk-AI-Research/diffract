"""Parameter handler interface and lightweight registry.

This module defines the abstract ParameterHandler interface and a minimal
registry for selecting and executing handlers. The design is intentionally
simple: handlers are tried in registration order without priorities, and
the registry returns both the processed weights and the handler instance
that performed the processing.

Key points:
    - Single-pass selection in registration order
    - Handlers declare produced ParameterType via parameter_type property
    - Registry returns (ptype, weights, handler) to avoid double lookup
    - Supported types are derived from registered handlers

Example:
    >>> registry = ParameterHandlerRegistry()
    >>> registry.register_handler_class(MyDenseHandler)
    >>> out = registry.process_parameter(module, "layer.weight")
    >>> if out:
    ...     ptype, weights, handler = out
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from diffract.core.data.nn.params.schema import ParameterType


class ParameterHandler[T](ABC):
    """Base class for a concrete parameter-type handler (no priority)."""

    @property
    @abstractmethod
    def parameter_type(self) -> ParameterType:
        """ParameterType produced by this handler."""
        raise NotImplementedError

    @abstractmethod
    def can_handle(self, parameter: T, param_name: str) -> bool:
        """Return True if handler can process the given parameter."""
        raise NotImplementedError

    @abstractmethod
    def process(self, parameter: T, param_name: str) -> NDArray[np.floating[Any]]:
        """Return processed weights as a NumPy array."""
        raise NotImplementedError

    def get_additional_metadata(
        self, _parameter: T, _param_name: str
    ) -> dict[str, Any]:
        """Return optional metadata; never raises and returns empty on failure."""
        return {}


class ParameterHandlerRegistry[T]:
    """Registry of handlers with registration-order selection."""

    def __init__(self) -> None:
        self._handlers: list[ParameterHandler[T]] = []

    def register_handler(self, handler: ParameterHandler[T]) -> None:
        """Register a handler instance for later selection."""
        self._handlers.append(handler)

    def register_handler_class(self, handler_cls: type[ParameterHandler[T]]) -> None:
        """Instantiate and register a handler class."""
        self.register_handler(handler_cls())

    def list_handlers(self) -> list[ParameterHandler[T]]:
        """Return a shallow copy of registered handlers."""
        return list(self._handlers)

    def get_supported_types(self) -> list[ParameterType]:
        """Return unique ParameterType values supported by registered handlers."""
        seen: set[ParameterType] = set()
        out: list[ParameterType] = []
        for h in self._handlers:
            if h.parameter_type not in seen:
                seen.add(h.parameter_type)
                out.append(h.parameter_type)
        return out

    def process_parameter(
        self,
        parameter: T,
        param_name: str,
    ) -> tuple[ParameterType, NDArray[np.floating[Any]], ParameterHandler[T]] | None:
        """Select a matching handler and process in one pass.

        Tries handlers in registration order, returning the first match's
        produced type, processed weights, and the handler instance.

        Args:
            parameter: Framework-specific parameter object.
            param_name: Human-readable parameter name.

        Returns:
            Tuple of (parameter type, weights array, handler) or None if no match.
        """
        for handler in self._handlers:
            if handler.can_handle(parameter, param_name):
                weights = handler.process(parameter, param_name)
                return handler.parameter_type, weights, handler
        return None
