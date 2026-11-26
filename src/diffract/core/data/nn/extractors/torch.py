"""PyTorch-specific parameter extractors.

This module provides parameter extractors specifically designed for PyTorch
models and state dictionaries. It implements the BaseParameterExtractor
interface with PyTorch-specific parameter iteration and handler registration.

Supported PyTorch Objects:
    - torch.nn.Module: Complete PyTorch models with named modules
    - dict[str, torch.Tensor]: PyTorch state dictionaries

The extractors use handler-based processing to support different parameter
types while maintaining extensibility for custom parameter handling. Currently
focused on dense (fully-connected) layer parameters with room for expansion.

Example:
    >>> import torch
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    ... )
    >>> extractor = TorchModuleExtractor(model)
    >>> report = extractor.extract_parameters(parameter_repository)
    >>> # For state dict:
    >>> state_dict = model.state_dict()
    >>> extractor = TorchStateDictExtractor(state_dict)
    >>> report = extractor.extract_parameters(parameter_repository)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import diffract.core.utils.imports as import_utils

from .base import BaseParameterExtractor
from .report import ExtractionItem

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch
    from torch import nn

    from .base import ExtractorOverrides
    from .handlers.base import ParameterHandler

logger = logging.getLogger(__name__)


def _is_bias(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf == "bias" or leaf.startswith("bias_") or leaf.endswith("_bias")


if not import_utils.is_available("torch"):
    logger.debug("PyTorch not available, disabling PyTorch extractors")

    class TorchModuleExtractor(BaseParameterExtractor):
        """Stub implementation when PyTorch is not available.

        Raises ImportError when instantiated to indicate PyTorch dependency
        is missing.
        """

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError since PyTorch is unavailable."""
            msg = "PyTorch package not available"
            raise ImportError(msg)

    class TorchStateDictExtractor(BaseParameterExtractor):
        """Stub implementation when PyTorch is not available.

        Raises ImportError when instantiated to indicate PyTorch dependency
        is missing.
        """

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError since PyTorch is unavailable."""
            msg = "PyTorch package not available"
            raise ImportError(msg)

else:
    from .handlers import TorchDenseHandler, TorchStateDictDenseHandler

    class TorchModuleExtractor(BaseParameterExtractor):
        """PyTorch nn.Module parameter extractor with handler-based processing.

        Extracts parameters from PyTorch nn.Module objects by iterating through
        named modules and processing them with registered handlers. Currently
        supports dense (fully-connected) layer parameters with extensible
        architecture for additional parameter types.

        Example:
            >>> import torch.nn as nn
            >>> model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 10))
            >>> extractor = TorchModuleExtractor(model)
            >>> report = extractor.extract_parameters(parameter_repository)

        Attributes:
            model: PyTorch nn.Module to extract parameters from.
        """

        def __init__(
            self,
            model: nn.Module,
            overrides: ExtractorOverrides | None = None,
            skip_not_implemented_types: bool = True,
            custom_handlers: list[ParameterHandler] | None = None,
        ) -> None:
            """Initialize PyTorch module extractor.

            Args:
                model: PyTorch nn.Module to extract parameters from.
                overrides: Optional extraction behavior overrides.
                skip_not_implemented_types: Whether to skip unsupported modules.
                custom_handlers: Optional list of custom parameter handlers.
            """
            super().__init__(
                model, overrides, skip_not_implemented_types, custom_handlers
            )

        def _register_default_handlers(self) -> None:
            """Register default handlers for PyTorch modules.

            Registers the TorchDenseHandler for processing dense/linear layers.
            Additional handlers can be registered for other layer types.
            """
            # Register only the DENSE handler that was originally supported
            self.handler_registry.register_handler_class(TorchDenseHandler)
            logger.debug(
                "Registered %d default handlers for PyTorch modules (DENSE only)",
                len(self.handler_registry.list_handlers()),
            )

        def _iter_parameters(self) -> Generator[tuple[str, nn.Module], None, None]:
            """Iterate over named modules, naming each by its weight tensor path.

            Yields all named modules in the model, allowing handlers to process
            each module type appropriately (including leaf modules and
            intermediate containers). The yielded name is the ``state_dict``
            path of the module's ``weight`` tensor — ``{module_path}.weight``,
            or ``weight`` for a bare root module — so the module view and the
            ``state_dict`` view agree on every name they share, and
            ``parameter_overrides`` keys match across views.

            Yields:
                Tuples of (weight_tensor_path, module) for each module.
            """
            for module_path, module in self.model.named_modules():
                name = f"{module_path}.weight" if module_path else "weight"
                yield name, module

        @staticmethod
        def _source_dtype(parameter: Any) -> str | None:
            return BaseParameterExtractor._source_dtype(
                getattr(parameter, "weight", None)
            )

        def _unhandled_item(
            self, name: str, parameter: nn.Module
        ) -> ExtractionItem | None:
            if getattr(parameter, "weight", None) is None:
                return None
            return super()._unhandled_item(name, parameter)

        def _additional_items(self) -> Generator[ExtractionItem, None, None]:
            for path, module in self.model.named_modules():
                prefix = f"{path}." if path else ""
                for name, tensor in module.named_parameters(
                    recurse=False, remove_duplicate=False
                ):
                    if name == "weight":
                        continue
                    dtype = BaseParameterExtractor._source_dtype(tensor)
                    if _is_bias(name):
                        yield ExtractionItem(prefix + name, "excluded", "bias", dtype)
                    else:
                        yield ExtractionItem(
                            prefix + name,
                            "unsupported",
                            f"No handler for this parameter of {type(module).__name__}",
                            dtype,
                        )
                for name, tensor in module.named_buffers(
                    recurse=False, remove_duplicate=False
                ):
                    if name != "weight":
                        yield ExtractionItem(
                            prefix + name,
                            "excluded",
                            "module buffer",
                            BaseParameterExtractor._source_dtype(tensor),
                        )

    class TorchStateDictExtractor(BaseParameterExtractor):
        """PyTorch state_dict parameter extractor with handler-based processing.

        Extracts parameters from PyTorch state dictionaries by iterating through
        tensor entries and processing them with registered handlers. State dicts
        provide direct access to parameter tensors without module structure.

        This extractor is useful for processing saved model weights, checkpoint
        files, or when working with parameter tensors directly without the
        full model architecture.

        Example:
            >>> import torch
            >>> state_dict = torch.load("model.pt")
            >>> extractor = TorchStateDictExtractor(state_dict)
            >>> report = extractor.extract_parameters(parameter_repository)

        Attributes:
            model: Dictionary mapping parameter names to tensors.
        """

        def __init__(
            self,
            model: dict[str, torch.Tensor],
            overrides: ExtractorOverrides | None = None,
            skip_not_implemented_types: bool = True,
            custom_handlers: list[ParameterHandler] | None = None,
        ) -> None:
            """Initialize PyTorch state dict extractor.

            Args:
                model: Dictionary mapping parameter names to torch.Tensor.
                overrides: Optional extraction behavior overrides.
                skip_not_implemented_types: Whether to skip unsupported tensors.
                custom_handlers: Optional list of custom parameter handlers.
            """
            super().__init__(
                model, overrides, skip_not_implemented_types, custom_handlers
            )

        def _register_default_handlers(self) -> None:
            """Register default handlers for PyTorch state dict tensors.

            Registers the TorchStateDictDenseHandler for processing dense layer
            parameter tensors from state dictionaries.
            """
            # Register only the DENSE handler that was originally supported
            self.handler_registry.register_handler_class(TorchStateDictDenseHandler)
            logger.debug(
                "Registered %d default handlers for PyTorch state_dict (DENSE only)",
                len(self.handler_registry.list_handlers()),
            )

        def _iter_parameters(self) -> Generator[tuple[str, torch.Tensor], None, None]:
            """Iterate over parameter tensors in the state dictionary.

            Yields all parameter name-tensor pairs from the state dictionary,
            allowing handlers to process each tensor appropriately.

            Yields:
                Tuples of (parameter_name, tensor) for each entry in state_dict.
            """
            state_dict: dict[str, torch.Tensor] = self.model
            yield from state_dict.items()

        def _unhandled_item(self, name: str, parameter: torch.Tensor) -> ExtractionItem:
            dtype = self._source_dtype(parameter)
            if not name.endswith("weight"):
                reason = "bias" if _is_bias(name) else "not a weight name"
                return ExtractionItem(name, "excluded", reason, dtype)
            return ExtractionItem(
                name,
                "unsupported",
                f"Expected a 2D weight with multiple elements, "
                f"got {tuple(parameter.shape)}",
                dtype,
            )
