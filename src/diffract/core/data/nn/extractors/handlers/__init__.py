"""Parameter handlers for extensible parameter processing.

This subpackage implements a handler-based architecture for processing neural
network parameters across frameworks. Handlers detect whether they can process
an input parameter and, if so, convert it to standardized NumPy weights and
provide optional metadata.

Components:
    - base: Abstract handler interface and lightweight registry
    - torch_handlers: PyTorch handlers for dense parameters

Example:
    >>> from diffract.core.data.nn.extractors.handlers import (
    ...     ParameterHandlerRegistry,
    ...     TorchDenseHandler,
    ... )
    >>> registry = ParameterHandlerRegistry()
    >>> registry.register_handler_class(TorchDenseHandler)
    >>> # Later: handler = registry.process_parameter(module, name)
"""

from .base import ParameterHandler, ParameterHandlerRegistry
from .flax_handlers import FlaxDenseHandler
from .onnx_handlers import OnnxDenseHandler
from .tensorflow_handlers import TensorFlowDenseHandler
from .torch_handlers import TorchDenseHandler, TorchStateDictDenseHandler

__all__ = [
    "FlaxDenseHandler",
    "OnnxDenseHandler",
    "ParameterHandler",
    "ParameterHandlerRegistry",
    "TensorFlowDenseHandler",
    "TorchDenseHandler",
    "TorchStateDictDenseHandler",
]
