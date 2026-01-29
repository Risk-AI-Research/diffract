"""Parameter extraction framework for neural network models.

This module provides a comprehensive framework for extracting parameters from
neural network models across different deep learning frameworks. It uses a
handler-based architecture for extensible and framework-specific parameter
processing.

Key Features:
    - Framework-agnostic parameter extraction interface
    - Handler-based processing for different parameter types
    - Factory pattern for automatic extractor creation
    - Override system for custom parameter metadata
    - Extensible architecture for new frameworks and parameter types

Supported Frameworks:
    - PyTorch: nn.Module and state_dict extraction
    - Extensible design for additional frameworks

Example:
    >>> from diffract.core.data.nn.extractors import create_extractor
    >>> extractor = create_extractor(pytorch_model)
    >>> parameters = extractor.extract_parameters(storage, cache)
"""

from .base import BaseParameterExtractor, ExtractorOverrides, ParameterOverrides
from .factory import create_extractor, get_supported_frameworks, get_supported_types
from .flax import FlaxParamsExtractor
from .interface import IParameterExtractor
from .onnx import OnnxModelExtractor
from .tensorflow import TensorFlowModelExtractor
from .torch import TorchModuleExtractor, TorchStateDictExtractor

__all__ = [
    "BaseParameterExtractor",
    "ExtractorOverrides",
    "FlaxParamsExtractor",
    "IParameterExtractor",
    "OnnxModelExtractor",
    "ParameterOverrides",
    "TensorFlowModelExtractor",
    "TorchModuleExtractor",
    "TorchStateDictExtractor",
    "create_extractor",
    "get_supported_frameworks",
    "get_supported_types",
]
