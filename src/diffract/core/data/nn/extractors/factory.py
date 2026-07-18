"""Factory pattern for creating framework-specific parameter extractors.

This module implements the factory pattern for automatically detecting neural
network model types and creating appropriate parameter extractors. It provides
a unified interface that abstracts away framework-specific details while
enabling extensible handler-based parameter processing.

The factory automatically detects supported frameworks and model types,
creating the most appropriate extractor implementation. It supports custom
handlers for specialized parameter processing and provides comprehensive
error handling for unsupported model types.

Supported Frameworks:
    - NumPy: dict[str, numpy.ndarray] mappings (no framework required)
    - PyTorch: torch.nn.Module and state_dict objects
    - Extensible: Easy to add support for additional frameworks

Key Features:
    - Automatic model type detection
    - Framework-specific extractor creation
    - Custom handler support for specialized processing
    - Comprehensive error reporting with supported types
    - Runtime framework availability checking

Example:
    >>> import torch
    >>> model = torch.nn.Linear(10, 5)
    >>> extractor = create_extractor(model)
    >>> # With custom handlers:
    >>> extractor = create_extractor(model, custom_handlers=[MyHandler()])
    >>> parameters = extractor.extract_parameters(storage, cache)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import diffract.core.utils.imports as import_utils

from .interface import IParameterExtractor

torch = import_utils.get_module("torch")
onnx = import_utils.get_module("onnx")
tf = import_utils.get_module("tensorflow")
flax = import_utils.get_module("flax")

logger = logging.getLogger(__name__)


def create_extractor(model: Any, *args: Any, **kwargs: Any) -> IParameterExtractor:
    """Create appropriate parameter extractor based on model type.

    Automatically detects the model type and framework, then creates the
    most suitable parameter extractor. Supports custom handlers and
    extractor overrides for specialized parameter processing.

    Args:
        model: Neural network model to extract parameters from.
        *args: Additional positional arguments passed to extractor.
        **kwargs: Additional keyword arguments passed to extractor.

    Returns:
        Parameter extractor instance appropriate for the model type.

    Raises:
        ImportError: If the model is not a NumPy dict and no supported
            frameworks are available.
        TypeError: If a dict of arrays carries non-string keys, or the
            model type is not supported by any available framework.

    Example:
        >>> extractor = create_extractor(torch_model)
        >>> extractor = create_extractor({"encoder.weight": np.random.rand(10, 5)})
        >>> extractor = create_extractor(state_dict, custom_handlers=[handler])
    """
    if isinstance(model, dict) and all(
        isinstance(name, str) and isinstance(value, np.ndarray)
        for name, value in model.items()
    ):
        from .numpy import NumpyDictExtractor

        logger.debug(
            "Creating NumpyDictExtractor for array dict with %d parameters",
            len(model),
        )

        return NumpyDictExtractor(*args, model=model, **kwargs)

    # An array dict with non-string keys is an invalid numpy-dict input, not
    # a framework model; rejecting it here keeps the outcome identical
    # whether or not any framework is installed.
    if (
        isinstance(model, dict)
        and model
        and all(isinstance(value, np.ndarray) for value in model.values())
    ):
        key_types = sorted(
            {type(name).__name__ for name in model if not isinstance(name, str)}
        )
        msg = (
            "Array dict keys must be parameter-name strings; got key "
            f"type(s): {', '.join(key_types)}. "
            "Pass a dict[str, numpy.ndarray] of weight matrices."
        )
        raise TypeError(msg)

    if not get_supported_frameworks():
        msg = (
            "No supported deep learning frameworks available. "
            "Please install PyTorch (torch) or other supported frameworks, "
            "or pass a dict[str, numpy.ndarray] of weight matrices."
        )
        raise ImportError(msg)

    if torch:
        # PyTorch state_dict (dictionary of tensors)
        if (
            isinstance(model, dict)
            and model
            and all(isinstance(v, torch.Tensor) for v in model.values())
        ):
            from .torch import TorchStateDictExtractor

            logger.debug(
                "Creating TorchStateDictExtractor for state dict with %d parameters",
                len(model),
            )

            return TorchStateDictExtractor(*args, model=model, **kwargs)

        # PyTorch nn.Module (standard PyTorch models)
        if isinstance(model, torch.nn.Module):
            from .torch import TorchModuleExtractor

            logger.debug(
                "Creating TorchModuleExtractor for %s model", type(model).__name__
            )

            return TorchModuleExtractor(*args, model=model, **kwargs)

    if onnx and isinstance(model, onnx.ModelProto):
        # ONNX model
        from .onnx import OnnxModelExtractor

        logger.debug("Creating OnnxModelExtractor for ONNX ModelProto")
        return OnnxModelExtractor(*args, model=model, **kwargs)

    if tf and isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)):
        # TensorFlow / Keras model or layer
        from .tensorflow import TensorFlowModelExtractor

        logger.debug("Creating TensorFlowModelExtractor for %s", type(model).__name__)
        return TensorFlowModelExtractor(*args, model=model, **kwargs)

    if flax:
        # Flax variables/params: mapping with "params" key or params mapping itself.
        # We avoid importing JAX types here; rely on duck-typing for mappings.
        try:
            from collections.abc import Mapping
        except ImportError:
            mapping_type = None  # type: ignore[assignment]
        else:
            mapping_type = Mapping

        if mapping_type is not None and isinstance(model, mapping_type):
            from .flax import FlaxParamsExtractor

            logger.debug("Creating FlaxParamsExtractor for params/variables mapping")
            return FlaxParamsExtractor(*args, model=model, **kwargs)

        if hasattr(model, "variables"):
            from .flax import FlaxParamsExtractor

            logger.debug("Creating FlaxParamsExtractor for bound Flax module")
            return FlaxParamsExtractor(*args, model=model, **kwargs)

    error_msg = f"Unsupported model type: {type(model).__name__}. "

    if supported_types := get_supported_types():
        error_msg += f"Supported types: {', '.join(supported_types)}"
    else:
        error_msg += "No frameworks available."

    raise TypeError(error_msg)


def get_supported_frameworks() -> list[str]:
    """Get list of available deep learning frameworks.

    Checks for the availability of supported deep learning frameworks
    and returns their names. This is useful for diagnostics and
    conditional feature availability.

    Returns:
        List of names of available frameworks.

    Example:
        >>> frameworks = get_supported_frameworks()
        >>> print(f"Available frameworks: {', '.join(frameworks)}")
    """
    frameworks: list[str] = []
    if torch:
        frameworks.append("PyTorch")
    if tf:
        frameworks.append("TensorFlow")
    if flax:
        frameworks.append("Flax")
    if onnx:
        frameworks.append("ONNX")
    return frameworks


def get_supported_types() -> list[str]:
    """Get list of supported model types across all frameworks.

    Returns a list of human-readable descriptions of supported model
    types. This is useful for error messages and documentation.

    Returns:
        List of supported model type descriptions.

    Example:
        >>> types = get_supported_types()
        >>> print(f"Supported types: {', '.join(types)}")
    """
    types: list[str] = ["dict[str, numpy.ndarray] (NumPy weight matrices)"]
    if torch:
        types.extend(
            [
                "torch.nn.Module (PyTorch models)",
                "dict[str, torch.Tensor] (PyTorch state_dict)",
            ]
        )
    if tf:
        types.extend(
            [
                "tf.keras.Model (TensorFlow/Keras models)",
                "tf.keras.layers.Layer (TensorFlow/Keras layers)",
            ]
        )
    if flax:
        types.extend(
            [
                'Mapping[str, Any] (Flax params tree / variables with key "params")',
                "Bound module with `.variables['params']` (Flax Linen bound module)",
            ]
        )
    if onnx:
        types.append("onnx.ModelProto (ONNX model)")
    return types
