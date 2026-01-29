"""PyTorch-specific parameter handlers for dense parameters.

This module implements handlers for extracting dense (fully-connected) layer
weights from PyTorch models and state dictionaries. Handlers declare their
produced ParameterType and provide optional metadata.

Behavior when PyTorch is unavailable:
    - Stub handler classes are defined that raise ImportError on instantiation
      to signal the missing dependency.

Example:
    >>> handler = TorchDenseHandler()
    >>> if handler.can_handle(module, name):
    ...     weights = handler.process(module, name)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self, cast

import diffract.core.utils.imports as import_utils
from diffract.core.data.nn.params.schema import ParameterType

from .base import ParameterHandler

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_DENSE_WEIGHT_NDIM = 2


if not import_utils.is_available("torch"):
    logger.debug("PyTorch not available, disabling corresponding handlers")

    class TorchDenseHandler(ParameterHandler):
        """Stub handler when PyTorch is not available."""

        def __new__(
            cls,
            *_args: Any,
            **_kwargs: Any,  # type: ignore[name-defined]
        ) -> Self:
            """Raise ImportError since PyTorch is unavailable."""
            msg = "PyTorch package not available"
            raise ImportError(msg)

    class TorchStateDictDenseHandler(ParameterHandler):
        """Stub handler when PyTorch is not available."""

        def __new__(
            cls,
            *_args: Any,
            **_kwargs: Any,  # type: ignore[name-defined]
        ) -> Self:
            """Raise ImportError since PyTorch is unavailable."""
            msg = "PyTorch package not available"
            raise ImportError(msg)

else:
    torch = import_utils.require("torch")
    nn = torch.nn

    class TorchDenseHandler(ParameterHandler[nn.Module]):
        """Handler for PyTorch Linear/Dense layers (nn.Linear)."""

        @property
        def parameter_type(self) -> ParameterType:
            """Return produced type for this handler (DENSE)."""
            return ParameterType.DENSE

        def can_handle(self, parameter: nn.Module, _param_name: str) -> bool:
            """Return True if parameter is a PyTorch nn.Linear module."""
            return isinstance(parameter, nn.Linear)

        def process(
            self, parameter: nn.Module, _param_name: str
        ) -> NDArray[np.floating[Any]]:
            """Convert nn.Linear weights to NumPy array using CPU tensors."""
            if not isinstance(parameter, nn.Linear):
                msg = f"Expected nn.Linear, got {type(parameter)}"
                raise TypeError(msg)
            return cast(
                "NDArray[np.floating[Any]]", parameter.weight.detach().cpu().numpy()
            )

        def get_additional_metadata(
            self, parameter: nn.Module, _param_name: str
        ) -> dict[str, Any]:
            """Return optional torch-specific metadata such as dtype."""
            if not isinstance(parameter, nn.Linear):
                return {}
            return {"torch_dtype": str(parameter.weight.dtype)}

    class TorchStateDictDenseHandler(ParameterHandler[torch.Tensor]):
        """Handler for 2D weight tensors in PyTorch state dicts."""

        @property
        def parameter_type(self) -> ParameterType:
            """Return produced type for this handler (DENSE)."""
            return ParameterType.DENSE

        def can_handle(self, parameter: torch.Tensor, param_name: str) -> bool:
            """Return True for 2D weight tensors with positive number of elements."""
            return (
                param_name.endswith("weight")
                and len(parameter.shape) == _DENSE_WEIGHT_NDIM
                and parameter.numel() > 1
            )

        def process(
            self, parameter: torch.Tensor, _param_name: str
        ) -> NDArray[np.floating[Any]]:
            """Convert torch.Tensor to NumPy array using CPU tensor."""
            return cast("NDArray[np.floating[Any]]", parameter.detach().cpu().numpy())

        def get_additional_metadata(
            self, parameter: torch.Tensor, _param_name: str
        ) -> dict[str, Any]:
            """Return optional torch-specific metadata such as dtype."""
            return {"torch_dtype": str(parameter.dtype)}
