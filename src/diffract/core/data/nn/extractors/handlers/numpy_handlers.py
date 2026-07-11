"""NumPy-specific parameter handlers for dense parameters.

This module implements handlers for extracting dense (fully-connected) layer
weights from plain dictionaries of NumPy arrays. Handlers declare their
produced ParameterType and provide optional metadata.

NumPy is a required dependency, so no availability stubs are defined.

Example:
    >>> handler = NumpyDenseHandler()
    >>> if handler.can_handle(array, name):
    ...     weights = handler.process(array, name)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from diffract.core.data.nn.params.schema import ParameterType

from .base import ParameterHandler

logger = logging.getLogger(__name__)

_DENSE_WEIGHT_NDIM = 2


class NumpyDenseHandler(ParameterHandler[NDArray[Any]]):
    """Handler for 2D floating-point arrays in NumPy dictionaries."""

    @property
    def parameter_type(self) -> ParameterType:
        """Return produced type for this handler (DENSE)."""
        return ParameterType.DENSE

    def can_handle(self, parameter: NDArray[Any], _param_name: str) -> bool:
        """Return True for 2D floating-point arrays with more than one element.

        Masked arrays are rejected: their mask would be silently discarded,
        computing metrics over raw fill values.
        """
        return (
            parameter.ndim == _DENSE_WEIGHT_NDIM
            and parameter.size > 1
            and np.issubdtype(parameter.dtype, np.floating)
            and not isinstance(parameter, np.ma.MaskedArray)
        )

    def process(
        self, parameter: NDArray[Any], _param_name: str
    ) -> NDArray[np.floating[Any]]:
        """Return the array as NumPy weights."""
        return cast("NDArray[np.floating[Any]]", parameter)

    def get_additional_metadata(
        self, parameter: NDArray[Any], _param_name: str
    ) -> dict[str, Any]:
        """Return optional NumPy-specific metadata such as dtype."""
        return {"numpy_dtype": str(parameter.dtype)}
