"""ONNX-specific parameter handlers for dense parameters.

This module implements handlers for extracting dense weights from ONNX models.
The extractor is responsible for selecting candidate weight arrays; this
handler validates and standardizes them.

Currently supports only DENSE weights by accepting 2D NumPy arrays.
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


if not import_utils.is_available("onnx"):
    logger.debug("ONNX not available, disabling corresponding handlers")

    class OnnxDenseHandler(ParameterHandler):
        """Stub handler when ONNX is not available."""

        def __new__(  # noqa: D102
            cls,
            *_args: Any,
            **_kwargs: Any,  # type: ignore[name-defined]
        ) -> Self:
            msg = "onnx package not available"
            raise ImportError(msg)

else:
    import numpy as np

    class OnnxDenseHandler(ParameterHandler[Any]):
        """Handler for ONNX dense weights represented as 2D NumPy arrays."""

        _EXPECTED_NDIM = 2

        @property
        def parameter_type(self) -> ParameterType:  # noqa: D102
            return ParameterType.DENSE

        def can_handle(self, parameter: Any, _param_name: str) -> bool:  # noqa: D102
            arr = np.asarray(parameter)
            return arr.ndim == self._EXPECTED_NDIM and arr.size > 1

        def process(  # noqa: D102
            self, parameter: Any, _param_name: str
        ) -> NDArray[np.floating[Any]]:
            arr = np.asarray(parameter)
            if arr.ndim != self._EXPECTED_NDIM:
                msg = (
                    f"Expected {self._EXPECTED_NDIM}D ONNX weight array, "
                    f"got shape {arr.shape}"
                )
                raise ValueError(msg)
            return cast("NDArray[np.floating[Any]]", arr)
