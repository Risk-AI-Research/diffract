"""TensorFlow-specific parameter handlers for dense parameters.

This module implements handlers for extracting dense (fully-connected) layer
weights from TensorFlow/Keras models. Currently supports only DENSE weights
from `tf.keras.layers.Dense`.
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


if not import_utils.is_available("tensorflow"):
    logger.debug("TensorFlow not available, disabling corresponding handlers")

    class TensorFlowDenseHandler(ParameterHandler):
        """Stub handler when TensorFlow is not available."""

        def __new__(  # noqa: D102
            cls,
            *_args: Any,
            **_kwargs: Any,  # type: ignore[name-defined]
        ) -> Self:
            msg = "TensorFlow package not available"
            raise ImportError(msg)

else:
    tf = import_utils.require("tensorflow")

    class TensorFlowDenseHandler(ParameterHandler[Any]):
        """Handler for TensorFlow/Keras Dense layers (tf.keras.layers.Dense)."""

        @property
        def parameter_type(self) -> ParameterType:  # noqa: D102
            return ParameterType.DENSE

        def can_handle(self, parameter: Any, _param_name: str) -> bool:  # noqa: D102
            return isinstance(parameter, tf.keras.layers.Dense)

        def process(  # noqa: D102
            self, parameter: Any, _param_name: str
        ) -> NDArray[np.floating[Any]]:
            if not isinstance(parameter, tf.keras.layers.Dense):
                msg = f"Expected tf.keras.layers.Dense, got {type(parameter)}"
                raise TypeError(msg)

            # Keras Dense exposes kernel as a Variable. Ensure it's created/built.
            if not hasattr(parameter, "kernel"):
                msg = "Dense layer has no 'kernel' attribute (model not built?)"
                raise ValueError(msg)

            kernel = parameter.kernel
            # In eager mode `.numpy()` is available; otherwise Keras models passed
            # for analysis are expected to be materialized.
            arr = kernel.numpy()
            return cast("NDArray[np.floating[Any]]", arr)

        def get_additional_metadata(  # noqa: D102
            self, parameter: Any, _param_name: str
        ) -> dict[str, Any]:
            if not isinstance(parameter, tf.keras.layers.Dense):
                return {}
            meta: dict[str, Any] = {
                "tf_dtype": getattr(parameter.kernel, "dtype", None),
            }
            # Useful for downstream analysis / sanity checks.
            with_size = getattr(parameter, "units", None)
            if with_size is not None:
                meta["units"] = with_size
            return meta
