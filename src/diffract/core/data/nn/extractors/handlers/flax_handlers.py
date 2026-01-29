"""Flax/JAX-specific parameter handlers for dense parameters.

This module implements handlers for extracting dense layer weights from Flax
parameter trees (typically `variables["params"]`).

Currently supports only DENSE weights by selecting 2D arrays whose leaf name
is `kernel` (Flax Linen Dense convention).
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


if not (import_utils.is_available("flax") and import_utils.is_available("jax")):
    logger.debug("Flax/JAX not available, disabling corresponding handlers")

    class FlaxDenseHandler(ParameterHandler):
        """Stub handler when Flax/JAX is not available."""

        def __new__(  # noqa: D102
            cls,
            *_args: Any,
            **_kwargs: Any,  # type: ignore[name-defined]
        ) -> Self:
            msg = "Flax/JAX packages not available"
            raise ImportError(msg)

else:
    jax = import_utils.require("jax")
    import numpy as np

    class FlaxDenseHandler(ParameterHandler[Any]):
        """Handler for Flax Dense kernel weights in params trees."""

        _EXPECTED_NDIM = 2

        @property
        def parameter_type(self) -> ParameterType:  # noqa: D102
            return ParameterType.DENSE

        def can_handle(self, parameter: Any, param_name: str) -> bool:  # noqa: D102
            # `parameter` is expected to be a leaf array-like.
            if not param_name.endswith(".kernel"):
                return False
            arr = np.asarray(jax.device_get(parameter))
            return arr.ndim == self._EXPECTED_NDIM and arr.size > 1

        def process(  # noqa: D102
            self, parameter: Any, _param_name: str
        ) -> NDArray[np.floating[Any]]:
            arr = np.asarray(jax.device_get(parameter))
            if arr.ndim != self._EXPECTED_NDIM:
                msg = (
                    f"Expected {self._EXPECTED_NDIM}D kernel array, "
                    f"got shape {arr.shape}"
                )
                raise ValueError(msg)
            return cast("NDArray[np.floating[Any]]", arr)
