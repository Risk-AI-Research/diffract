"""TensorFlow/Keras parameter extractors.

Provides a parameter extractor for TensorFlow Keras models.
Currently supports only dense (fully-connected) layer kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import diffract.core.utils.imports as import_utils

from .base import BaseParameterExtractor

if TYPE_CHECKING:
    from collections.abc import Generator

    from .base import ExtractorOverrides
    from .handlers.base import ParameterHandler

logger = logging.getLogger(__name__)


if not import_utils.is_available("tensorflow"):
    logger.debug("TensorFlow not available, disabling TensorFlow extractors")

    class TensorFlowModelExtractor(BaseParameterExtractor):
        """Stub implementation when TensorFlow is not available."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:  # noqa: D102
            msg = "TensorFlow package not available"
            raise ImportError(msg)

else:
    tf = import_utils.require("tensorflow")

    from .handlers import TensorFlowDenseHandler

    def _walk_layers(obj: Any) -> Generator[Any, None, None]:
        """Yield layers contained in a Keras model/layer.

        Uses public attributes (`layers`) when available and falls back to yielding
        the object itself. This avoids relying on private APIs that may change
        across TensorFlow/Keras versions.
        """
        if isinstance(obj, tf.keras.Model):
            for layer in obj.layers:
                yield from _walk_layers(layer)
            return

        if isinstance(obj, tf.keras.layers.Layer):
            yield obj
            # Some composite layers expose nested layers via `.layers`.
            nested = getattr(obj, "layers", None)
            if isinstance(nested, list):
                for layer in nested:
                    yield from _walk_layers(layer)
            return

    class TensorFlowModelExtractor(BaseParameterExtractor):
        """TensorFlow Keras model parameter extractor (DENSE only)."""

        def __init__(
            self,
            model: Any,
            overrides: ExtractorOverrides | None = None,
            skip_not_implemented_types: bool = True,
            custom_handlers: list[ParameterHandler] | None = None,
        ) -> None:
            super().__init__(
                model, overrides, skip_not_implemented_types, custom_handlers
            )

        def _register_default_handlers(self) -> None:
            # Register only the DENSE handler.
            self.handler_registry.register_handler_class(TensorFlowDenseHandler)
            logger.debug(
                "Registered %d default handlers for TensorFlow models (DENSE only)",
                len(self.handler_registry.list_handlers()),
            )

        def _iter_parameters(self) -> Generator[tuple[str, Any], None, None]:
            """Yield Dense layers as parameters.

            We intentionally yield only Dense layers and name parameters as
            `<layer.name>.kernel` to match weight-level naming convention.
            """
            model = self.model
            if not isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)):
                msg = (
                    "TensorFlowModelExtractor expects a tf.keras.Model or "
                    f"tf.keras.layers.Layer, got {type(model)}"
                )
                raise TypeError(msg)

            for layer in _walk_layers(model):
                if isinstance(layer, tf.keras.layers.Dense):
                    yield f"{layer.name}.kernel", layer
