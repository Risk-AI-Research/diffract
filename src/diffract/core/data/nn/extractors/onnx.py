"""ONNX parameter extractors.

Provides a parameter extractor for ONNX `ModelProto` objects.
Currently supports only dense (fully-connected) weights, extracted from `Gemm`
nodes when their weight input is present in graph initializers.
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


if not import_utils.is_available("onnx"):
    logger.debug("ONNX not available, disabling ONNX extractors")

    class OnnxModelExtractor(BaseParameterExtractor):
        """Stub implementation when ONNX is not available."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:  # noqa: D102
            msg = "onnx package not available"
            raise ImportError(msg)

else:
    import numpy as np

    onnx = import_utils.require("onnx")
    numpy_helper = import_utils.require("onnx.numpy_helper")

    from .handlers import OnnxDenseHandler

    def _onnx_node_display_name(node: Any) -> str:
        if getattr(node, "name", ""):
            return node.name
        outputs = list(getattr(node, "output", []) or [])
        if outputs:
            return outputs[0]
        return "gemm"

    class OnnxModelExtractor(BaseParameterExtractor):
        """ONNX ModelProto parameter extractor (DENSE only)."""

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
            self.handler_registry.register_handler_class(OnnxDenseHandler)
            logger.debug(
                "Registered %d default handlers for ONNX models (DENSE only)",
                len(self.handler_registry.list_handlers()),
            )

        def _iter_parameters(self) -> Generator[tuple[str, Any], None, None]:
            """Yield dense weight arrays from Gemm nodes.

            For `Gemm` node: inputs are typically [A, B, C]. We treat B as the
            weight matrix, and ignore bias (C).
            """
            model = self.model
            if not isinstance(model, onnx.ModelProto):
                msg = f"OnnxModelExtractor expects onnx.ModelProto, got {type(model)}"
                raise TypeError(msg)

            graph = model.graph
            initializer_by_name: dict[str, Any] = {
                init.name: init for init in graph.initializer
            }

            min_gemm_inputs = 2
            for node in graph.node:
                if getattr(node, "op_type", None) != "Gemm":
                    continue

                inputs = list(getattr(node, "input", []) or [])
                if len(inputs) < min_gemm_inputs:
                    continue

                weight_name = inputs[1]
                init = initializer_by_name.get(weight_name)
                if init is None:
                    continue

                weight_arr = numpy_helper.to_array(init)
                # Ensure NumPy array for downstream handlers.
                weight_arr = np.asarray(weight_arr)

                node_name = _onnx_node_display_name(node)
                yield f"{node_name}.weight", weight_arr
