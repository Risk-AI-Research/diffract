"""Flax/JAX parameter extractors.

Provides a parameter extractor for Flax models by reading a variables/params tree.
Currently supports only dense (fully-connected) kernel weights.

Supported inputs:
  - A mapping with key `"params"` (e.g. flax variables FrozenDict)
  - A bound Flax Linen module with `.variables` containing `"params"`
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

import diffract.core.utils.imports as import_utils

from .base import BaseParameterExtractor

if TYPE_CHECKING:
    from collections.abc import Generator

    from .base import ExtractorOverrides
    from .handlers.base import ParameterHandler

logger = logging.getLogger(__name__)


def _flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Generator[tuple[str, Any], None, None]:
    """Flatten nested mappings into dot-separated key paths."""
    for k, v in mapping.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            yield from _flatten_mapping(v, prefix=name)
        else:
            yield name, v


if not (import_utils.is_available("flax") and import_utils.is_available("jax")):
    logger.debug("Flax/JAX not available, disabling Flax extractors")

    class FlaxParamsExtractor(BaseParameterExtractor):
        """Stub implementation when Flax/JAX is not available."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:  # noqa: D102
            msg = "Flax/JAX packages not available"
            raise ImportError(msg)

else:
    from .handlers import FlaxDenseHandler

    class FlaxParamsExtractor(BaseParameterExtractor):
        """Flax params/variables extractor (DENSE only)."""

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
            self.handler_registry.register_handler_class(FlaxDenseHandler)
            logger.debug(
                "Registered %d default handlers for Flax params (DENSE only)",
                len(self.handler_registry.list_handlers()),
            )

        def _iter_parameters(self) -> Generator[tuple[str, Any], None, None]:
            """Iterate params leaves.

            We extract only params named `*.kernel` (Dense convention). Other leaves
            (bias, embeddings, etc.) are ignored by handlers.
            """
            obj = self.model

            # Case 1: bound module with variables dict.
            if hasattr(obj, "variables"):
                variables = obj.variables
                if isinstance(variables, Mapping) and "params" in variables:
                    params = variables["params"]
                else:
                    params = None
            else:
                params = None

            # Case 2: variables mapping or params mapping passed directly.
            if params is None:
                if isinstance(obj, Mapping) and "params" in obj:
                    params = obj["params"]
                elif isinstance(obj, Mapping):
                    # Treat passed object as the params tree.
                    params = obj

            if not isinstance(params, Mapping):
                msg = (
                    "FlaxParamsExtractor expects a mapping of params (or variables "
                    'mapping containing key "params"), got '
                    f"{type(self.model)}"
                )
                raise TypeError(msg)

            yield from _flatten_mapping(params)
