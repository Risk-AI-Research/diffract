"""NumPy-specific parameter extractors.

This module provides a parameter extractor for plain dictionaries mapping
parameter names to NumPy arrays. It implements the BaseParameterExtractor
interface without requiring any deep learning framework, since NumPy arrays
already match the library's internal weight format.

Supported Objects:
    - dict[str, numpy.ndarray]: Mappings of parameter names to arrays

Example:
    >>> import numpy as np
    >>> weights = {"encoder.weight": np.random.rand(10, 5)}
    >>> extractor = NumpyDictExtractor(weights)
    >>> parameters = extractor.extract_parameters(storage, cache)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseParameterExtractor
from .handlers import NumpyDenseHandler

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from .base import ExtractorOverrides
    from .handlers.base import ParameterHandler

logger = logging.getLogger(__name__)


class NumpyDictExtractor(BaseParameterExtractor):
    """NumPy dictionary parameter extractor with handler-based processing.

    Extracts parameters from dictionaries mapping names to NumPy arrays by
    iterating through entries and processing them with registered handlers.
    This is the framework-free entry point: weight matrices loaded from
    ``.npy``/``.npz`` files or via ``safetensors.numpy`` can be analyzed
    without installing any deep learning framework.

    Example:
        >>> import numpy as np
        >>> weights = {"encoder.weight": np.random.rand(10, 5)}
        >>> extractor = NumpyDictExtractor(weights)
        >>> params = extractor.extract_parameters(storage, cache)

    Attributes:
        model: Dictionary mapping parameter names to arrays.
    """

    def __init__(
        self,
        model: dict[str, NDArray[Any]],
        overrides: ExtractorOverrides | None = None,
        skip_not_implemented_types: bool = True,
        custom_handlers: list[ParameterHandler] | None = None,
    ) -> None:
        """Initialize NumPy dictionary extractor.

        Args:
            model: Dictionary mapping parameter names to numpy.ndarray.
            overrides: Optional extraction behavior overrides.
            skip_not_implemented_types: Whether to skip unsupported arrays.
            custom_handlers: Optional list of custom parameter handlers.
        """
        super().__init__(model, overrides, skip_not_implemented_types, custom_handlers)

    def _register_default_handlers(self) -> None:
        """Register default handlers for NumPy array dictionaries.

        Registers the NumpyDenseHandler for processing dense weight matrices.
        """
        self.handler_registry.register_handler_class(NumpyDenseHandler)
        logger.debug(
            "Registered %d default handlers for NumPy arrays (DENSE only)",
            len(self.handler_registry.list_handlers()),
        )

    def _iter_parameters(self) -> Generator[tuple[str, NDArray[Any]], None, None]:
        """Iterate over arrays in the dictionary.

        Yields all name-array pairs from the dictionary, allowing handlers
        to process each array appropriately.

        Yields:
            Tuples of (parameter_name, array) for each entry in the dict.
        """
        arrays: dict[str, NDArray[Any]] = self.model
        yield from arrays.items()
