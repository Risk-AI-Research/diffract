"""Base classes and utilities for parameter extraction.

This module provides the foundational infrastructure for parameter extraction
from neural network models. It implements a handler-based architecture that
allows for extensible and framework-specific parameter processing.

Key Components:
    - ExtractorOverrides: Configuration for customizing parameter metadata
    - ParameterOverrides: Per-parameter override specifications
    - BaseParameterExtractor: Abstract base class for all extractors

The handler-based approach allows different parameter types to be processed
by specialized handlers while maintaining a consistent extraction interface.
This design supports extensibility for new parameter types and frameworks.

Example:
    >>> overrides = ExtractorOverrides(
    ...     model_id="custom_model",
    ...     parameter_overrides={"layer1.weight": ParameterOverrides(ptype="CUSTOM")},
    ... )
    >>> extractor = ConcreteExtractor(model, overrides=overrides)
    >>> parameters = extractor.extract_parameters(storage, cache)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from diffract.core.constants import WEIGHTS_FIELD
from diffract.core.data.identity import validate_identifier
from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils.hashing import get_unique_id

from .interface import IParameterExtractor

if TYPE_CHECKING:
    from collections.abc import Generator

    from .handlers.base import ParameterHandler, ParameterHandlerRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExtractorOverrides:
    """Configuration for customizing parameter extraction behavior.

    Provides a mechanism to override default parameter metadata and
    model identification during the extraction process. This allows
    for custom naming, typing, and categorization of parameters.

    Attributes:
        model_id: Custom model identifier to override default.
        parameter_overrides: Per-parameter override specifications.
    """

    @dataclass
    class ParameterOverrides:
        """Override specifications for individual parameters.

        Allows customization of parameter metadata on a per-parameter
        basis, including name changes, type reclassification, and extra metadata.

        Attributes:
            name: Custom parameter name to override default.
            ptype: Custom parameter type to override default.
            other_meta: Additional metadata to merge into ParameterMetadata.other_meta.
        """

        name: str | None = field(default=None)
        ptype: str | ParameterType | None = field(default=None)
        other_meta: dict[str, Any] | None = field(default=None)

    model_id: str | None = field(default=None)
    parameter_overrides: dict[str, ParameterOverrides] = field(default_factory=dict)


ParameterOverrides = ExtractorOverrides.ParameterOverrides


class BaseParameterExtractor(IParameterExtractor, ABC):
    """Abstract base class for parameter extractors with handler-based processing.

    Provides a common framework for extracting parameters from neural network
    models using a pluggable handler system. Handles parameter iteration,
    metadata construction, override application, and error management.

    The extractor uses a registry of parameter handlers to process different
    types of parameters. Each handler specializes in extracting weights and
    metadata from specific parameter types (e.g., dense layers, convolutions).

    Attributes:
        model: The neural network model to extract parameters from.
        overrides: Configuration for customizing extraction behavior.
        skip_not_implemented_types: Whether to skip unsupported parameter types.
        handler_registry: Registry of parameter handlers for processing.
    """

    def __init__(
        self,
        model: Any,
        overrides: ExtractorOverrides | None = None,
        skip_not_implemented_types: bool = True,
        custom_handlers: list[ParameterHandler] | None = None,
    ) -> None:
        """Initialize the base parameter extractor.

        Args:
            model: Neural network model to extract parameters from.
            overrides: Optional extraction behavior overrides.
            skip_not_implemented_types: Whether to skip unsupported parameters.
            custom_handlers: Optional list of custom parameter handlers.
        """
        self.model = model
        self.overrides = overrides or ExtractorOverrides()
        self.skip_not_implemented_types = skip_not_implemented_types

        # Import here to avoid circular imports
        from .handlers.base import ParameterHandlerRegistry

        self.handler_registry: ParameterHandlerRegistry[Any] = ParameterHandlerRegistry[
            Any
        ]()
        self._register_default_handlers()

        if custom_handlers:
            for handler in custom_handlers:
                self.handler_registry.register_handler(handler)

    @abstractmethod
    def _register_default_handlers(self) -> None:
        """Register default parameter handlers for this extractor.

        Subclasses must implement this method to register the appropriate
        handlers for their specific framework and parameter types.
        """

    def extract_parameters(
        self,
        parameter_repository: IParameterRepository,
    ) -> None:
        """Extract all parameters from the model using registered handlers.

        Iterates through model parameters, processes each with appropriate
        handlers, constructs metadata with overrides, and creates parameter
        proxies for storage and caching.

        Args:
            parameter_repository: Repository that owns storage/cache and receives
                extracted parameters.

        Raises:
            NotImplementedError: If parameter type unsupported and not skipping.
            Exception: If handler processing fails and not skipping.
        """
        with parameter_repository:
            model_id = get_unique_id()

            for param_idx, (param_name, parameter) in enumerate(
                self._iter_parameters()
            ):
                try:
                    result = self.handler_registry.process_parameter(
                        parameter, param_name
                    )
                except Exception:
                    # Preserve error context; decide policy: skip or raise
                    logger.exception('Handler failed for "%s"', param_name)
                    if self.skip_not_implemented_types:
                        continue
                    raise

                if result is None:
                    if self.skip_not_implemented_types:
                        logger.debug('Skipping unsupported parameter "%s"', param_name)
                        continue
                    msg = (
                        f'No handler found for parameter "{param_name}" of type '
                        f"{type(parameter)}"
                    )
                    raise NotImplementedError(msg)

                ptype, weights, handler = result

                param_meta = self._build_metadata(
                    parameter, model_id, param_name, param_idx, ptype, handler
                )
                param_proxy = ParameterDataProxy.create_and_store(
                    meta=param_meta,
                    repository=parameter_repository,
                )
                param_proxy.set_field(WEIGHTS_FIELD, weights)

                logger.debug(
                    'Extracted parameter "%s" with type %s (handler=%s)',
                    param_meta.name,
                    param_meta.ptype,
                    type(handler).__name__,
                )

                parameter_repository.append(param_proxy)

    def _build_metadata(
        self,
        parameter: Any,
        model_id: str,
        param_name: str,
        param_idx: int,
        ptype: ParameterType,
        handler: ParameterHandler,
    ) -> ParameterMetadata:
        """Build parameter metadata with handler input and overrides.

        Constructs base metadata, enriches it with handler-provided additional
        metadata, and applies any configured overrides.

        Args:
            parameter: The original parameter object.
            model_id: Identifier of the model the parameter belongs to.
            param_name: Name of the parameter.
            param_idx: Index of parameter in model iteration order.
            ptype: Parameter type determined by handler.
            handler: Handler that processed this parameter.

        Returns:
            Complete parameter metadata with overrides applied.
        """
        metadata: dict[str, Any] = {
            "name": param_name,
            "ptype": ptype,
            "model_id": model_id,
            "other_meta": {"in_model_idx": param_idx},
        }

        # Best-effort metadata from the selected handler (no exceptions)
        with suppress(Exception):
            extra = handler.get_additional_metadata(parameter, param_name) or {}
            metadata["other_meta"].update(extra)

        self._apply_overrides(metadata)
        validate_identifier(metadata["name"], kind="parameter name")
        return ParameterMetadata(**metadata)

    def _apply_overrides(self, metadata: dict[str, Any]) -> None:
        """Apply configured overrides to parameter metadata.

        Modifies the metadata dictionary in-place based on configured
        parameter-specific and global overrides. Preserves original
        values in other_meta for traceability.

        Args:
            metadata: Parameter metadata dictionary to modify.
        """
        param_name = metadata["name"]

        if param_name in self.overrides.parameter_overrides:
            param_overrides = self.overrides.parameter_overrides[param_name]

            if param_overrides.name:
                metadata["name"] = param_overrides.name
                metadata.setdefault("other_meta", {})["original_name"] = param_name

            if param_overrides.ptype is not None:
                metadata.setdefault("other_meta", {})["original_ptype"] = metadata[
                    "ptype"
                ].name
                metadata["ptype"] = (
                    ParameterType.from_string(param_overrides.ptype)
                    if isinstance(param_overrides.ptype, str)
                    else param_overrides.ptype
                )

            if param_overrides.other_meta:
                metadata.setdefault("other_meta", {}).update(param_overrides.other_meta)

        if self.overrides.model_id:
            metadata.setdefault("other_meta", {})["original_model_id"] = metadata[
                "model_id"
            ]
            metadata["model_id"] = self.overrides.model_id

    @abstractmethod
    def _iter_parameters(self) -> Generator[tuple[str, Any], None, None]:
        """Iterate over model parameters yielding (name, parameter) tuples.

        Subclasses must implement this method to provide framework-specific
        parameter iteration logic.

        Yields:
            Tuples of (parameter_name, parameter_object) for each parameter.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
