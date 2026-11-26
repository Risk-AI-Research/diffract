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
    >>> report = extractor.extract_parameters(parameter_repository)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from diffract.core.constants import WEIGHTS_FIELD
from diffract.core.data.identity import (
    is_extra_state_key,
    normalize_parameter_name,
    validate_parameter_name,
)
from diffract.core.data.nn.params.interface import IParameterRepository
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.utils.hashing import get_unique_id

from .interface import IParameterExtractor
from .report import ExtractionItem, ExtractionReport

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Iterable

    import numpy as np

    from diffract.core.data.nn.instances.metadata import InstanceMetadata

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
        *,
        instance: InstanceMetadata | None = None,
        step: int = 0,
        required_names: Collection[str] = (),
    ) -> ExtractionReport:
        """Extract all parameters from the model using registered handlers.

        Iterates through model parameters, processes each with appropriate
        handlers, constructs metadata with overrides, and creates parameter
        proxies for storage and caching.

        Args:
            parameter_repository: Repository that owns storage/cache and receives
                extracted parameters.
            instance: Store identity reserved for this extraction.
            step: Numeric checkpoint for the extracted parameters.
            required_names: Exact source paths to validate before storing weights.

        Raises:
            NotImplementedError: If parameter type unsupported and not skipping.
            Exception: If handler processing fails and not skipping.
        """
        additional = tuple(self._additional_items())
        prepared = self._prepare_required(required_names, additional)
        items: list[ExtractionItem] = []
        with parameter_repository, ExitStack() as rollback:
            written: list[ParameterDataProxy] = []
            rollback.callback(
                self._discard_partial_extraction, parameter_repository, written
            )
            model_id = (
                instance.name
                if instance and not self.overrides.model_id
                else get_unique_id()
            )
            instance_uid = instance.instance_uid if instance else None

            stripped = 0
            for param_idx, (param_name, parameter) in enumerate(
                self._iter_parameters()
            ):
                result = (
                    prepared.pop(param_name)
                    if param_name in prepared
                    else self._process(param_name, parameter)
                )
                if result is None or isinstance(result, ExtractionItem):
                    if result is not None:
                        items.append(result)
                    continue

                ptype, weights, handler = result

                param_meta = self._build_metadata(
                    parameter,
                    model_id,
                    param_name,
                    param_idx,
                    ptype,
                    handler,
                    step,
                    instance_uid,
                )
                if "raw_name" in param_meta.other_meta:
                    stripped += 1
                source_dtype = self._source_dtype(parameter)
                storage_dtype = str(weights.dtype)
                if source_dtype is not None:
                    param_meta.other_meta["source_dtype"] = source_dtype
                param_meta.other_meta["storage_dtype"] = storage_dtype
                if instance_uid is None:
                    instance_uid = parameter_repository.instances.get_or_create(
                        param_meta.model_id
                    )
                param_proxy = ParameterDataProxy.create_and_store(
                    meta=param_meta,
                    repository=parameter_repository,
                    instance_uid=instance_uid,
                )
                written.append(param_proxy)
                param_proxy.set_field(WEIGHTS_FIELD, weights, initial=True)
                items.append(
                    ExtractionItem(
                        param_name,
                        "extracted",
                        source_dtype=source_dtype,
                        storage_dtype=storage_dtype,
                        stored_name=param_meta.name,
                    )
                )

                logger.debug(
                    'Extracted parameter "%s" with type %s (handler=%s)',
                    param_meta.name,
                    param_meta.ptype,
                    type(handler).__name__,
                )

            rollback.pop_all()
            if stripped:
                logger.info(
                    "Stripped wrapper prefixes from %d parameter names in model %s",
                    stripped,
                    self.overrides.model_id or model_id,
                )
        return ExtractionReport((*items, *additional))

    def _prepare_required(
        self, names: Collection[str], additional: tuple[ExtractionItem, ...]
    ) -> dict[str, tuple[ParameterType, np.ndarray, ParameterHandler]]:
        """Convert required weights once, before publishing any checkpoint fields."""
        if not names:
            return {}
        pending = dict.fromkeys(names, "absent from input")
        ready = {}
        for item in additional:
            if item.name in pending:
                pending[item.name] = item.reason or item.status
        for name, parameter in self._iter_parameters():
            if name not in pending:
                continue
            result = self._process(name, parameter)
            if isinstance(result, tuple):
                ptype, weights, handler = result
                ready[name] = ptype, weights.copy(order="K"), handler
                del pending[name]
                if not pending:
                    break
            elif result is not None:
                pending[name] = result.reason or result.status
        if pending:
            details = ", ".join(
                f"{key} ({reason})" for key, reason in sorted(pending.items())
            )
            raise ValueError(f"Required parameters could not be extracted: {details}")
        return ready

    def _process(
        self, name: str, parameter: Any
    ) -> tuple[ParameterType, np.ndarray, ParameterHandler] | ExtractionItem | None:
        if is_extra_state_key(name):
            logger.info("Skipping auxiliary state key: %s", name)
            return ExtractionItem(name, "excluded", "auxiliary state")
        try:
            result = self.handler_registry.process_parameter(parameter, name)
        except Exception as exc:
            if not self.skip_not_implemented_types:
                raise
            logger.exception('Handler failed for "%s"', name)
            return ExtractionItem(
                name,
                "failed",
                f"{type(exc).__name__}: {exc}",
                self._source_dtype(parameter),
            )
        if result is not None:
            return result
        if not self.skip_not_implemented_types:
            raise NotImplementedError(
                f'No handler found for parameter "{name}" of type {type(parameter)}'
            )
        return self._unhandled_item(name, parameter)

    def _unhandled_item(self, name: str, parameter: Any) -> ExtractionItem | None:
        if name.rsplit(".", 1)[-1] == "bias":
            return ExtractionItem(
                name, "excluded", "bias", self._source_dtype(parameter)
            )
        return ExtractionItem(
            name,
            "unsupported",
            f"No matching handler for {type(parameter).__name__}",
            self._source_dtype(parameter),
        )

    def _additional_items(self) -> Iterable[ExtractionItem]:
        return ()

    @staticmethod
    def _source_dtype(parameter: Any) -> str | None:
        dtype = getattr(parameter, "dtype", None)
        return None if dtype is None else str(dtype).removeprefix("torch.")

    @staticmethod
    def _discard_partial_extraction(
        repository: IParameterRepository, written: list[ParameterDataProxy]
    ) -> None:
        for proxy in written:
            proxy.erase_field(WEIGHTS_FIELD)
            repository.remove_by_uid(proxy.meta.uid)

    def _build_metadata(
        self,
        parameter: Any,
        model_id: str,
        param_name: str,
        param_idx: int,
        ptype: ParameterType,
        handler: ParameterHandler,
        step: int,
        instance_uid: str | None = None,
    ) -> ParameterMetadata:
        """Build parameter metadata with handler input and overrides.

        Constructs base metadata, enriches it with handler-provided additional
        metadata, and applies any configured overrides.

        Args:
            parameter: The original parameter object.
            model_id: Identifier of the model the parameter belongs to.
            param_name: Name of the parameter.
            param_idx: Index of parameter in model iteration order.
            step: Numeric checkpoint for this parameter.
            instance_uid: Model instance already reserved in the store.
            ptype: Parameter type determined by handler.
            handler: Handler that processed this parameter.

        Returns:
            Complete parameter metadata with overrides applied.
        """
        name = normalize_parameter_name(param_name)
        metadata: dict[str, Any] = {
            "name": name,
            "instance_uid": instance_uid,
            "step": step,
            "ptype": ptype,
            "model_id": model_id,
            "other_meta": {"in_model_idx": param_idx},
        }

        # Best-effort metadata from the selected handler (no exceptions)
        with suppress(Exception):
            extra = handler.get_additional_metadata(parameter, param_name) or {}
            metadata["other_meta"].update(extra)

        self._apply_overrides(metadata, raw_name=param_name)
        metadata["other_meta"]["extractor"] = type(self).__name__
        if name != param_name:
            metadata["other_meta"]["raw_name"] = param_name
        if metadata["name"] != name:
            validate_parameter_name(metadata["name"])
        return ParameterMetadata(**metadata)

    def _apply_overrides(self, metadata: dict[str, Any], *, raw_name: str) -> None:
        """Apply configured overrides to parameter metadata.

        Modifies the metadata dictionary in-place based on configured
        parameter-specific and global overrides. Preserves original
        values in other_meta for traceability.

        Args:
            metadata: Parameter metadata dictionary to modify.
            raw_name: Original tensor path accepted for existing overrides.
        """
        param_name = metadata["name"]

        overrides = self.overrides.parameter_overrides
        param_overrides = overrides.get(param_name) or overrides.get(raw_name)
        if param_overrides is not None:
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
