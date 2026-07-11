"""Results namespace for Session."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffract.core.compute.registry import KernelRegistry
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import IParameterView

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class ResultsEraserError(Exception):
    """Error during results erasure."""


class ResultsEraser:
    """Utility for erasing computed results with dependency resolution.

    Handles kernel dependency resolution to determine the full set of
    fields to erase.

    Example:
        >>> eraser = ResultsEraser(kernel_registry=registry)
        >>> fields = eraser.resolve_fields_to_erase(
        ...     ["frob_norm"], erase_dependent_also=True
        ... )
        >>> eraser.erase(view=session._get_view(), fields=fields)
    """

    def __init__(self, kernel_registry: KernelRegistry) -> None:
        """Initialize the eraser.

        Args:
            kernel_registry: Registry for kernel dependency resolution.
        """
        self._registry = kernel_registry

    def resolve_fields_to_erase(
        self,
        fields_to_erase: Iterable[str],
        erase_dependent_also: bool,
    ) -> set[str]:
        """Resolve kernel dependencies to get full set of fields to erase.

        Args:
            fields_to_erase: Initial fields to erase.
            erase_dependent_also: If True, include fields that depend
                on the specified fields.

        Returns:
            Complete set of fields to erase.

        Raises:
            ResultsEraserError: If a field cannot be produced by any kernel.
        """
        result: list[str] = []

        for field_name in fields_to_erase:
            if not self._registry.can_produce_field(field_name):
                msg = f"Registry cannot produce '{field_name}'"
                raise ResultsEraserError(msg)

            result.append(field_name)

            if erase_dependent_also:
                result.extend(self._find_dependent_fields(field_name))

        return set(result)

    def _find_dependent_fields(self, field_name: str) -> list[str]:
        """Find all fields that depend on the given field."""
        dependent_fields: list[str] = []

        producer = self._registry.get_kernel_producing_field(field_name)
        all_kernels = self._registry.list_kernels()

        for other_kernel in all_kernels:
            if other_kernel == producer:
                continue

            dependencies = self._registry.resolve_dependencies(other_kernel)
            fields_dependencies: list[str] = []

            for dependency in dependencies:
                fields_dependencies.extend(
                    self._registry.get_fields_kernel_require(dependency)
                )

            if field_name in fields_dependencies:
                dependent_fields.extend(
                    self._registry.get_fields_kernel_produce(other_kernel)
                )

        return dependent_fields

    def erase(
        self,
        *,
        parameters: IParameterView,
        aggregates: AggregateView | None = None,
        fields: set[str],
    ) -> None:
        """Erase fields from parameters and aggregates.

        Args:
            parameters: Parameter view to erase scalar fields from.
            aggregates: Aggregate view to erase aggregate entries from.
            fields: Set of base field names to erase.
        """
        logger.debug("Erasing fields: %s", ", ".join(sorted(fields)))

        # Erase scalar fields from parameters
        re_patterns = [rf"{re.escape(f)}" for f in fields]
        parameters.erase_fields_with_regexp(*re_patterns)

        # Erase aggregate entries by field_name
        if aggregates is not None:
            aggregates_to_erase = aggregates.filter_by_field_name(*fields)
            if aggregates_to_erase:
                logger.debug("Erasing %d aggregate entries", len(aggregates_to_erase))
                aggregates_to_erase.clear(erase=True)
