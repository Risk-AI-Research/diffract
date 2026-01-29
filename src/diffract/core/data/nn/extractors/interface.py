"""Parameter extractor interface and protocols.

This module defines the core protocol that all parameter extractor
implementations must follow. It provides a framework-agnostic contract
for parameter extraction operations from neural network models.

The interface ensures consistent parameter extraction behavior across
different deep learning frameworks while allowing framework-specific
optimizations and parameter handling strategies.

Example:
    >>> extractor: IParameterExtractor = get_extractor(model)
    >>> parameters = extractor.extract_parameters(storage, cache)
    >>> print(f"Extracted {len(parameters)} parameters")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterRepository


@runtime_checkable
class IParameterExtractor(Protocol):
    """Protocol defining the interface for neural network parameter extractors.

    Provides a unified interface for extracting parameters from neural network
    models across different deep learning frameworks. All extractor implementations
    must implement this protocol to ensure consistent behavior.

    The extraction process should traverse the model structure and extract all
    relevant parameters along with their metadata, returning a standardized
    parameter collection that can be used across the diffract system.
    """

    def extract_parameters(
        self,
        parameter_repository: IParameterRepository,
    ) -> None:
        """Extract all parameters from the neural network model.

        Traverses the model structure and extracts all relevant weight parameters
        along with their metadata. The extraction should be framework-specific
        and add standardized ParameterDataProxy objects to the repository.

        Args:
            parameter_repository: Repository to store extracted parameters.

        Raises:
            NotImplementedError: If parameter type is not supported.
            ValueError: If model structure is invalid.
            RuntimeError: If extraction process fails.
        """
        ...
