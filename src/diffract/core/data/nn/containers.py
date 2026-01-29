"""Dependency injection containers for model parameter management.

This module provides dependency injection configuration for model parameter
extraction and collection management using the dependency-injector library.
It integrates parameter extractors, storage managers, metadata index, and
cache managers for comprehensive parameter lifecycle management.

The container provides factory methods for creating parameter extractors
and initializing parameter collections, ensuring proper dependency
injection and configuration management.

Example:
    >>> container = ModelParametersContainer()
    >>> container.config.extractor.framework.from_value("pytorch")
    >>> container.storage_manager.override(storage_instance)
    >>> container.metadata_index.override(metadata_instance)
    >>> container.cache_manager.override(cache_instance)
    >>> collection = container.parameter_repository()
"""

from __future__ import annotations

from dependency_injector import containers, providers

from diffract.core.compute.parallel import ParallelContext
from diffract.core.metadata.interface import IMetadataIndex
from diffract.core.storage.interface import IStorageManager
from diffract.core.utils.build import build_with_defaults

from .aggregates.repository import AggregateRepository
from .extractors.factory import create_extractor
from .params.repository import ParameterRepository


class ModelParametersContainer(containers.DeclarativeContainer):
    """Dependency injection container for model parameter components.

    Provides comprehensive dependency injection setup for parameter extraction,
    collection management, and data lifecycle operations. The container manages
    the integration between parameter extractors, storage systems, metadata index,
    and caching mechanisms.

    The container uses external dependencies for storage, metadata index, and
    cache managers, allowing flexible backend configuration while maintaining
    consistent parameter extraction and collection interfaces.

    Attributes:
        config: Configuration provider for parameter extraction settings.
        storage_manager: External dependency for persistent storage.
        metadata_index: External dependency for structured metadata.
        cache_manager: External dependency for temporary caching.
        extractor_factory: Factory for creating parameter extractors.
        parameter_repository: Singleton for parameter repository.
        aggregate_repository: Singleton for aggregate repository.
    """

    config = providers.Configuration()

    # External dependencies (injected from parent containers)
    storage_manager = providers.Dependency(instance_of=IStorageManager)
    metadata_index = providers.Dependency(instance_of=IMetadataIndex)
    cache_manager = providers.Dependency()
    parallel = providers.Dependency(instance_of=ParallelContext)

    parameter_repository = providers.Singleton(
        ParameterRepository.initialize,
        storage_manager=storage_manager,
        metadata_index=metadata_index,
        cache_manager=cache_manager,
        parallel=parallel,
    )

    aggregate_repository = providers.Singleton(
        AggregateRepository.initialize,
        storage_manager=storage_manager,
        metadata_index=metadata_index,
        cache_manager=cache_manager,
        parallel=parallel,
    )

    # Parameter extractor factories
    extractor_factory = providers.Factory(
        build_with_defaults,
        create_extractor,
        config.extractor.as_(dict),
    )
