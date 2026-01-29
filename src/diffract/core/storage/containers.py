"""Dependency injection containers for storage management.

This module provides dependency injection configuration for storage managers
using the dependency-injector library. It supports multiple storage backends
with configurable selection for persistent data storage.

Example:
    >>> container = StorageContainer()
    >>> container.config.backend.from_value("hdf5")
    >>> container.config.hdf5.path.from_value("/path/to/storage.h5")
    >>> storage_manager = container.storage_manager()
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from dependency_injector import containers, providers

from diffract.core.utils.build import build_with_defaults

from .hdf5_manager import HDF5StorageManager
from .hybrid_manager import HybridStorageManager
from .interface import IStorageManager
from .ram_manager import RAMStorageManager
from .sqlite_manager import SQLiteStorageManager
from .zarr_manager import ZarrStorageManager

logger = logging.getLogger(__name__)


class StorageContainer(containers.DeclarativeContainer):
    """Dependency injection container for storage-related components.

    Provides configurable storage manager selection between different
    persistent storage backends. Currently supports HDF5 for hierarchical
    data storage with compression and atomic operations.

    The container manages storage backend lifecycle and configuration,
    ensuring proper initialization and resource management.

    Attributes:
        config: Configuration provider for storage settings.
        hdf5_manager: Singleton HDF5 storage manager provider.
        storage_manager: Selector for active storage backend.
    """

    config = providers.Configuration()

    hdf5_manager = providers.Singleton(
        build_with_defaults,
        HDF5StorageManager,
        config.hdf5.as_(dict),
    )

    sqlite_manager = providers.Singleton(
        build_with_defaults,
        SQLiteStorageManager,
        config.sqlite.as_(dict),
    )

    zarr_manager = providers.Singleton(
        build_with_defaults,
        ZarrStorageManager,
        config.zarr.as_(dict),
    )

    ram_manager = providers.Singleton(RAMStorageManager)

    # Selectors for hybrid storage backends
    _light_backends = providers.Selector(
        config.hybrid.light,
        sqlite=sqlite_manager,
        ram=ram_manager,
        hdf5=hdf5_manager,
        zarr=zarr_manager,
    )

    _heavy_backends = providers.Selector(
        config.hybrid.heavy,
        hdf5=hdf5_manager,
        zarr=zarr_manager,
        sqlite=sqlite_manager,
        ram=ram_manager,
    )

    hybrid_manager = providers.Singleton(
        HybridStorageManager,
        light_storage=_light_backends,
        heavy_storage=_heavy_backends,
        array_threshold=config.hybrid.array_threshold,
    )

    storage_manager = providers.Selector(
        config.backend,
        hdf5=hdf5_manager,
        ram=ram_manager,
        sqlite=sqlite_manager,
        hybrid=hybrid_manager,
        zarr=zarr_manager,
    )

    storage_manager_resource = providers.Resource(
        lambda manager: _connect_wrapper(manager),
        storage_manager,
    )


@contextmanager
def _connect_wrapper(manager: IStorageManager) -> Generator[None, None, None]:
    logger.debug("Init resource: %s", manager.__class__.__name__)
    if hasattr(manager, "connect"):
        logger.debug("Init connection for: %s", manager.__class__.__name__)
        manager.connect()
    try:
        yield manager
    finally:
        if hasattr(manager, "close"):
            logger.debug("Close connection for: %s", manager.__class__.__name__)
            manager.close()
