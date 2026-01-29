"""Dependency injection containers for metadata index management.

This module provides dependency injection configuration for metadata index
using the dependency-injector library.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from dependency_injector import containers, providers

from diffract.core.utils.build import build_with_defaults

from .interface import IMetadataIndex
from .sqlite_index import SQLiteMetadataIndex

logger = logging.getLogger(__name__)


class MetadataContainer(containers.DeclarativeContainer):
    """Dependency injection container for metadata index components.

    Provides configurable metadata index selection between different
    backends. Currently supports SQLite for structured metadata storage.
    """

    config = providers.Configuration()

    sqlite_index = providers.Singleton(
        build_with_defaults,
        SQLiteMetadataIndex,
        config.sqlite.as_(dict),
    )

    metadata_index = providers.Selector(
        config.backend,
        sqlite=sqlite_index,
    )

    metadata_index_resource = providers.Resource(
        lambda index: _connect_wrapper(index),
        metadata_index,
    )


@contextmanager
def _connect_wrapper(index: IMetadataIndex) -> Generator[IMetadataIndex, None, None]:
    logger.debug("Init resource: %s", index.__class__.__name__)
    if hasattr(index, "connect"):
        logger.debug("Init connection for: %s", index.__class__.__name__)
        index.connect()
    try:
        yield index
    finally:
        if hasattr(index, "close") and getattr(index, "need_close_until_del", True):
            logger.debug("Close connection for: %s", index.__class__.__name__)
            index.close()
