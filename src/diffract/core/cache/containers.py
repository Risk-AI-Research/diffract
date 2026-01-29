"""Dependency injection containers for cache management.

This module provides dependency injection configuration for cache managers
using the dependency-injector library. It supports multiple cache backends
with configurable selection.

Example:
    >>> container = CacheContainer()
    >>> container.config.backend.from_value("redis")
    >>> cache_manager = container.cache_manager()
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from dependency_injector import containers, providers

from diffract.core.utils.build import build_with_defaults

from .redis_manager import RedisLRUCacheManager
from .simple_manager import SimpleLRUCacheManager

if TYPE_CHECKING:
    from .interface import ICacheManager


logger = logging.getLogger(__name__)


class CacheContainer(containers.DeclarativeContainer):
    """Dependency injection container for cache-related components.

    Provides configurable cache manager selection between Redis and simple
    in-memory implementations. Configuration is handled through the config
    provider with backend selection and specific backend settings.

    Attributes:
        config: Configuration provider for cache settings.
        redis_manager: Singleton Redis cache manager provider.
        simple_manager: Singleton simple cache manager provider.
        cache_manager: Selector for active cache backend.
    """

    config = providers.Configuration()

    redis_manager = providers.Singleton(
        build_with_defaults,
        RedisLRUCacheManager,
        config.redis.as_(dict),
    )

    simple_manager = providers.Singleton(
        build_with_defaults,
        SimpleLRUCacheManager,
        config.simple.as_(dict),
    )

    none_manager = providers.Singleton(lambda: None)

    cache_manager = providers.Selector(
        config.backend,
        redis=redis_manager,
        simple=simple_manager,
        none=none_manager,
    )

    cache_manager_resource = providers.Resource(
        lambda manager: _connect_wrapper(manager),
        cache_manager,
    )


@contextmanager
def _connect_wrapper(manager: ICacheManager | None) -> Iterator[ICacheManager | None]:
    if manager is None:
        logger.debug("Cache disabled (backend=none)")
        yield None
        return
    else:
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
