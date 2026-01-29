"""Cache management module providing flexible caching backends.

This module offers a unified interface for caching with support for multiple
backends including Redis and in-memory implementations. It uses dependency
injection for flexible configuration and backend selection.

Example:
    >>> from diffract.core.cache import CacheContainer
    >>> container = CacheContainer()
    >>> cache_manager = container.cache_manager()
"""

from .containers import CacheContainer
from .interface import UID, ICacheManager
from .redis_manager import RedisLRUCacheManager
from .simple_manager import SimpleLRUCacheManager

__all__ = [
    "UID",
    "CacheContainer",
    "ICacheManager",
    "RedisLRUCacheManager",
    "SimpleLRUCacheManager",
]
