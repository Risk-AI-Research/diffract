"""Redis-based LRU cache manager implementation.

This module provides a Redis-backed cache manager that implements the
ICacheManager protocol. It uses Redis as a RAM-only volatile cache with
LRU eviction and configurable memory limits.

Features:
    - RAM-only storage with LRU eviction
    - Configurable memory limits and TTL
    - Safe typed serialization via the shared storage codec
    - Connection pooling and error handling
    - Health monitoring and memory usage tracking

Example:
    >>> manager = RedisLRUCacheManager(host="localhost", max_memory_mb=512)
    >>> manager.set_field("user123", "profile", user_data)
    >>> profile = manager.get_field("user123", "profile")
"""

from __future__ import annotations

import contextlib
import logging
from types import TracebackType
from typing import Any, Self

import diffract.core.utils.imports as import_utils
from diffract.core.storage.serialization import decode_value, encode_value
from diffract.core.utils.exceptions import format_exception_message

from .interface import UID, ICacheManager

logger = logging.getLogger(__name__)

_TAG_HEADER_BYTES = 1
_DECODE_MISS_ERRORS = (
    ValueError,
    EOFError,
    OSError,
    IndexError,
    UnicodeDecodeError,
)


def _frame(payload: bytes, tag: str) -> bytes:
    """Prefix a codec payload with its tag so the tag survives the round trip."""
    tag_bytes = tag.encode("utf-8")
    return bytes((len(tag_bytes),)) + tag_bytes + payload


def _unframe(data: bytes) -> tuple[bytes, str]:
    """Split a framed blob back into its codec payload and tag."""
    tag_len = data[0]
    tag = data[_TAG_HEADER_BYTES : _TAG_HEADER_BYTES + tag_len].decode("utf-8")
    payload = data[_TAG_HEADER_BYTES + tag_len :]
    return payload, tag


if not import_utils.is_available("redis"):
    logger.debug("Redis not available, disabling Redis cache manager")

    class RedisLRUCacheManager(ICacheManager):
        """Stub implementation when Redis is not available.

        Raises ImportError when instantiated to indicate Redis dependency
        is missing.
        """

        def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
            """Raise ImportError because the optional dependency is missing."""
            msg = "Redis package not available"
            raise ImportError(msg)

else:
    redis = import_utils.require("redis")
    RedisConnectionError = redis.exceptions.ConnectionError
    RedisError = redis.exceptions.RedisError

    class RedisLRUCacheManager(ICacheManager):
        """Redis-based LRU cache manager with RAM-only storage.

        Implements ICacheManager using Redis as the backend storage with
        LRU eviction policy and configurable memory limits. Designed for
        high-performance caching with minimal persistence overhead.

        The manager configures Redis for RAM-only operation by disabling
        RDB snapshots and AOF logging, focusing on speed over durability.
        Cached values are serialized through the shared storage codec.

        Attributes:
            _key_prefix: Prefix for all Redis keys to avoid collisions.
            _ttl_seconds: Time-to-live for cache entries in seconds.
            _max_memory_mb: Maximum memory usage in megabytes.
            _pool: Redis connection pool for efficient connection management.
            _redis: Redis client instance.
        """

        def __init__(
            self,
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: str | None = None,
            *,
            max_memory_mb: int = 256,
            ttl_seconds: int | None = None,
            key_prefix: str = "diffract:cache:",
            socket_timeout: float = 5.0,
            socket_connect_timeout: float = 5.0,
            retry_on_timeout: bool = True,
            max_connections: int = 64,
        ) -> None:
            """Initialize Redis LRU cache manager.

            Args:
                host: Redis server hostname.
                port: Redis server port number.
                db: Redis database number to use.
                password: Redis authentication password.
                max_memory_mb: Maximum memory usage in MB (must be positive).
                ttl_seconds: Time-to-live for entries in seconds (None = no expiry).
                key_prefix: Prefix for all Redis keys.
                socket_timeout: Socket timeout in seconds.
                socket_connect_timeout: Connection timeout in seconds.
                retry_on_timeout: Whether to retry on timeout.
                max_connections: Maximum connections in pool.

            Raises:
                ValueError: If max_memory_mb is not positive.
                RedisConnectionError: If Redis connection fails.
            """
            if max_memory_mb <= 0:
                msg = "max_memory_mb must be positive"
                raise ValueError(msg)

            self._key_prefix = key_prefix
            self._ttl_seconds = ttl_seconds
            self._max_memory_mb = max_memory_mb

            self._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=retry_on_timeout,
                max_connections=max_connections,
                decode_responses=False,
            )
            self._redis = redis.Redis(connection_pool=self._pool)

            self._initialize_redis()

        def _initialize_redis(self) -> None:
            """Configure Redis for RAM-only operation with LRU eviction.

            Sets up Redis configuration for optimal caching performance by
            disabling persistence and configuring memory limits.

            Raises:
                RedisConnectionError: If Redis initialization fails.
            """
            try:
                self._redis.ping()

                # Disable persistence for RAM-only operation
                try:
                    self._redis.config_set("save", "")
                    self._redis.config_set("appendonly", "no")
                except RedisError as e:
                    logger.warning(
                        "Failed to disable Redis persistence: %s",
                        format_exception_message(e),
                    )

                # Configure memory limits and LRU eviction
                self._redis.config_set("maxmemory", self._max_memory_mb * 1024 * 1024)
                self._redis.config_set("maxmemory-policy", "allkeys-lru")

                logger.info(
                    "Redis cache initialized (max_memory=%d MB, ttl=%s)",
                    self._max_memory_mb,
                    self._ttl_seconds,
                )
            except RedisError as e:
                msg = f"Redis initialization failed: {format_exception_message(e)}"
                raise RedisConnectionError(msg) from e

        def _make_key(self, obj_uid: UID, field_name: str) -> bytes:
            """Generate Redis key for object field.

            Args:
                obj_uid: Unique identifier for the cached object.
                field_name: Name of the field.

            Returns:
                Encoded Redis key as bytes.
            """
            return f"{self._key_prefix}{obj_uid}:{field_name}".encode()

        def has_field(self, obj_uid: UID, field_name: str) -> bool:
            """Check if an object has a specific field in cache.

            Args:
                obj_uid: Unique identifier for the cached object.
                field_name: Name of the field to check.

            Returns:
                True if the field exists, False otherwise or on error.
            """
            try:
                return bool(self._redis.exists(self._make_key(obj_uid, field_name)))
            except RedisError as e:
                logger.debug(
                    "Redis has_field error for %s/%s: %s",
                    obj_uid,
                    field_name,
                    format_exception_message(e),
                    exc_info=True,
                )
                return False

        def get_field(self, obj_uid: UID, field_name: str) -> Any:
            """Retrieve a field value from cache.

            Args:
                obj_uid: Unique identifier for the cached object.
                field_name: Name of the field to retrieve.

            Returns:
                The cached value or None if field doesn't exist or on error.
            """
            try:
                key = self._make_key(obj_uid, field_name)
                data = self._redis.get(key)
                if data is None:
                    return None
                try:
                    payload, tag = _unframe(data)
                    return decode_value(payload, tag)
                except _DECODE_MISS_ERRORS:
                    with contextlib.suppress(RedisError):
                        self._redis.delete(key)
                    return None
            except RedisError as e:
                logger.debug(
                    "Redis get_field error for %s/%s: %s",
                    obj_uid,
                    field_name,
                    format_exception_message(e),
                    exc_info=True,
                )
                return None

        def list_objs_has_field(self, field_name: str) -> list[UID]:
            """List all object UIDs that have a specific field.

            Uses Redis SCAN command for non-blocking iteration over keys.

            Args:
                field_name: Name of the field to search for.

            Returns:
                List of UIDs for objects that have the specified field.
            """
            prefix = self._key_prefix.encode("utf-8")
            suffix = f":{field_name}".encode()
            pattern = prefix + b"*" + suffix

            uids: list[str] = []
            cursor: int = 0
            try:
                while True:
                    cursor, keys = self._redis.scan(
                        cursor=cursor, match=pattern, count=1000
                    )

                    pre_len, suf_len = len(prefix), len(suffix)

                    for k in keys:
                        if k.startswith(prefix) and k.endswith(suffix):
                            uid_bytes = k[pre_len:-suf_len]
                            if uid_bytes:
                                uids.append(uid_bytes.decode("utf-8"))

                    if cursor == 0:
                        break
            except RedisError as e:
                logger.debug(
                    "Redis list_objs_has_field error for %s: %s",
                    field_name,
                    format_exception_message(e),
                    exc_info=True,
                )
                return []
            else:
                return uids

        def set_field(self, obj_uid: UID, field_name: str, value: Any) -> None:
            """Store a field value in cache.

            Args:
                obj_uid: Unique identifier for the cached object.
                field_name: Name of the field to store.
                value: Value to cache. Supported kinds are NumPy arrays,
                    ``bytes``, and JSON-serializable values.

            Raises:
                ValueError: If the value is not a supported kind.
                RedisError: If storage operation fails.
            """
            blob = _frame(*encode_value(value))
            try:
                key = self._make_key(obj_uid, field_name)

                if self._ttl_seconds is not None:
                    self._redis.setex(key, self._ttl_seconds, blob)
                else:
                    self._redis.set(key, blob)

            except RedisError:
                logger.exception("Redis set_field error")
                raise

        def erase_field(self, obj_uid: UID, field_name: str) -> None:
            """Remove a specific field from cache.

            Args:
                obj_uid: Unique identifier for the cached object.
                field_name: Name of the field to remove.

            Raises:
                RedisError: If removal operation fails.
            """
            try:
                self._redis.delete(self._make_key(obj_uid, field_name))
            except RedisError:
                logger.exception("Redis erase_field error")
                raise

        def erase_field_for_all(self, field_name: str) -> None:
            """Remove a field from all cached objects.

            Args:
                field_name: Name of the field to remove from all objects.

            Raises:
                RedisError: If removal operation fails.
            """
            try:
                prefix = self._key_prefix.encode("utf-8")
                suffix = f":{field_name}".encode()
                pattern = prefix + b"*" + suffix
                cursor: int = 0
                keys_to_delete: list[bytes] = []
                while True:
                    cursor, keys = self._redis.scan(
                        cursor=cursor, match=pattern, count=1000
                    )
                    keys_to_delete.extend(keys)
                    if cursor == 0:
                        break
                if keys_to_delete:
                    self._redis.delete(*keys_to_delete)
            except RedisError:
                logger.exception("Redis erase_field_for_all error")
                raise

        def clear(self) -> None:
            """Remove all cached data with this manager's prefix.

            Uses batched deletion to avoid blocking Redis for large datasets.

            Raises:
                RedisError: If clear operation fails.
            """
            try:
                pattern = f"{self._key_prefix}*".encode()

                cursor: int = 0
                batch: list[bytes] = []
                delete_batch_size = 5000
                while True:
                    cursor, keys = self._redis.scan(
                        cursor=cursor, match=pattern, count=2000
                    )
                    batch.extend(keys)
                    if len(batch) >= delete_batch_size:
                        self._redis.delete(*batch)
                        batch.clear()
                    if cursor == 0:
                        break

                if batch:
                    self._redis.delete(*batch)
            except RedisError:
                logger.exception("Redis clear error")
                raise

        def get_memory_usage(self) -> dict[str, Any]:
            """Get Redis memory usage statistics.

            Returns:
                Dictionary with memory usage metrics including used memory,
                max memory, evicted keys, and fragmentation ratio.
            """
            try:
                info = self._redis.info("memory")
                used = float(info.get("used_memory", 0))
                return {
                    "used_memory_mb": used / (1024 * 1024),
                    "max_memory_mb": float(self._max_memory_mb),
                    "evicted_keys": info.get("evicted_keys"),
                    "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio"),
                }
            except RedisError as e:
                logger.warning(
                    "Redis get_memory_usage error: %s",
                    format_exception_message(e),
                    exc_info=True,
                )
                return {}

        def get_available_bytes(self) -> int | None:
            """Return estimated available cache capacity in bytes."""
            info = self.get_memory_usage()
            if not info:
                return None
            used = info.get("used_memory_mb", 0) * 1024 * 1024
            max_mem = info.get("max_memory_mb", 0) * 1024 * 1024
            return max(0, int(max_mem - used))

        def health_check(self) -> dict[str, Any]:
            """Perform Redis health check.

            Returns:
                Dictionary with health status and error details if unhealthy.
            """
            try:
                ok = self._redis.ping()
            except RedisError as e:
                return {
                    "status": "unhealthy",
                    "error": format_exception_message(e),
                }
            else:
                return {"status": "healthy" if ok else "unhealthy"}

        def close(self) -> None:
            """Close Redis connection pool.

            Safely disconnects all connections in the pool. Should be called
            when the cache manager is no longer needed.
            """
            with contextlib.suppress(Exception):
                self._pool.disconnect()

        def __enter__(self) -> Self:
            """Context manager entry.

            Returns:
                Self for use in with statements.
            """
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            """Context manager exit.

            Automatically closes the connection pool when exiting context.

            Args:
                exc_type: Exception type if an exception occurred.
                exc: Exception instance if an exception occurred.
                tb: Traceback if an exception occurred.
            """
            self.close()
