"""In-memory LRU cache manager implementation.

This module provides a pure Python in-memory cache manager that implements
the ICacheManager protocol. It uses OrderedDict for LRU tracking and
pickle for size estimation, making it suitable for development and testing
environments where Redis is not available.

Features:
    - Process-local LRU cache with configurable memory limits
    - Optional TTL support for cache entries
    - Thread-safe via GIL (single-threaded access)
    - Pickle-based size estimation for memory accounting
    - No external dependencies

Limitations:
    - Process-local only (not suitable for multi-process production)
    - Approximate memory accounting using pickle size
    - Single-threaded performance characteristics

Example:
    >>> manager = SimpleLRUCacheManager(max_memory_mb=128, ttl_seconds=3600)
    >>> manager.set_field("user123", "profile", user_data)
    >>> profile = manager.get_field("user123", "profile")
"""

from __future__ import annotations

import pickle
import threading
import time
from collections import OrderedDict
from typing import Any

from .interface import UID, ICacheManager


class SimpleLRUCacheManager(ICacheManager):
    """In-memory LRU cache manager with TTL support and thread-safety.

    Implements ICacheManager using pure Python data structures for caching.
    Uses OrderedDict to maintain LRU order and pickle serialization for
    memory size calculation with overhead accounting.

    This implementation is designed for single-process applications and
    development environments. For production multi-process setups,
    use RedisLRUCacheManager instead.

    Features:
        - Thread-safe operations via internal locking
        - Accurate memory accounting with overhead factor
        - Efficient field-based indexing for fast lookups
        - Automatic cleanup of expired entries
        - LRU eviction with configurable memory limits

    Attributes:
        _max_bytes: Maximum cache size in bytes.
        _ttl: Time-to-live for cache entries in seconds.
        _key_prefix: Prefix for cache keys.
        _store: Main storage mapping keys to (data, timestamp) tuples.
        _lru: OrderedDict tracking LRU order of keys.
        _field_index: Index mapping field names to sets of UIDs.
        _current_bytes: Current estimated cache size in bytes.
        _lock: Thread lock for thread-safe operations.
    """

    def __init__(
        self,
        *,
        max_memory_mb: int = 256,
        ttl_seconds: int | None = None,
        key_prefix: str = "diffract:cache:",
        memory_overhead_factor: float = 1.1,
    ) -> None:
        """Initialize the simple LRU cache manager.

        Args:
            max_memory_mb: Maximum cache size in MB (must be positive).
            ttl_seconds: Time-to-live for entries in seconds (None = no expiry).
            key_prefix: Prefix for cache keys to avoid collisions.
            memory_overhead_factor: Factor to account for Python object overhead
                (dict entries, OrderedDict entries, string keys, metadata).

        Raises:
            ValueError: If max_memory_mb is not positive or memory_overhead_factor <= 0.
        """
        if max_memory_mb <= 0:
            msg = "max_memory_mb must be positive"
            raise ValueError(msg)

        if memory_overhead_factor <= 0:
            msg = "memory_overhead_factor must be positive"
            raise ValueError(msg)

        self._max_bytes = max_memory_mb * 1024 * 1024
        self._ttl = ttl_seconds
        self._key_prefix = key_prefix
        self._memory_overhead_factor = memory_overhead_factor

        self._store: dict[str, tuple[bytes, float]] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._field_index: dict[str, set[str]] = {}
        self._current_bytes = 0
        self._lock = threading.Lock()

    def _make_key(self, obj_uid: UID, field_name: str) -> str:
        """Generate cache key for object field.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field.

        Returns:
            String cache key.
        """
        return f"{self._key_prefix}{obj_uid}:{field_name}"

    def _calculate_entry_size(self, key: str, data: bytes) -> int:
        """Calculate estimated memory size of a cache entry.

        Accounts for key string, data bytes, timestamp (float), tuple overhead,
        and Python object overhead using a multiplier factor.

        Args:
            key: Cache key string.
            data: Pickled data bytes.

        Returns:
            Estimated memory size in bytes.
        """
        # Base size: key length + data length + timestamp (8 bytes for float)
        base_size = len(key.encode("utf-8")) + len(data) + 8
        # Apply overhead factor for Python object overhead
        return int(base_size * self._memory_overhead_factor)

    def _extract_uid_field(self, key: str) -> tuple[str, str] | None:
        """Extract UID and field name from cache key.

        Args:
            key: Cache key string.

        Returns:
            Tuple of (uid, field_name) or None if key format is invalid.
        """
        if not key.startswith(self._key_prefix):
            return None
        suffix = key[len(self._key_prefix) :]
        if ":" not in suffix:
            return None
        uid, field_name = suffix.split(":", 1)
        return uid, field_name

    def _update_field_index(self, obj_uid: str, field_name: str, adding: bool) -> None:
        """Update the field index when adding or removing entries.

        Args:
            obj_uid: Object UID.
            field_name: Field name.
            adding: True if adding entry, False if removing.
        """
        if adding:
            if field_name not in self._field_index:
                self._field_index[field_name] = set()
            self._field_index[field_name].add(obj_uid)
        elif field_name in self._field_index:
            self._field_index[field_name].discard(obj_uid)
            if not self._field_index[field_name]:
                del self._field_index[field_name]

    def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from cache.

        Iterates through all entries and removes those that have expired.
        This is called periodically to prevent accumulation of expired entries.
        """
        if self._ttl is None:
            return

        now = time.time()
        expired_keys = []

        for key, (_data, ts) in self._store.items():
            if (now - ts) > self._ttl:
                expired_keys.append(key)

        for key in expired_keys:
            data, _ = self._store.pop(key, (b"", 0.0))
            entry_size = self._calculate_entry_size(key, data)
            self._current_bytes -= entry_size
            self._lru.pop(key, None)

            # Update field index
            uid_field = self._extract_uid_field(key)
            if uid_field:
                uid, field_name = uid_field
                self._update_field_index(uid, field_name, adding=False)

    def _expired(self, ts: float) -> bool:
        """Check if a timestamp indicates an expired entry.

        Args:
            ts: Timestamp to check against current time.

        Returns:
            True if entry is expired, False otherwise.
        """
        return self._ttl is not None and (time.time() - ts) > self._ttl

    def _touch_lru(self, key: str) -> None:
        """Update LRU position for a key.

        Moves key to end of LRU order (most recently used position).
        If key doesn't exist in LRU tracking, adds it.

        Args:
            key: Cache key to update LRU position for.
        """
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None

    def _evict_until_fit(self) -> None:
        """Evict least recently used entries until under memory limit.

        Removes entries from the beginning of LRU order until the current
        cache size is within the configured memory limit.
        """
        while self._current_bytes > self._max_bytes and self._lru:
            oldest_key, _ = self._lru.popitem(last=False)
            data, _ = self._store.pop(oldest_key, (b"", 0.0))
            entry_size = self._calculate_entry_size(oldest_key, data)
            self._current_bytes -= entry_size

            # Update field index
            uid_field = self._extract_uid_field(oldest_key)
            if uid_field:
                uid, field_name = uid_field
                self._update_field_index(uid, field_name, adding=False)

    def has_field(self, obj_uid: UID, field_name: str) -> bool:
        """Check if an object has a specific field in cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to check.

        Returns:
            True if the field exists and is not expired, False otherwise.
        """
        key = self._make_key(obj_uid, field_name)
        with self._lock:
            item = self._store.get(key)

            if not item:
                return False

            data, ts = item
            if self._expired(ts):
                entry_size = self._calculate_entry_size(key, data)
                self._store.pop(key, None)
                self._lru.pop(key, None)
                self._current_bytes -= entry_size
                self._update_field_index(obj_uid, field_name, adding=False)
                return False

            return True

    def get_field(self, obj_uid: UID, field_name: str) -> Any:
        """Retrieve a field value from cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to retrieve.

        Returns:
            The cached value or None if field doesn't exist or is expired.
        """
        key = self._make_key(obj_uid, field_name)
        with self._lock:
            item = self._store.get(key)

            if not item:
                return None

            data, ts = item

            if self._expired(ts):
                entry_size = self._calculate_entry_size(key, data)
                self._store.pop(key, None)
                self._lru.pop(key, None)
                self._current_bytes -= entry_size
                self._update_field_index(obj_uid, field_name, adding=False)
                return None

            self._touch_lru(key)

            try:
                return pickle.loads(data)  # noqa: S301
            except (
                ModuleNotFoundError,
                ImportError,
                AttributeError,
                ValueError,
                pickle.UnpicklingError,
                EOFError,
            ):
                # Cache may contain pickles produced by a different Python/numpy
                # version. Treat as a cache miss and delete the corrupted entry.
                entry_size = self._calculate_entry_size(key, data)
                self._store.pop(key, None)
                self._lru.pop(key, None)
                self._current_bytes -= entry_size
                self._update_field_index(obj_uid, field_name, adding=False)
                return None

    def list_objs_has_field(self, field_name: str) -> list[UID]:
        """List all object UIDs that have a specific field.

        Uses field index for efficient lookup when available, falling back
        to iteration through all keys with cleanup of expired entries.

        Args:
            field_name: Name of the field to search for.

        Returns:
            List of UIDs for objects that have the specified field.
        """
        with self._lock:
            # Use field index for efficient lookup if available
            if field_name in self._field_index:
                # Clean up expired entries for this field
                valid_uids = []
                for uid in self._field_index[field_name]:
                    key = self._make_key(uid, field_name)
                    item = self._store.get(key)
                    if item:
                        data, ts = item
                        if not self._expired(ts):
                            valid_uids.append(uid)
                        else:
                            # Clean up expired entry
                            entry_size = self._calculate_entry_size(key, data)
                            self._store.pop(key, None)
                            self._lru.pop(key, None)
                            self._current_bytes -= entry_size
                # Update index if some entries were expired
                if len(valid_uids) != len(self._field_index[field_name]):
                    self._field_index[field_name] = set(valid_uids)
                return valid_uids

            # Fallback: iterate through all keys
            suffix = f":{field_name}"
            result: list[str] = []
            now = time.time()

            # Iterate over snapshot to avoid mutation during iteration
            for key in tuple(self._store.keys()):
                if not key.endswith(suffix) or not key.startswith(self._key_prefix):
                    continue

                data, ts = self._store.get(key, (b"", 0.0))
                if self._ttl is not None and (now - ts) > self._ttl:
                    entry_size = self._calculate_entry_size(key, data)
                    self._store.pop(key, None)
                    self._lru.pop(key, None)
                    self._current_bytes -= entry_size
                    # Update field index
                    uid_field = self._extract_uid_field(key)
                    if uid_field:
                        uid, _ = uid_field
                        self._update_field_index(uid, field_name, adding=False)
                    continue

                uid = key[len(self._key_prefix) : -len(suffix)]
                if uid:
                    result.append(uid)

            return result

    def set_field(self, obj_uid: UID, field_name: str, value: Any) -> None:
        """Store a field value in cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to store.
            value: Value to cache (must be pickle-serializable).
        """
        key = self._make_key(obj_uid, field_name)
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        entry_size = self._calculate_entry_size(key, data)

        with self._lock:
            # Remove old entry size if updating existing key
            old = self._store.get(key)
            if old:
                old_data, _ = old
                old_entry_size = self._calculate_entry_size(key, old_data)
                self._current_bytes -= old_entry_size

            self._store[key] = (data, time.time())
            self._current_bytes += entry_size
            self._touch_lru(key)
            self._update_field_index(obj_uid, field_name, adding=True)

            # Periodic cleanup of expired entries (every 100 operations)
            if len(self._store) % 100 == 0:
                self._cleanup_expired_entries()

            self._evict_until_fit()

    def erase_field(self, obj_uid: UID, field_name: str) -> None:
        """Remove a specific field from cache.

        Args:
            obj_uid: Unique identifier for the cached object.
            field_name: Name of the field to remove.
        """
        key = self._make_key(obj_uid, field_name)
        with self._lock:
            item = self._store.pop(key, None)
            if item:
                data, _ = item
                entry_size = self._calculate_entry_size(key, data)
                self._current_bytes -= entry_size
                self._update_field_index(obj_uid, field_name, adding=False)
            self._lru.pop(key, None)

    def erase_field_for_all(self, field_name: str) -> None:
        """Remove a field from all cached objects.

        Args:
            field_name: Name of the field to remove from all objects.
        """
        with self._lock:
            # Use field index for efficient removal if available
            if field_name in self._field_index:
                uids_to_remove = list(self._field_index[field_name])
                for uid in uids_to_remove:
                    key = self._make_key(uid, field_name)
                    item = self._store.pop(key, None)
                    if item:
                        data, _ = item
                        entry_size = self._calculate_entry_size(key, data)
                        self._current_bytes -= entry_size
                    self._lru.pop(key, None)
                # Clear the field index entry
                del self._field_index[field_name]
            else:
                # Fallback to iteration if index not available
                suffix = f":{field_name}"
                # Iterate over snapshot to avoid mutation during iteration
                for key in tuple(self._store.keys()):
                    if key.endswith(suffix) and key.startswith(self._key_prefix):
                        data, _ = self._store.pop(key, (b"", 0.0))
                        entry_size = self._calculate_entry_size(key, data)
                        self._current_bytes -= entry_size
                        self._lru.pop(key, None)
                        # Update field index
                        uid_field = self._extract_uid_field(key)
                        if uid_field:
                            uid, _ = uid_field
                            self._update_field_index(uid, field_name, adding=False)

    def clear(self) -> None:
        """Remove all cached data.

        Completely empties the cache and resets memory usage counters.
        """
        with self._lock:
            self._store.clear()
            self._lru.clear()
            self._field_index.clear()
            self._current_bytes = 0

    def get_available_bytes(self) -> int:
        """Return estimated available cache capacity in bytes."""
        with self._lock:
            return max(0, self._max_bytes - self._current_bytes)
