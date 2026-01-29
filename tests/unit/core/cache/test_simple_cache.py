"""Unit tests for in-process LRU cache manager."""

from __future__ import annotations

import time

import numpy as np
import pytest

from diffract.core.cache.simple_manager import SimpleLRUCacheManager

pytestmark = pytest.mark.unit


@pytest.fixture(scope="function")
def cache() -> SimpleLRUCacheManager:
    c = SimpleLRUCacheManager(
        max_memory_mb=4, ttl_seconds=None, key_prefix="diffract:test:cache:"
    )
    c.clear()
    yield c
    c.clear()


def test_set_get_simple_value(cache: SimpleLRUCacheManager) -> None:
    uid = "obj-1"
    field = "x"
    value = {"a": 1, "b": [1, 2, 3]}

    cache.set_field(uid, field, value)
    got = cache.get_field(uid, field)

    assert got == value
    assert cache.has_field(uid, field) is True


def test_overwrite_value(cache: SimpleLRUCacheManager):
    uid = "obj-2"
    field = "y"

    cache.set_field(uid, field, 1)
    cache.set_field(uid, field, 2)

    assert cache.get_field(uid, field) == 2


def test_numpy_array_roundtrip(cache: SimpleLRUCacheManager):
    uid = "obj-3"
    field = "weights"

    arr = np.random.randn(128, 64).astype(np.float32)
    cache.set_field(uid, field, arr)

    got = cache.get_field(uid, field)
    assert isinstance(got, np.ndarray)
    assert got.shape == arr.shape
    assert got.dtype == arr.dtype
    np.testing.assert_allclose(got, arr)


def test_list_objs_has_field(cache: SimpleLRUCacheManager):
    field = "meta"
    uids = {"u1", "u2", "u3"}
    for u in uids:
        cache.set_field(u, field, {"ok": True})

    result = set(cache.list_objs_has_field(field))
    assert uids.issubset(result)


def test_erase_field(cache: SimpleLRUCacheManager):
    uid = "obj-4"
    field = "temp"
    cache.set_field(uid, field, 42)

    cache.erase_field(uid, field)
    assert cache.get_field(uid, field) is None
    assert cache.has_field(uid, field) is False


def test_erase_field_for_all(cache: SimpleLRUCacheManager):
    field = "shared"
    uids = ["s1", "s2", "s3"]
    for u in uids:
        cache.set_field(u, field, u)

    cache.erase_field_for_all(field)

    for u in uids:
        assert cache.get_field(u, field) is None


def test_clear(cache: SimpleLRUCacheManager):
    cache.set_field("a", "f", 1)
    cache.set_field("b", "f", 2)

    cache.clear()

    assert cache.get_field("a", "f") is None
    assert cache.get_field("b", "f") is None


def test_ttl_expiration():
    cache = SimpleLRUCacheManager(
        max_memory_mb=4, ttl_seconds=1, key_prefix="diffract:test:cache:ttl:"
    )
    cache.clear()

    cache.set_field("t1", "k", "v")
    assert cache.get_field("t1", "k") == "v"

    time.sleep(1.2)
    assert cache.get_field("t1", "k") is None

    cache.clear()


def test_lru_eviction(cache: SimpleLRUCacheManager):
    # Fill cache with multiple small entries to exceed 4MB limit
    # Each value ~64KB
    value = np.random.bytes(64 * 1024)
    for i in range(200):
        cache.set_field(f"obj{i}", "v", value)

    # After pressure, adding a new key keeps cache responsive
    cache.set_field("probe", "x", 1)
    assert cache.get_field("probe", "x") == 1


def test_get_available_bytes_empty_cache(cache: SimpleLRUCacheManager):
    """Empty cache should have full capacity available."""
    max_bytes = 4 * 1024 * 1024  # 4MB
    available = cache.get_available_bytes()
    assert available == max_bytes


def test_get_available_bytes_after_writes(cache: SimpleLRUCacheManager):
    """Available bytes should decrease after writes."""
    initial = cache.get_available_bytes()

    # Write some data
    cache.set_field("obj1", "data", np.zeros(1000))
    after_write = cache.get_available_bytes()

    assert after_write < initial
    assert after_write > 0
