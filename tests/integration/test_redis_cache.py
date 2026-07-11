"""Integration tests for Redis-backed cache manager."""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pytest

from diffract.core.cache.redis_manager import RedisLRUCacheManager

pytestmark = [pytest.mark.integration, pytest.mark.network]

REDIS_HOST = os.getenv("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("TEST_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))  # use isolated DB by default


@pytest.fixture(scope="function")
def cache() -> RedisLRUCacheManager:
    try:
        cache = RedisLRUCacheManager(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            max_memory_mb=16,
            ttl_seconds=None,
            key_prefix="diffract:test:cache:",
        )
    except Exception as e:  # noqa: BLE001 - test environment dependent
        pytest.skip(f"Redis not available: {e}")
    # The manager caps the whole Redis instance at max_memory_mb, and
    # clear() only removes this manager's own key prefix: leftovers from
    # other fixtures and previous runs in the shared test db count toward
    # the cap and starve the eviction test. Start from an empty db.
    cache._redis.flushdb()
    yield cache
    cache.clear()
    cache.close()


class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Foo) and other.x == self.x


def test_set_get_simple_value(cache: RedisLRUCacheManager) -> None:
    uid = "obj-1"
    field = "x"
    value = {"a": 1, "b": [1, 2, 3]}

    cache.set_field(uid, field, value)
    got = cache.get_field(uid, field)

    assert got == value
    assert cache.has_field(uid, field) is True


def test_overwrite_value(cache: RedisLRUCacheManager) -> None:
    uid = "obj-2"
    field = "y"

    cache.set_field(uid, field, 1)
    cache.set_field(uid, field, 2)

    assert cache.get_field(uid, field) == 2


def test_numpy_array_roundtrip(cache: RedisLRUCacheManager) -> None:
    uid = "obj-3"
    field = "weights"

    arr = np.random.randn(128, 64).astype(np.float32)
    cache.set_field(uid, field, arr)

    got = cache.get_field(uid, field)
    assert isinstance(got, np.ndarray)
    assert got.shape == arr.shape
    assert got.dtype == arr.dtype
    np.testing.assert_allclose(got, arr)


def test_list_objs_has_field(cache: RedisLRUCacheManager) -> None:
    field = "meta"
    uids = {"u1", "u2", "u3"}
    for u in uids:
        cache.set_field(u, field, {"ok": True})

    result = set(cache.list_objs_has_field(field))
    assert uids.issubset(result)


def test_erase_field(cache: RedisLRUCacheManager) -> None:
    uid = "obj-4"
    field = "temp"
    cache.set_field(uid, field, 42)

    cache.erase_field(uid, field)
    assert cache.get_field(uid, field) is None
    assert cache.has_field(uid, field) is False


def test_erase_field_for_all(cache: RedisLRUCacheManager) -> None:
    field = "shared"
    uids = ["s1", "s2", "s3"]
    for u in uids:
        cache.set_field(u, field, u)

    cache.erase_field_for_all(field)

    for u in uids:
        assert cache.get_field(u, field) is None


def test_clear(cache: RedisLRUCacheManager) -> None:
    cache.set_field("a", "f", 1)
    cache.set_field("b", "f", 2)

    cache.clear()

    assert cache.get_field("a", "f") is None
    assert cache.get_field("b", "f") is None


def test_ttl_expiration() -> None:
    pytest.importorskip("redis")
    cache = RedisLRUCacheManager(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        max_memory_mb=16,
        ttl_seconds=1,
        key_prefix="diffract:test:cache:ttl:",
    )
    cache.clear()

    cache.set_field("t1", "k", "v")
    assert cache.get_field("t1", "k") == "v"

    time.sleep(1.2)
    assert cache.get_field("t1", "k") is None

    cache.clear()
    cache.close()


def test_ram_only_settings(cache: RedisLRUCacheManager) -> None:
    # Validate that persistence is disabled where possible
    info_persistence = cache._redis.config_get(pattern="appendonly")
    # config_get returns dict like {"appendonly": "no"}
    assert info_persistence.get("appendonly", "no") == "no"

    info_save = cache._redis.config_get(pattern="save")
    # When disabled via CONFIG SET save "", it returns {"save": ""}
    assert info_save.get("save", "") == ""


def test_pickle_compatibility(cache: RedisLRUCacheManager) -> None:
    obj = Foo(7)
    cache.set_field("o", "foo", obj)
    got = cache.get_field("o", "foo")
    assert got == obj


@pytest.mark.slow
def test_large_value_eviction_behavior(cache: RedisLRUCacheManager) -> None:
    # This test does not assert eviction deterministically, but ensures no errors
    # when inserting values larger than the memory cap (Redis should evict old keys).
    big = bytearray(2 * 1024 * 1024)  # 2MB

    for i in range(20):
        cache.set_field(f"obj{i}", "big", big)

    # Operations still succeed and cache is responsive
    cache.set_field("probe", "x", 1)
    assert cache.get_field("probe", "x") == 1
