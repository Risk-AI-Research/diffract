"""Tests for HybridStorageManager."""
import os
import tempfile

import numpy as np
import pytest

from diffract.core.storage.hdf5_manager import HDF5StorageManager
from diffract.core.storage.hybrid_manager import HybridStorageManager
from diffract.core.storage.sqlite_manager import SQLiteStorageManager


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory


@pytest.fixture
def storage(tmpdir: str) -> HybridStorageManager:
    """Create a hybrid manager with a low threshold for testing."""
    light = SQLiteStorageManager(
        path=os.path.join(tmpdir, "store.db"),
        array_threshold=1024 * 1024,  # 1MB - will store small arrays inline
    )
    light.connect()

    # keep_file_open=False avoids HDF5 read/write handle conflicts without SWMR
    heavy = HDF5StorageManager(
        path=os.path.join(tmpdir, "store.h5"),
        swmr=False,
        verify_index=True,
        keep_file_open=False,
    )

    # Use 1KB threshold so we can easily test routing
    hybrid = HybridStorageManager(
        light_storage=light,
        heavy_storage=heavy,
        array_threshold=1024,
    )

    try:
        yield hybrid
    finally:
        hybrid.close()


def test_small_value_goes_to_sqlite(storage: HybridStorageManager) -> None:
    """Small dict should be stored in SQLite only."""
    uid, field = "u1", "meta"
    value = {"key": "value", "num": 42}

    storage.set_field(uid, field, value)

    assert storage.has_field(uid, field)
    assert storage.get_field(uid, field) == value

    # Verify it's actually in SQLite, not HDF5
    assert storage.light.has_field(uid, field)
    assert not storage.heavy.has_field(uid, field)


def test_large_array_goes_to_hdf5(storage: HybridStorageManager) -> None:
    """Large array should be stored in HDF5 with sentinel in SQLite."""
    uid, field = "u2", "weights"
    # Array larger than 1KB threshold
    arr = np.random.randn(100, 100).astype(np.float32)
    assert arr.nbytes > 1024

    storage.set_field(uid, field, arr)

    assert storage.has_field(uid, field)
    got = storage.get_field(uid, field)
    np.testing.assert_allclose(got, arr)

    # Verify routing: sentinel in SQLite, data in HDF5
    assert storage.light.has_field(uid, field)
    assert storage.light.get_field(uid, field) == "__HEAVY__"
    assert storage.heavy.has_field(uid, field)


def test_small_array_goes_to_sqlite(storage: HybridStorageManager) -> None:
    """Small array should stay in SQLite."""
    uid, field = "u3", "small_arr"
    arr = np.array([1, 2, 3], dtype=np.int32)
    assert arr.nbytes < 1024

    storage.set_field(uid, field, arr)

    assert storage.has_field(uid, field)
    np.testing.assert_array_equal(storage.get_field(uid, field), arr)

    # Should be in SQLite, not HDF5
    assert storage.light.has_field(uid, field)
    assert not storage.heavy.has_field(uid, field)


def test_get_field_metadata_from_both_backends(storage: HybridStorageManager) -> None:
    uid_sqlite, field_sqlite = "m1", "small_meta"
    arr_small = np.ones((2, 2), dtype=np.float32)
    storage.set_field(uid_sqlite, field_sqlite, arr_small)

    meta_sqlite = storage.get_field_metadata(uid_sqlite, field_sqlite)
    assert meta_sqlite is not None
    assert meta_sqlite.get("kind") == "matrix"

    uid_h5, field_h5 = "m2", "big_meta"
    arr_big = np.random.randn(100, 100).astype(np.float32)
    storage.set_field(uid_h5, field_h5, arr_big)

    meta_h5 = storage.get_field_metadata(uid_h5, field_h5)
    assert meta_h5 is not None
    assert meta_h5.get("kind") in ("matrix", "ndarray")

def test_overwrite_moves_data_between_backends(storage: HybridStorageManager) -> None:
    """Overwriting a field should move data and clean up old location."""
    uid, field = "u4", "data"

    # Start with small value in SQLite
    storage.set_field(uid, field, {"small": True})
    assert storage.light.has_field(uid, field)
    assert not storage.heavy.has_field(uid, field)

    # Overwrite with large array -> moves to HDF5
    large_arr = np.random.randn(100, 100).astype(np.float32)
    storage.set_field(uid, field, large_arr)
    assert storage.light.get_field(uid, field) == "__HEAVY__"
    assert storage.heavy.has_field(uid, field)

    # Overwrite with small value -> moves back to SQLite, HDF5 cleaned
    storage.set_field(uid, field, {"small": True})
    assert storage.light.has_field(uid, field)
    assert storage.light.get_field(uid, field) == {"small": True}
    assert not storage.heavy.has_field(uid, field)


def test_get_field_raises_when_missing(storage: HybridStorageManager) -> None:
    """get_field should raise KeyError if field doesn't exist anywhere."""
    with pytest.raises(KeyError):
        storage.get_field("nonexistent_uid", "nonexistent_field")


def test_erase_field_removes_from_both(storage: HybridStorageManager) -> None:
    """erase_field should remove data from both backends."""
    uid = "u7"

    # Large array (HDF5 + sentinel)
    large_arr = np.random.randn(100, 100).astype(np.float32)
    storage.set_field(uid, "large", large_arr)
    storage.erase_field(uid, "large")
    assert not storage.has_field(uid, "large")
    assert not storage.light.has_field(uid, "large")
    assert not storage.heavy.has_field(uid, "large")

    # Small value (SQLite only)
    storage.set_field(uid, "small", {"x": 1})
    storage.erase_field(uid, "small")
    assert not storage.has_field(uid, "small")


def test_erase_field_handles_hdf5_only(storage: HybridStorageManager) -> None:
    """erase_field should work for HDF5-only data (no SQLite entry)."""
    uid, field = "u8", "orphan"
    storage.heavy.set_field(uid, field, np.array([1]))

    storage.erase_field(uid, field)
    assert not storage.heavy.has_field(uid, field)


def test_erase_obj_cleans_both_backends(storage: HybridStorageManager) -> None:
    """erase_obj should remove all fields from both backends."""
    uid = "u9"

    storage.set_field(uid, "small", {"a": 1})
    storage.set_field(uid, "large", np.random.randn(100, 100).astype(np.float32))
    storage.heavy.set_field(uid, "hdf5_only", np.array([1, 2]))

    storage.erase_obj(uid)

    assert not storage.has_field(uid, "small")
    assert not storage.has_field(uid, "large")
    assert not storage.has_field(uid, "hdf5_only")
    assert uid not in storage.list_objs()


def test_erase_field_for_all_clears_both(storage: HybridStorageManager) -> None:
    """erase_field_for_all should remove field from both backends."""
    storage.set_field("a", "target", {"x": 1})
    storage.set_field("b", "target", np.random.randn(100, 100).astype(np.float32))
    storage.heavy.set_field("c", "target", np.array([1]))

    storage.erase_field_for_all("target")

    assert not storage.has_field("a", "target")
    assert not storage.has_field("b", "target")
    assert not storage.has_field("c", "target")


def test_clear_empties_both_backends(storage: HybridStorageManager) -> None:
    """clear should remove all data from both backends."""
    storage.set_field("x", "f1", 1)
    storage.set_field("y", "f2", np.random.randn(100, 100).astype(np.float32))

    storage.clear()

    assert storage.list_objs() == []
    assert storage.list_fields() == []


def test_context_manager_returns_self(storage: HybridStorageManager) -> None:
    """Context manager should return self for use in 'as' clause."""
    with storage as mgr:
        assert mgr is storage
        mgr.set_field("ctx_test", "f", 1)

    assert storage.get_field("ctx_test", "f") == 1


def test_batch_context_defers_writes(tmpdir: str) -> None:
    """Batch context should commit writes on exit."""
    light = SQLiteStorageManager(path=os.path.join(tmpdir, "batch.db"))
    light.connect()
    heavy = HDF5StorageManager(
        path=os.path.join(tmpdir, "batch.h5"),
        swmr=False,
        keep_file_open=False,
    )

    hybrid = HybridStorageManager(
        light_storage=light,
        heavy_storage=heavy,
        array_threshold=1024,
    )

    try:
        with hybrid:
            hybrid.set_field("u", "f", {"value": 1})

        # After exiting context, data should be committed
        assert hybrid.get_field("u", "f") == {"value": 1}
    finally:
        hybrid.close()

