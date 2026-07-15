from pathlib import Path

import numpy as np
import pytest

from diffract.core.storage.hdf5_manager import HDF5StorageManager


@pytest.fixture
def h5tmp(temp_dir: Path) -> str:
    return str(temp_dir / "store.h5")


@pytest.fixture
def storage(h5tmp: str) -> HDF5StorageManager:
    # Keep SWMR off in tests to avoid platform/HDF5-build-specific behavior.
    s = HDF5StorageManager(
        path=h5tmp,
        root="root",
        compression="lzf",
        swmr=False,
        verify_index=True,
        keep_file_open=True,
        readonly=False,
    )
    try:
        yield s
    finally:
        s.close()


def test_set_get_dict_roundtrip(storage: HDF5StorageManager) -> None:
    uid = "u1"
    meta = {"a": 1, "b": [1, 2, 3], "s": "ok"}

    storage.set_field(uid, "__metadata__", meta)
    assert storage.has_field(uid, "__metadata__") is True
    assert storage.get_field(uid, "__metadata__") == meta


def test_set_get_scalar_roundtrip(storage: HDF5StorageManager) -> None:
    uid = "u_scalar"
    storage.set_field(uid, "x", 123)
    assert storage.get_field(uid, "x") == 123


def test_set_get_ndarray_roundtrip(storage: HDF5StorageManager) -> None:
    uid = "u2"
    field = "weights"
    arr = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)

    storage.set_field(uid, field, arr)
    got = storage.get_field(uid, field)

    assert isinstance(got, np.ndarray)
    assert got.shape == arr.shape
    assert got.dtype == arr.dtype
    np.testing.assert_allclose(got, arr)


def test_get_field_metadata(storage: HDF5StorageManager) -> None:
    uid = "meta1"
    field = "weights"
    arr = np.ones((4, 5), dtype=np.float64)

    storage.set_field(uid, field, arr)

    meta = storage.get_field_metadata(uid, field)
    assert meta is not None
    assert meta.get("kind") == "matrix"
    assert tuple(meta.get("shape") or []) == (4, 5)
    assert meta.get("dtype") == "float64"


def test_overwrite_field(storage: HDF5StorageManager) -> None:
    uid = "u3"
    field = "v"
    storage.set_field(uid, field, 1)
    storage.set_field(uid, field, 2)
    assert storage.get_field(uid, field) == 2


def test_list_fields_excludes_index_group(storage: HDF5StorageManager) -> None:
    storage.set_field("a", "f1", 1)
    storage.set_field("a", "f2", 2)

    all_fields = set(storage.list_fields())
    assert "__index__" not in all_fields
    assert {"f1", "f2"}.issubset(all_fields)

    fields_for_a = set(storage.list_fields("a"))
    assert {"f1", "f2"}.issubset(fields_for_a)


def test_list_objs_and_index_persistence(h5tmp: str) -> None:
    s1 = HDF5StorageManager(path=h5tmp, root="root", swmr=False, verify_index=True)
    try:
        s1.set_field("i1", "f", 1)
        s1.set_field("i2", "f", 2)
        assert set(s1.list_objs()) >= {"i1", "i2"}
    finally:
        s1.close()

    # Reopen with verify_index on; should still see objects.
    s2 = HDF5StorageManager(path=h5tmp, root="root", swmr=False, verify_index=True)
    try:
        assert set(s2.list_objs()) >= {"i1", "i2"}
    finally:
        s2.close()


def test_list_objs_has_field(storage: HDF5StorageManager) -> None:
    storage.set_field("x1", "common", 1)
    storage.set_field("x2", "common", 2)
    storage.set_field("x3", "other", 3)

    objs = set(storage.list_objs_has_field("common"))
    assert {"x1", "x2"}.issubset(objs)
    assert "x3" not in objs


def test_erase_field_removes_object_when_last_field(
    storage: HDF5StorageManager,
) -> None:
    uid = "z1"
    storage.set_field(uid, "t", 1)
    assert uid in storage.list_objs()

    storage.erase_field(uid, "t")
    assert storage.has_field(uid, "t") is False
    assert uid not in storage.list_objs()


def test_erase_obj_removes_all_fields_and_index(storage: HDF5StorageManager) -> None:
    uid = "z2"
    storage.set_field(uid, "a", 1)
    storage.set_field(uid, "b", 2)
    assert uid in storage.list_objs()

    storage.erase_obj(uid)
    assert storage.has_field(uid, "a") is False
    assert storage.has_field(uid, "b") is False
    assert uid not in storage.list_objs()


def test_erase_field_for_all_updates_index_when_verify_enabled(
    storage: HDF5StorageManager,
) -> None:
    # q1/q2 only have 'dead'; q3 has 'alive' only.
    storage.set_field("q1", "dead", 1)
    storage.set_field("q2", "dead", 2)
    storage.set_field("q3", "alive", 3)
    assert set(storage.list_objs()) >= {"q1", "q2", "q3"}

    storage.erase_field_for_all("dead")

    assert storage.has_field("q1", "dead") is False
    assert storage.has_field("q2", "dead") is False
    assert storage.has_field("q3", "alive") is True
    # verify_index=True in fixture => index should not include q1/q2 anymore.
    assert "q1" not in storage.list_objs()
    assert "q2" not in storage.list_objs()
    assert "q3" in storage.list_objs()


def test_clear_resets_storage(storage: HDF5StorageManager) -> None:
    storage.set_field("m1", "f", 1)
    storage.set_field("m2", "f", 2)
    assert set(storage.list_objs()) >= {"m1", "m2"}

    storage.clear()
    assert storage.list_objs() == []
    assert storage.list_fields() == []


def test_readonly_allows_reads_but_rejects_writes(h5tmp: str) -> None:
    s1 = HDF5StorageManager(path=h5tmp, root="root", swmr=False, readonly=False)
    try:
        s1.set_field("r1", "k", {"a": 1})
    finally:
        s1.close()

    s2 = HDF5StorageManager(path=h5tmp, root="root", swmr=False, readonly=True)
    try:
        assert s2.get_field("r1", "k") == {"a": 1}
        with pytest.raises(OSError):
            s2.set_field("r1", "k2", 123)
    finally:
        s2.close()


def test_nested_batch_contexts(storage: HDF5StorageManager) -> None:
    """Test that nested context managers work correctly."""
    uid = "u_nested"
    with storage:
        storage.set_field(uid, "a", 1)
        with storage:  # Nested context
            storage.set_field(uid, "b", 2)
            storage.set_field(uid, "c", 3)
        # Inner context exited, but outer still active
        storage.set_field(uid, "d", 4)
    # All fields should be committed after outermost context exits
    assert storage.get_field(uid, "a") == 1
    assert storage.get_field(uid, "b") == 2
    assert storage.get_field(uid, "c") == 3
    assert storage.get_field(uid, "d") == 4


def test_nested_batch_context_operations_queued(storage: HDF5StorageManager) -> None:
    """Test that operations in nested contexts are queued and flushed when needed."""
    uid = "u_nested_queue"
    with storage:
        storage.set_field(uid, "a", 1)
        with storage:  # Nested context
            storage.set_field(uid, "b", 2)
            # Note: In HDF5, calling has_field/get_field inside a batch context
            # triggers _perform_batch_operations(), so operations become visible
            # immediately.
            assert storage.has_field(uid, "a")  # This flushes operations
            assert storage.has_field(uid, "b")
        storage.set_field(uid, "c", 3)
    # All fields should be committed after outermost context exits
    assert storage.get_field(uid, "a") == 1
    assert storage.get_field(uid, "b") == 2
    assert storage.get_field(uid, "c") == 3
