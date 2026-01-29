import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("zarr")

from diffract.core.storage.zarr_manager import ZarrStorageManager


@pytest.fixture
def zarr_store_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "zarr_store")


@pytest.fixture
def storage(zarr_store_dir: str) -> ZarrStorageManager:
    s = ZarrStorageManager(store_url=zarr_store_dir, root="root", readonly=False)
    s.connect()
    try:
        yield s
    finally:
        s.close()


def test_set_get_json_roundtrip(storage: ZarrStorageManager) -> None:
    uid = "u1"
    meta = {"a": 1, "b": [1, 2, 3], "s": "ok"}

    storage.set_field(uid, "__metadata__", meta)
    assert storage.has_field(uid, "__metadata__") is True
    assert storage.get_field(uid, "__metadata__") == meta


def test_set_get_pickle_fallback(storage: ZarrStorageManager) -> None:
    uid = "u_pickle"
    field = "v"
    value = {1, 2, 3}

    storage.set_field(uid, field, value)
    assert storage.get_field(uid, field) == value


def test_set_get_ndarray_roundtrip(storage: ZarrStorageManager) -> None:
    uid = "u2"
    field = "weights"
    arr = np.random.randn(64, 32).astype(np.float32)

    storage.set_field(uid, field, arr)
    got = storage.get_field(uid, field)

    assert isinstance(got, np.ndarray)
    assert got.shape == arr.shape
    assert got.dtype == arr.dtype
    np.testing.assert_allclose(got, arr)


def test_get_field_metadata(storage: ZarrStorageManager) -> None:
    uid = "meta1"
    field = "weights"
    arr = np.ones((4, 5), dtype=np.float64)

    storage.set_field(uid, field, arr)
    meta = storage.get_field_metadata(uid, field)

    assert meta is not None
    assert meta.get("kind") == "matrix"
    assert tuple(meta.get("shape") or []) == (4, 5)
    assert meta.get("dtype") == "float64"


def test_list_fields_and_objs(storage: ZarrStorageManager) -> None:
    storage.set_field("a", "f1", 1)
    storage.set_field("a", "f2", {"x": 1})
    storage.set_field("b", "f2", {"y": 2})

    assert set(storage.list_fields("a")) >= {"f1", "f2"}
    assert set(storage.list_fields()) >= {"f1", "f2"}
    assert set(storage.list_objs()) >= {"a", "b"}


def test_list_objs_has_field(storage: ZarrStorageManager) -> None:
    storage.set_field("x1", "common", 1)
    storage.set_field("x2", "common", 2)
    storage.set_field("x3", "other", 3)

    objs = set(storage.list_objs_has_field("common"))
    assert {"x1", "x2"}.issubset(objs)
    assert "x3" not in objs


def test_registry_integrity_after_erase(storage: ZarrStorageManager) -> None:
    storage.set_field("a", "f1", 1)
    storage.set_field("b", "f2", 2)

    storage.erase_field("a", "f1")
    assert "f1" not in storage.list_fields()
    assert "a" not in storage.list_objs()

    storage.erase_obj("b")
    assert "f2" not in storage.list_fields()
    assert "b" not in storage.list_objs()


def test_erase_field_removes_object_when_last_field(storage: ZarrStorageManager) -> None:
    uid = "z1"
    storage.set_field(uid, "t", 1)
    assert uid in storage.list_objs()

    storage.erase_field(uid, "t")
    assert storage.has_field(uid, "t") is False
    assert uid not in storage.list_objs()


def test_erase_obj_removes_all_fields(storage: ZarrStorageManager) -> None:
    uid = "z2"
    storage.set_field(uid, "a", 1)
    storage.set_field(uid, "b", {"x": 2})
    assert uid in storage.list_objs()

    storage.erase_obj(uid)
    assert storage.has_field(uid, "a") is False
    assert storage.has_field(uid, "b") is False
    assert uid not in storage.list_objs()


def test_erase_field_for_all(storage: ZarrStorageManager) -> None:
    storage.set_field("q1", "dead", 1)
    storage.set_field("q2", "dead", 2)
    storage.set_field("q3", "alive", 3)

    storage.erase_field_for_all("dead")

    assert storage.has_field("q1", "dead") is False
    assert storage.has_field("q2", "dead") is False
    assert storage.has_field("q3", "alive") is True


def test_clear_resets_storage(storage: ZarrStorageManager) -> None:
    storage.set_field("m1", "f", 1)
    storage.set_field("m2", "f", 2)
    assert set(storage.list_objs()) >= {"m1", "m2"}

    storage.clear()
    assert storage.list_objs() == []
    assert storage.list_fields() == []


def test_nested_batch_contexts(storage: ZarrStorageManager) -> None:
    uid = "u_nested"
    with storage:
        storage.set_field(uid, "a", 1)
        with storage:
            storage.set_field(uid, "b", 2)
            storage.set_field(uid, "c", 3)
        storage.set_field(uid, "d", 4)
    assert storage.get_field(uid, "a") == 1
    assert storage.get_field(uid, "b") == 2
    assert storage.get_field(uid, "c") == 3
    assert storage.get_field(uid, "d") == 4


def test_read_inside_batch_context_flushes_pending_ops(storage: ZarrStorageManager) -> None:
    uid = "u_read_flush"
    with storage:
        storage.set_field(uid, "a", 1)
        storage.set_field(uid, "b", 2)
        assert storage.has_field(uid, "a")
        assert storage.get_field(uid, "b") == 2


def test_readonly_allows_reads_rejects_writes(zarr_store_dir: str) -> None:
    s1 = ZarrStorageManager(store_url=zarr_store_dir, root="root", readonly=False)
    s1.connect()
    try:
        s1.set_field("r1", "k", {"a": 1})
    finally:
        s1.close()

    s2 = ZarrStorageManager(store_url=zarr_store_dir, root="root", readonly=True)
    s2.connect()
    try:
        assert s2.get_field("r1", "k") == {"a": 1}
        with pytest.raises(OSError):
            s2.set_field("r1", "k2", 123)
    finally:
        s2.close()
