"""Tests for RAMStorageManager."""
from __future__ import annotations

import pytest

from diffract.core.storage.ram_manager import RAMStorageManager


@pytest.fixture
def storage() -> RAMStorageManager:
    return RAMStorageManager()


def test_set_get_roundtrip(storage: RAMStorageManager) -> None:
    storage.set_field("u1", "f", {"a": 1})
    assert storage.has_field("u1", "f")
    assert storage.get_field("u1", "f") == {"a": 1}


def test_overwrite(storage: RAMStorageManager) -> None:
    storage.set_field("u2", "f", 1)
    storage.set_field("u2", "f", 2)
    assert storage.get_field("u2", "f") == 2


def test_list_fields_and_objs(storage: RAMStorageManager) -> None:
    storage.set_field("u3", "a", 1)
    storage.set_field("u4", "b", 2)
    storage.set_field("u3", "c", 3)

    assert set(storage.list_fields()) >= {"a", "b", "c"}
    assert set(storage.list_fields("u3")) == {"a", "c"}
    assert set(storage.list_objs()) >= {"u3", "u4"}


def test_list_objs_has_field(storage: RAMStorageManager) -> None:
    storage.set_field("u5", "k", 1)
    storage.set_field("u6", "k", 2)
    storage.set_field("u7", "q", 3)

    assert set(storage.list_objs_has_field("k")) == {"u5", "u6"}
    assert "u7" not in storage.list_objs_has_field("k")


def test_erase_field(storage: RAMStorageManager) -> None:
    storage.set_field("u8", "f", 1)
    storage.erase_field("u8", "f")
    assert not storage.has_field("u8", "f")


def test_erase_obj(storage: RAMStorageManager) -> None:
    storage.set_field("u9", "a", 1)
    storage.set_field("u9", "b", 2)

    storage.erase_obj("u9")

    assert not storage.has_field("u9", "a")
    assert not storage.has_field("u9", "b")
    assert "u9" not in storage.list_objs()


def test_erase_field_for_all(storage: RAMStorageManager) -> None:
    storage.set_field("u10", "dead", 1)
    storage.set_field("u11", "dead", 2)
    storage.set_field("u12", "alive", 3)

    storage.erase_field_for_all("dead")

    assert not storage.has_field("u10", "dead")
    assert not storage.has_field("u11", "dead")
    assert storage.has_field("u12", "alive")


def test_clear(storage: RAMStorageManager) -> None:
    storage.set_field("u13", "f1", 1)
    storage.set_field("u14", "f2", 2)

    storage.clear()

    assert storage.list_objs() == []
    assert storage.list_fields() == []


def test_context_manager_returns_self() -> None:
    with RAMStorageManager() as mgr:
        assert isinstance(mgr, RAMStorageManager)
        mgr.set_field("ctx", "f", 1)
        assert mgr.get_field("ctx", "f") == 1

