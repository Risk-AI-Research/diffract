"""Persistent instance identity through the wired repository."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier
from uuid import UUID

import pytest

from diffract.core.data.metadata.errors import UniqueConstraintError
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.data.nn.instances import InstanceRepository

pytestmark = pytest.mark.unit


def test_wired_instances_keep_identity_across_rename(ram_container):
    repo = ram_container.nn.instance_repository()
    upper = repo.get_or_create("Model")
    lower = repo.get_or_create("model")
    assert UUID(upper).version == 4
    assert UUID(upper).hex == upper
    assert upper != lower
    assert repo.get_or_create("Model") == upper
    original = repo.get_by_name("Model")
    assert original == repo.get(upper)
    assert original.instance_uid == upper
    assert original.display_label is None
    assert original.json_data == {}
    assert original.created_at > 0
    with pytest.raises(UniqueConstraintError):
        repo.rename(upper, "model")
    repo.rename(upper, "z")
    assert repo.get_by_name("Model") is None
    assert repo.get_by_name("z").created_at == original.created_at
    assert [row.instance_uid for row in repo.list()] == [lower, upper]
    with pytest.raises(KeyError):
        repo.rename("absent", "lost")


def test_independent_connections_resolve_one_uid(tmp_path):
    path = str(tmp_path / "models.db")
    with closing(SQLiteMetadataIndex(path)) as index:
        index.connect()
    ready = Barrier(2)

    def resolve():
        with closing(SQLiteMetadataIndex(path)) as index:
            index.connect()
            ready.wait(timeout=5)
            return InstanceRepository.initialize(index).get_or_create("shared")

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = (
            future.result(timeout=8)
            for future in [pool.submit(resolve), pool.submit(resolve)]
        )
    assert left == right
    with closing(SQLiteMetadataIndex(path)) as index:
        assert [
            (row.instance_uid, row.name) for row in InstanceRepository(index).list()
        ] == [(left, "shared")]
