"""Tests for the SQLite metadata index."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffract.core.data.metadata.sqlite_index import (
    IN_MEMORY_DATABASE,
    SQLiteMetadataIndex,
)

pytestmark = pytest.mark.unit


def test_in_memory_sentinel_creates_no_filesystem_entry(temp_dir: Path) -> None:
    """The sentinel selects an in-memory database, so the index must not treat it
    as a file: no directory or file named ':memory:' appears on disk."""
    index = SQLiteMetadataIndex(IN_MEMORY_DATABASE)
    try:
        index.connect()
    finally:
        index.close()

    assert not (Path.cwd() / IN_MEMORY_DATABASE).exists()
    assert not (temp_dir / IN_MEMORY_DATABASE).exists()


def test_file_backed_index_creates_its_parent_directory(temp_dir: Path) -> None:
    """An ordinary path is still a path: the index creates the parent directory
    and the database file."""
    db_path = temp_dir / "nested" / "index.db"

    index = SQLiteMetadataIndex(str(db_path))
    try:
        index.connect()
    finally:
        index.close()

    assert db_path.parent.is_dir()
    assert db_path.exists()
