"""Bounded retries across SQLite result codes and transaction boundaries."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from diffract.core.data.metadata.errors import (
    IndexContentionError,
    SnapshotIsolationError,
    UniqueConstraintError,
)
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("code", "expected", "attempts"),
    [
        (5, IndexContentionError, 3),
        (261, IndexContentionError, 3),
        (773, IndexContentionError, 3),
        (517, SnapshotIsolationError, 1),
        (sqlite3.SQLITE_LOCKED, sqlite3.OperationalError, 1),
    ],
)
def test_extended_code_controls_retry_budget(code, expected, attempts):
    index = SQLiteMetadataIndex(":memory:")
    calls = []
    failure = sqlite3.OperationalError("same message")
    failure.sqlite_errorcode = code

    def attempt():
        calls.append(None)
        raise failure

    with pytest.raises(expected) as error:
        index.with_write_retry(attempt, jitter_ms=(0, 0))
    assert error.value is failure or error.value.__cause__ is failure
    assert len(calls) == attempts


def test_arbiter_retry_restarts_complete_transaction(tmp_path):
    with closing(SQLiteMetadataIndex(str(tmp_path / "arbiter.db"))) as index:
        index.define_table("items", {})
        index.insert("items", "existing")
        attempts = []

        def attempt():
            with index:
                attempts.append(None)
                index.insert("items", "first")
                index.insert("items", "existing" if len(attempts) == 1 else "second")

        index.with_write_retry(
            attempt, retry_on=(UniqueConstraintError,), jitter_ms=(0, 0)
        )
        assert len(attempts) == 2
        assert sorted(index.list_uids("items")) == ["existing", "first", "second"]
        with index, pytest.raises(RuntimeError, match="transaction"):
            index.with_write_retry(attempt)
        assert len(attempts) == 2
