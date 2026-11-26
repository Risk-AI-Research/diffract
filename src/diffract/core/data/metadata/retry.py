"""Bounded retry for SQLite metadata writers."""

from __future__ import annotations

import random
import sqlite3
import time
from collections.abc import Callable

from .errors import IndexContentionError, SnapshotIsolationError

RETRY_ATTEMPTS = 3
RETRY_JITTER_MS = (10, 50)


def is_retryable(exc: sqlite3.Error) -> bool:
    """Return whether the SQLite result code is a transient writer conflict."""
    code = getattr(exc, "sqlite_errorcode", 0)
    return code & 0xFF == sqlite3.SQLITE_BUSY and code != sqlite3.SQLITE_BUSY_SNAPSHOT


def with_write_retry[T](
    attempt: Callable[[], T],
    *,
    in_transaction: Callable[[], bool],
    attempts: int = RETRY_ATTEMPTS,
    jitter_ms: tuple[int, int] = RETRY_JITTER_MS,
    retry_on: tuple[type[Exception], ...] = (),
) -> T:
    """Retry a complete operation from a clean transaction boundary."""
    if attempts < 1 or not 0 <= jitter_ms[0] <= jitter_ms[1]:
        raise ValueError("Expected a positive attempt count and ordered jitter bounds")
    for number in range(1, attempts + 1):
        if in_transaction():
            raise RuntimeError("Write retry requires a closed transaction")
        try:
            return attempt()
        except (sqlite3.Error, *retry_on) as exc:
            if isinstance(exc, sqlite3.Error):
                if (
                    getattr(exc, "sqlite_errorcode", None)
                    == sqlite3.SQLITE_BUSY_SNAPSHOT
                ):
                    raise SnapshotIsolationError(
                        "Stale read snapshot: use BEGIN IMMEDIATE for writes"
                    ) from exc
                if not is_retryable(exc):
                    raise
            if number == attempts:
                raise IndexContentionError(
                    f"Metadata write contention after {number} attempts"
                ) from exc
            time.sleep(random.uniform(*jitter_ms) / 1000)  # nosec B311
    raise AssertionError("unreachable")
