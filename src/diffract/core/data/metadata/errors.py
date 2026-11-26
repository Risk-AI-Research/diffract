"""Constraint errors exposed by metadata row operations."""

from __future__ import annotations

import sqlite3


class MetadataIndexError(Exception):
    """A metadata index operation failed."""


class IndexContentionError(MetadataIndexError):
    """A metadata writer exhausted its contention budget."""


class SnapshotIsolationError(MetadataIndexError):
    """A write used a stale read transaction."""


class ConstraintViolationError(MetadataIndexError):
    """A write violates a constraint on a metadata table."""

    kind = "unknown"

    def __init__(
        self,
        message: str,
        *,
        statement_table: str,
        table: str | None,
        columns: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.statement_table = statement_table
        self.table = table
        self.columns = columns


class UniqueConstraintError(ConstraintViolationError):
    """A primary or unique key is already present."""

    kind = "unique"


class ForeignKeyViolationError(ConstraintViolationError):
    """A referenced row is missing or still in use."""

    kind = "foreign_key"


class NotNullViolationError(ConstraintViolationError):
    """A required column has no value."""

    kind = "not_null"


class CheckConstraintError(ConstraintViolationError):
    """A value falls outside a table's declared constraints."""

    kind = "check"


def constraint_error(
    error: sqlite3.IntegrityError, statement_table: str
) -> ConstraintViolationError:
    """Keep the SQLite diagnostic and identify columns when SQLite names them."""
    name = getattr(error, "sqlite_errorname", None)
    message = str(error)
    kind = {
        "SQLITE_CONSTRAINT_UNIQUE": UniqueConstraintError,
        "SQLITE_CONSTRAINT_PRIMARYKEY": UniqueConstraintError,
        "SQLITE_CONSTRAINT_FOREIGNKEY": ForeignKeyViolationError,
        "SQLITE_CONSTRAINT_NOTNULL": NotNullViolationError,
        "SQLITE_CONSTRAINT_CHECK": CheckConstraintError,
    }.get(name, ConstraintViolationError)
    table = None
    columns: tuple[str, ...] = ()
    if kind in {UniqueConstraintError, NotNullViolationError}:
        prefix = "NOT NULL" if kind is NotNullViolationError else "UNIQUE"
        label, _, detail = message.partition(": ")
        refs = [item.partition(".") for item in detail.split(", ")]
        if (
            label == f"{prefix} constraint failed"
            and all(
                dot and t.isidentifier() and col.isidentifier() for t, dot, col in refs
            )
            and len({t for t, _, _ in refs}) == 1
        ):
            table = refs[0][0]
            columns = tuple(col for _, _, col in refs)
    result = kind(
        message, statement_table=statement_table, table=table, columns=columns
    )
    if name == "SQLITE_CONSTRAINT_PRIMARYKEY":
        result.kind = "primary_key"
    return result
