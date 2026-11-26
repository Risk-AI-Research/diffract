"""Keep metadata SQL inside the index and bulk SQL inside the SQLite value store."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import diffract

pytestmark = pytest.mark.unit

_SQL = re.compile(
    r"^\s*(?:SELECT\s+.+?\s+FROM\s+|INSERT\s+(?:OR\s+\w+\s+)?INTO\s+|"
    r"DELETE\s+FROM\s+|UPDATE\s+\S+\s+SET\s+|(?:CREATE|DROP|ALTER)\s+"
    r"(?:TEMP\s+|UNIQUE\s+)?(?:TABLE|INDEX|TRIGGER|VIEW)\s+|PRAGMA\s+\w+|"
    r"BEGIN\s+(?:IMMEDIATE|EXCLUSIVE)|WITH\s+\w+\s+AS\s*\()",
    re.IGNORECASE | re.DOTALL,
)


def _offenders(source: str) -> set[int]:
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
        and ast.get_docstring(node, clean=False) is not None
    }
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "sqlite3" for alias in node.names):
                lines.add(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "sqlite3"
        ):
            lines.add(node.lineno)
        elif isinstance(node, ast.JoinedStr):
            sample = "".join(
                part.value if isinstance(part, ast.Constant) else "identifier"
                for part in node.values
            )
            if _SQL.match(sample):
                lines.add(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and _SQL.match(node.value)
        ):
            lines.add(node.lineno)
    return lines


def test_sql_is_confined_to_its_storage_tier():
    root = Path(diffract.__file__).parent
    offenders = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.is_relative_to("core/data/metadata") or relative == Path(
            "core/storage/sqlite_manager.py"
        ):
            continue
        if lines := _offenders(path.read_text(encoding="utf-8")):
            offenders[str(relative)] = sorted(lines)
    assert not offenders


def test_gate_distinguishes_repository_sql_from_docstrings():
    source = '''"""Select records from several sources."""
from sqlite3 import connect
import sqlite3 as database
query = f"SELECT {column} FROM instances"
write = "INSERT INTO instances VALUES (?)"
'''
    assert _offenders(source) == {2, 3, 4, 5}
