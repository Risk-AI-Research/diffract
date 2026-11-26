"""Physical addresses shared by value stores and migrations."""

from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import quote, unquote

from diffract.core.data.identity import validate_field_name

_VALUE_KEY_PARTS = 2
_LEGACY_SQLITE_PARTS = 4
_LEGACY_SQLITE_PREFIX = "sqlite-legacy"


def is_catalog_owner(name: str) -> bool:
    """Identify UUID and aggregate digest groups in a physical value store."""
    return re.fullmatch(r"(?:[0-9a-f]{32}){1,2}", name) is not None


def _encode_field_segment(field: str) -> str:
    encoded = quote(field, safe="")
    return encoded.replace(".", "%2E") if encoded in {".", ".."} else encoded


_cached_field_segment = lru_cache(maxsize=256)(_encode_field_segment)


def value_key(uid: str, field: str) -> str:
    """Return the two-component physical key for a stored value."""
    validate_field_name(field)
    if not isinstance(uid, str) or len(physical_parts(uid)) != 1:
        raise ValueError(f"Invalid value owner UID: {uid!r}")
    encoded = (
        _cached_field_segment(field)
        if type(field) is str
        else _encode_field_segment(field)
    )
    return f"{uid}/{encoded}"


def physical_parts(key: str) -> tuple[str, ...]:
    """Check an opaque relative path before passing it to a backend."""
    parts = tuple(key.split("/"))
    if any(part in {"", ".", ".."} or "\\" in part or "\0" in part for part in parts):
        raise ValueError(f"Invalid physical value key: {key!r}")
    return parts


def legacy_sqlite_key(table: str, field: str, uid: str) -> str:
    """Encode an existing row address without treating its components as paths."""
    parts = ("_" + quote(part, safe="") for part in (table, field, uid))
    return "/".join((_LEGACY_SQLITE_PREFIX, *parts))


def is_legacy_sqlite_key(key: str) -> bool:
    """Return whether a catalog address uses the legacy SQLite encoding."""
    return key.startswith(_LEGACY_SQLITE_PREFIX + "/")


def row_address(key: str) -> tuple[str, str, str]:
    """Map a catalog key or legacy table/field/UID address to row columns."""
    parts = physical_parts(key)
    if len(parts) == _LEGACY_SQLITE_PARTS and is_legacy_sqlite_key(key):
        _, table, field, uid = parts
        if not all(part.startswith("_") for part in (table, field, uid)):
            raise ValueError(f"Invalid legacy SQLite value key: {key!r}")
        return unquote(table[1:]), unquote(field[1:]), unquote(uid[1:])
    if len(parts) == _VALUE_KEY_PARTS:
        uid, field = parts
        return "", field, uid
    table, field, uid = parts
    return table, field, uid
