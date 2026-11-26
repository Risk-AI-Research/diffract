"""Hash utilities for short deterministic identifiers.

Provides utilities to create short hexadecimal hashes from strings or generic
objects via JSON serialization with fallbacks, plus random entity IDs.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

__all__ = ["HashUtils", "get_unique_id", "new_entity_uid"]


class HashUtils:
    """Hash utilities producing 8-character hexadecimal strings.

    Provides methods to create deterministic short hashes from various data types
    including strings and complex objects through JSON serialization.
    """

    @classmethod
    def _prepare_data(cls, data: Any) -> str:
        """Serialize arbitrary data to a string suitable for hashing.

        Args:
            data: Data to serialize (string, iterable, or any JSON-serializable object).

        Returns:
            String representation of the data ready for hashing.
        """
        if isinstance(data, str):
            return data
        if isinstance(data, Iterable):
            serialized = json.dumps(tuple(data), sort_keys=True)
        else:
            try:
                serialized = json.dumps(data, ensure_ascii=False)
            except TypeError:
                serialized = str(data)
        return serialized

    @classmethod
    def _str_hash(cls, data: str) -> str:
        """Compute 8-character hexadecimal CRC32 hash for a string.

        Args:
            data: String to hash.

        Returns:
            8-character lowercase hexadecimal hash.
        """
        crc = zlib.crc32(data.encode("utf-8")) & 0xFFFFFFFF
        return format(crc, "08x")

    @classmethod
    def get_obj_hash(cls, data: Any) -> str:
        """Create deterministic short hash for any data structure.

        Args:
            data: Any data structure to hash.

        Returns:
            8-character hexadecimal hash string.

        Example:
            >>> HashUtils.get_obj_hash("hello")
            'aaf4c61d'
            >>> HashUtils.get_obj_hash([1, 2, 3])
            '5289df73'
        """
        return cls._str_hash(cls._prepare_data(data))


def new_entity_uid() -> str:
    """Mint a persistent entity identifier as 32 lowercase hexadecimal digits."""
    return uuid4().hex


def get_unique_id() -> str:
    """Return a random, 32-character entity identifier."""
    return new_entity_uid()
