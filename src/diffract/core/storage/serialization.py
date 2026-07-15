"""Shared, safe codec for stored field values.

Storage backends serialize bulk field values through this module. It never
deserializes arbitrary Python objects: every payload is either a NumPy ``.npy``
buffer, raw bytes, or a UTF-8 JSON document, and decoding dispatches on an
explicit tag. Because no branch reconstructs an opaque object, a crafted blob
cannot trigger code execution on load (CWE-502).

A value that is neither a NumPy array, bytes, nor JSON-serializable is rejected
rather than encoded, so unsupported kinds fail loudly at write time instead of
being smuggled through an object serializer.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import numpy as np

_TAG_NDARRAY = "ndarray"
_TAG_BYTES = "bytes"
_TAG_JSON = "json"

_SUPPORTED = "a numpy array, a JSON-serializable value, or bytes"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError


def _unsupported(value: Any) -> ValueError:
    return ValueError(
        f"cannot serialize value of type {type(value).__name__!r}; "
        f"expected {_SUPPORTED}"
    )


def encode_value(value: Any) -> tuple[bytes, str]:
    """Encode a stored value into a payload and its type tag.

    Args:
        value: Value to serialize. Supported kinds are NumPy arrays (of a
            non-object dtype), ``bytes`` or ``bytearray``, and any
            JSON-serializable value (NumPy scalar types are normalized to their
            Python equivalents).

    Returns:
        Tuple of ``(payload, tag)`` for later :func:`decode_value`.

    Raises:
        ValueError: If the value is not one of the supported kinds.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise _unsupported(value)
        bio = BytesIO()
        np.save(bio, value)
        return bio.getvalue(), _TAG_NDARRAY
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), _TAG_BYTES
    try:
        payload = json.dumps(value, default=_json_default).encode("utf-8")
    except TypeError as exc:
        raise _unsupported(value) from exc
    return payload, _TAG_JSON


def decode_value(payload: bytes, tag: str) -> Any:
    """Decode a payload previously produced by :func:`encode_value`.

    Args:
        payload: Serialized bytes.
        tag: Type tag returned by :func:`encode_value`.

    Returns:
        The reconstructed value.

    Raises:
        ValueError: If the tag is not recognized.
    """
    if tag == _TAG_NDARRAY:
        return np.load(BytesIO(payload))
    if tag == _TAG_BYTES:
        return payload
    if tag == _TAG_JSON:
        return json.loads(payload.decode("utf-8"))
    msg = f"unknown serialization tag: {tag!r}"
    raise ValueError(msg)
