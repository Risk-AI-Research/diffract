"""Round-trip and rejection tests for the storage value codec."""

from __future__ import annotations

import numpy as np
import pytest

from diffract.core.storage.serialization import decode_value, encode_value

pytestmark = pytest.mark.unit


def test_ndarray_roundtrip_preserves_dtype_shape_values() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload, tag = encode_value(arr)
    assert tag == "ndarray"
    out = decode_value(payload, tag)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    assert np.array_equal(out, arr)


def test_empty_ndarray_roundtrip() -> None:
    arr = np.empty((0, 4), dtype=np.int64)
    payload, tag = encode_value(arr)
    out = decode_value(payload, tag)
    assert out.dtype == arr.dtype
    assert out.shape == (0, 4)
    assert np.array_equal(out, arr)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (7, 7),
        (3.5, 3.5),
        (np.float64(2.25), 2.25),
        (np.int64(-11), -11),
        (True, True),
    ],
)
def test_scalar_roundtrip_normalizes_to_python(value: object, expected: object) -> None:
    payload, tag = encode_value(value)
    assert tag == "json"
    out = decode_value(payload, tag)
    assert out == expected
    assert type(out) is type(expected)


def test_dict_with_nested_list_roundtrip() -> None:
    value = {"a": 1, "b": [1, 2, 3], "c": {"d": None}}
    payload, tag = encode_value(value)
    assert tag == "json"
    assert decode_value(payload, tag) == value


def test_list_roundtrip() -> None:
    value = [1, "two", 3.0, None, [4]]
    payload, tag = encode_value(value)
    assert decode_value(payload, tag) == value


def test_str_and_none_roundtrip() -> None:
    for value in ("hello", None):
        payload, tag = encode_value(value)
        assert decode_value(payload, tag) == value


def test_bytes_roundtrip() -> None:
    value = b"\x00\x01\x02\xff"
    payload, tag = encode_value(value)
    assert tag == "bytes"
    out = decode_value(payload, tag)
    assert out == value
    assert isinstance(out, bytes)


def test_bytearray_roundtrip_equals_bytes() -> None:
    value = bytearray(b"payload")
    payload, tag = encode_value(value)
    assert tag == "bytes"
    out = decode_value(payload, tag)
    assert out == value
    assert isinstance(out, bytes)


def test_numpy_scalar_inside_container_roundtrips() -> None:
    value = {"weights": [np.float64(1.0), np.int32(2)]}
    payload, tag = encode_value(value)
    assert tag == "json"
    assert decode_value(payload, tag) == {"weights": [1.0, 2]}


def test_custom_object_raises_actionable_error() -> None:
    class Widget:
        pass

    with pytest.raises(ValueError, match=r"numpy array.*JSON-serializable.*bytes"):
        encode_value(Widget())


def test_container_holding_ndarray_raises() -> None:
    with pytest.raises(ValueError, match="expected a numpy array"):
        encode_value({"payload": np.arange(3)})


def test_object_dtype_ndarray_raises() -> None:
    arr = np.array([object(), object()], dtype=object)
    with pytest.raises(ValueError, match=r"numpy array.*JSON-serializable.*bytes"):
        encode_value(arr)


def test_decode_unknown_tag_raises() -> None:
    with pytest.raises(ValueError, match="unknown serialization tag: 'exec'"):
        decode_value(b"anything", "exec")
