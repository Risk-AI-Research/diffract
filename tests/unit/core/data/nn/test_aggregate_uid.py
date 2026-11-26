"""Golden identity bytes for set and ordered aggregate membership."""

from __future__ import annotations

import pytest

from diffract.core.data.nn.aggregates.identity import aggregate_identity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("field", "members", "directed", "digest"),
    [
        (
            "f",
            (),
            False,
            "b9dbb008291882d546f739ec9b1aae9d688961ba34272028524a873aa594e897",
        ),
        (
            "k",
            ("q2", "q1", "q1"),
            False,
            "df6bf94922bf4b1cbe24e3f3a1015d1b8de693cb527e29d1a66c257431fb2a7c",
        ),
        (
            "k",
            ("q2", "q1"),
            True,
            "ad8d8486930e7f5835a5719c24217492de1d22631f26396f4dee992413ad6e03",
        ),
        (
            "k",
            ("q1", "q2"),
            True,
            "bb96bf2973d2a53eedca3fa01cef65c1aa7de0a65ec30c1371b4e6784f6cd9b1",
        ),
    ],
)
def test_aggregate_identity_bytes(field, members, directed, digest) -> None:
    assert aggregate_identity(field, members, directed=directed).uid == digest
