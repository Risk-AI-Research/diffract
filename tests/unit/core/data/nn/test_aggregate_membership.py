"""Pinned aggregate digests and their set and pair semantics."""

from itertools import permutations

import pytest

from diffract.core.data.nn.aggregates.identity import aggregate_identity

pytestmark = pytest.mark.unit

A = "01" * 16
B = "23" * 16
C = "fe" * 16
SINGLETON = "b1b05f8ad7d4ce7e0b11bd3842dde74b5dbbe9d7e805e16ba0953dd2a00ef655"
THREE = "f8615e0dc054f0e2af5f2d5b5887bec5e2d90ffdc707b2ded4b369f30b2ce929"
DIRECTED = "cac1b02e48d40e72db7dba7f5d1d565c98b0338abc50d8fa3c3275703cdd051c"


def test_canonical_aggregate_vectors() -> None:
    assert aggregate_identity("score", [A]).uid == SINGLETON
    for permutation in permutations([A, B, C]):
        identity = aggregate_identity("score", [*permutation, B])
        assert identity.uid == THREE
        assert identity.members == (("member", A), ("member", B), ("member", C))

    ordered = aggregate_identity("score", [B, A], directed=True)
    assert ordered.uid == DIRECTED
    assert ordered.members == (("source", B), ("target", A))
    assert aggregate_identity("score", [A, B], directed=True).uid != DIRECTED
    assert aggregate_identity("score", [B, A]).uid != DIRECTED
    assert aggregate_identity("different", [A]).uid != SINGLETON
