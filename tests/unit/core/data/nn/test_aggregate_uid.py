"""Unit tests for the aggregate-uid mint (AggregateMetadata.create_uid_from_context).

The mint is the single writer-side composer of the legacy uid grammar. Its
output is the storage identity of every stored aggregate, so the exact byte
shape is pinned here: a drifting mint silently orphans every previously
stored aggregate (dedup misses, "already computed" checks fail open).
"""

from __future__ import annotations

import pytest

from diffract.core.data.nn.aggregates.metadata import AggregateMetadata

pytestmark = pytest.mark.unit


def test_uid_is_the_bare_field_name_without_context() -> None:
    uid = AggregateMetadata.create_uid_from_context(
        field_name="l_overlap", context_models=(), context_params=()
    )
    assert uid == "l_overlap"


def test_uid_golden_shapes_match_v03_stored_identity() -> None:
    assert (
        AggregateMetadata.create_uid_from_context(
            field_name="f", context_models=("m1",), context_params=()
        )
        == "f@models[m1]"
    )
    assert (
        AggregateMetadata.create_uid_from_context(
            field_name="f", context_models=(), context_params=("p1", "p2")
        )
        == "f@params[p1,p2]"
    )
    assert (
        AggregateMetadata.create_uid_from_context(
            field_name="agreement",
            context_models=("m1", "m2"),
            context_params=("w1", "w2"),
        )
        == "agreement@models[m1,m2]@params[w1,w2]"
    )


def test_uid_sorts_context_members() -> None:
    forward = AggregateMetadata.create_uid_from_context(
        field_name="k", context_models=("m1", "m2"), context_params=("p1", "p2")
    )
    reversed_ = AggregateMetadata.create_uid_from_context(
        field_name="k", context_models=("m2", "m1"), context_params=("p2", "p1")
    )
    assert forward == reversed_ == "k@models[m1,m2]@params[p1,p2]"


def test_uid_discriminates_field_models_and_params() -> None:
    base = AggregateMetadata.create_uid_from_context(
        field_name="k", context_models=("m1",), context_params=("p1",)
    )
    assert base != AggregateMetadata.create_uid_from_context(
        field_name="k2", context_models=("m1",), context_params=("p1",)
    )
    assert base != AggregateMetadata.create_uid_from_context(
        field_name="k", context_models=("m2",), context_params=("p1",)
    )
    assert base != AggregateMetadata.create_uid_from_context(
        field_name="k", context_models=("m1",), context_params=("p2",)
    )
