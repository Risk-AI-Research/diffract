"""Unit tests for identifier / field-name validation (core/data/identity)."""

from __future__ import annotations

import pytest

from diffract.core.data.identity import (
    ALLOWED_IDENTIFIER_CHARACTERS,
    IdentifierValidationError,
    first_invalid_identifier_char,
    first_storage_unsafe_char,
    validate_field_name,
    validate_identifier,
)
from diffract.core.data.nn.aggregates.metadata import AggregateMetadata
from diffract.session.resolver import resolve

pytestmark = pytest.mark.unit

GRAMMAR_SEPARATORS = ["@", "[", "]", ",", "/"]


@pytest.mark.parametrize(
    "value",
    ["m1", "model-a", "bert-base", "layer.0.weight", "run_001", "a.b-c_d.0", "W"],
)
def test_valid_identifiers_pass(value: str) -> None:
    validate_identifier(value, kind="model id")
    assert first_invalid_identifier_char(value) is None


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_every_grammar_separator_is_rejected(sep: str) -> None:
    value = f"run{sep}x"
    with pytest.raises(IdentifierValidationError) as excinfo:
        validate_identifier(value, kind="model id")
    assert repr(sep) in str(excinfo.value)
    assert "model id" in str(excinfo.value)
    assert first_invalid_identifier_char(value) == sep


def test_separator_anywhere_is_caught() -> None:
    for value in ["@lead", "trail@", "mid@dle"]:
        with pytest.raises(IdentifierValidationError):
            validate_identifier(value, kind="parameter name")


def test_empty_identifier_is_rejected() -> None:
    with pytest.raises(IdentifierValidationError, match="non-empty"):
        validate_identifier("", kind="model id")


def test_non_string_identifier_is_rejected() -> None:
    with pytest.raises(IdentifierValidationError, match="must be a string"):
        validate_identifier(123, kind="model id")  # type: ignore[arg-type]


def test_allowed_alphabet_excludes_the_grammar_separators() -> None:
    for sep in GRAMMAR_SEPARATORS:
        assert sep not in ALLOWED_IDENTIFIER_CHARACTERS
    for ch in "_-.a0":
        assert ch in ALLOWED_IDENTIFIER_CHARACTERS


def test_rejected_identifier_would_have_broken_the_uid_grammar() -> None:
    clean = AggregateMetadata.create_uid_from_context(
        field_name="metric", context_models=("run_0",), context_params=()
    )
    selector = resolve(clean)
    assert (selector.field, selector.models) == ("metric", ("run_0",))

    exotic = AggregateMetadata.create_uid_from_context(
        field_name="metric", context_models=("run@0",), context_params=()
    )
    assert resolve(exotic).models != ("run@0",)


def test_field_name_allows_contextual_suffix() -> None:
    validate_field_name("stable_rank@models[m1,m2]@params[layer.0.weight]")


@pytest.mark.parametrize("bad", list('<>:"/\\|?*'))
def test_field_name_rejects_storage_hostile_chars(bad: str) -> None:
    value = f"metric{bad}x"
    with pytest.raises(IdentifierValidationError) as excinfo:
        validate_field_name(value)
    assert repr(bad) in str(excinfo.value)
    assert first_storage_unsafe_char(value) == bad


def test_field_name_slash_is_rejected() -> None:
    with pytest.raises(IdentifierValidationError, match="'/'"):
        validate_field_name("grad/norm")


def test_empty_field_name_is_rejected() -> None:
    with pytest.raises(IdentifierValidationError, match="non-empty"):
        validate_field_name("")
