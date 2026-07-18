"""Translate the data layer's identifier errors into the typed session error."""

from __future__ import annotations

from diffract.core.data.identity import (
    IdentifierValidationError,
    validate_field_name as _core_validate_field_name,
    validate_identifier as _core_validate_identifier,
)
from diffract.session.errors import InvalidIdentifierError

__all__ = [
    "check_field_name",
    "check_identifier",
]


def check_identifier(value: str, *, kind: str) -> None:
    """Validate an identifier, raising InvalidIdentifierError on failure."""
    try:
        _core_validate_identifier(value, kind=kind)
    except IdentifierValidationError as exc:
        raise InvalidIdentifierError(str(exc)) from exc


def check_field_name(value: str, *, kind: str = "field name") -> None:
    """Validate a field name, raising InvalidIdentifierError on failure."""
    try:
        _core_validate_field_name(value, kind=kind)
    except IdentifierValidationError as exc:
        raise InvalidIdentifierError(str(exc)) from exc
