"""Identifier and field-name validation for the data layer.

Identity strings (``model_id``, parameter ``name``, aggregate context members)
are embedded into the aggregate-uid grammar ``field@models[m1,m2]@params[p1]``
and into ``{obj_uid}/{field}`` storage keys, so the grammar separators
``@ [ ] ,`` and the path separator ``/`` are rejected at ingest. The alphabet
and predicate are also consumed by the store-migration tooling.
"""

from __future__ import annotations

import re
import string

__all__ = [
    "ALLOWED_IDENTIFIER_CHARACTERS",
    "FIELD_NAME_CHARSET_DESCRIPTION",
    "IDENTIFIER_CHARSET_DESCRIPTION",
    "STORAGE_UNSAFE_CHARACTERS",
    "STORAGE_UNSAFE_PATTERN",
    "IdentifierValidationError",
    "first_invalid_identifier_char",
    "first_storage_unsafe_char",
    "validate_field_name",
    "validate_identifier",
]


class IdentifierValidationError(ValueError):
    """Raised when an identifier or field name violates the accepted alphabet."""


# Characters permitted in an identifier: alphanumerics plus ``_``, ``-``, ``.``.
ALLOWED_IDENTIFIER_CHARACTERS = frozenset(string.ascii_letters + string.digits + "_-.")

IDENTIFIER_CHARSET_DESCRIPTION = (
    "Identifiers may contain only ASCII letters, digits, and the characters "
    "'_', '-' and '.'; this excludes the aggregate-context separators "
    "'@', '[', ']', ',' and the storage path separator '/'."
)

# Characters that break storage keying when used in a field name.
STORAGE_UNSAFE_CHARACTERS = frozenset('<>:"/\\|?*')

# Compiled form of :data:`STORAGE_UNSAFE_CHARACTERS` for metadata validation.
STORAGE_UNSAFE_PATTERN = re.compile(r'[<>:"/\\|?*]')

FIELD_NAME_CHARSET_DESCRIPTION = (
    "Field names must not contain the storage path separator '/' or any of the "
    'characters < > : " \\ | ? * (they become storage-key path segments).'
)


def first_invalid_identifier_char(value: str) -> str | None:
    """Return the first character outside the identifier alphabet, or ``None``."""
    return next((ch for ch in value if ch not in ALLOWED_IDENTIFIER_CHARACTERS), None)


def first_storage_unsafe_char(value: str) -> str | None:
    """Return the first storage-unsafe character of a field name, or ``None``."""
    return next((ch for ch in value if ch in STORAGE_UNSAFE_CHARACTERS), None)


def validate_identifier(value: str, *, kind: str) -> None:
    """Validate an identifier against the accepted alphabet.

    Args:
        value: The identifier to check.
        kind: Human-readable role interpolated into the error message.

    Raises:
        IdentifierValidationError: If ``value`` is not a non-empty string over
            the accepted alphabet; the message names the offending character.
    """
    if not isinstance(value, str):
        raise IdentifierValidationError(
            f"A {kind} must be a string, got {type(value).__name__}."
        )
    if not value:
        raise IdentifierValidationError(f"A {kind} must be a non-empty string.")

    bad = first_invalid_identifier_char(value)
    if bad is not None:
        raise IdentifierValidationError(
            f"Invalid {kind} {value!r}: the character {bad!r} is not allowed. "
            f"{IDENTIFIER_CHARSET_DESCRIPTION}"
        )


def validate_field_name(value: str, *, kind: str = "field name") -> None:
    """Validate a field name for storage-key safety.

    The aggregate-grammar separators stay legal here (an ingested name may carry
    a contextual suffix); only storage-hostile characters are rejected.

    Args:
        value: The field name to check.
        kind: Human-readable role interpolated into the error message.

    Raises:
        IdentifierValidationError: If ``value`` is not a non-empty string or
            contains a storage-unsafe character; the message names it.
    """
    if not isinstance(value, str):
        raise IdentifierValidationError(
            f"A {kind} must be a string, got {type(value).__name__}."
        )
    if not value:
        raise IdentifierValidationError(f"A {kind} must be a non-empty string.")

    bad = first_storage_unsafe_char(value)
    if bad is not None:
        raise IdentifierValidationError(
            f"Invalid {kind} {value!r}: the character {bad!r} is not allowed. "
            f"{FIELD_NAME_CHARSET_DESCRIPTION}"
        )
