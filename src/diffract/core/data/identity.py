"""Identifier and field-name validation for the data layer.

Model and parameter names use a restricted alphabet so callers can address
them through structured selectors or the legacy input grammar.
"""

from __future__ import annotations

import re
import string
from operator import index

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


class DuplicateParameterError(ValueError):
    """A parameter already occupies a model, step and name coordinate."""

    def __init__(self, instance_uid: str, step: int, name: str) -> None:
        self.instance_uid = instance_uid
        self.step = step
        self.name = name
        super().__init__(
            f"Duplicate parameter at instance={instance_uid!r}, "
            f"step={step}, name={name!r}."
        )


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
    match = STORAGE_UNSAFE_PATTERN.search(value)
    return match[0] if match is not None else None


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


_WRAPPER_PREFIXES = (
    "module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
)


def training_step(value: object) -> int:
    """Return a nonnegative step representable by a SQLite integer."""
    if isinstance(value, bool):
        raise TypeError("A training step must be a nonnegative integer.")
    try:
        step = index(value)
    except TypeError as exc:
        raise TypeError(
            f"Invalid training step {value!r}; expected an integer."
        ) from exc
    if not 0 <= step <= 2**63 - 1:
        raise ValueError(f"Training step {step} is outside the SQLite integer range.")
    return step


def is_extra_state_key(name: object) -> bool:
    """Return whether a state dictionary key denotes auxiliary module state."""
    return isinstance(name, str) and name.rsplit(".", 1)[-1] == "_extra_state"


def validate_parameter_name(name: str) -> None:
    """Verify the identifier alphabet and the components of a tensor path."""
    validate_identifier(name, kind="parameter name")
    if "" in name.split("."):
        raise IdentifierValidationError(
            f"Parameter name {name!r} has an empty path component."
        )


def normalize_parameter_name(name: str) -> str:
    """Remove only leading wrapper components from a validated tensor path."""
    validate_parameter_name(name)
    while name.startswith(_WRAPPER_PREFIXES):
        name = name.split(".", 1)[1]
    return name


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
