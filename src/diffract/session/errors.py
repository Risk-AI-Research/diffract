"""Session-level error types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffract.core.data.nn.extractors.report import ExtractionReport


class SessionError(Exception):
    """Base exception class for session-related errors."""


class EmptyExtractionError(SessionError):
    """An input has no extractable weights. Inspect coverage for the reasons."""

    def __init__(self, model_id: str, coverage: ExtractionReport) -> None:
        self.coverage = coverage
        reasons = ", ".join(
            f"{item.name} ({item.reason})" for item in coverage.items[:5]
        )
        super().__init__(
            f"Model '{model_id}' produced no parameters. "
            f"{reasons or 'No parameter inputs.'} Supported inputs include PyTorch, "
            "TensorFlow, Flax and ONNX models with the matching extra installed, "
            "and dictionaries of 2D floating-point NumPy arrays."
        )


class ModelNotFoundError(SessionError):
    """Raised when a model is not found in the session."""


class ModelAlreadyExistsError(SessionError):
    """Raised when a model with the same id already exists."""


class KernelNotFoundError(SessionError):
    """Raised when a kernel is not found in the registry."""


class FieldNotFoundError(SessionError):
    """Raised when a requested result field is neither stored nor producible."""


class ScopeValidationError(SessionError):
    """Raised when the active scope is incompatible with a kernel's apply level.

    A binary cross-model kernel requires exactly two model/checkpoint
    coordinates. An incompatible scope raises this error before execution.
    """


class InvalidIdentifierError(SessionError):
    """Raised when a model id, parameter name, or field name is not accepted.

    Identity strings are validated at every ingest boundary against a documented
    alphabet; the message names the offending character.
    """


class IncompatibleStoreError(SessionError):
    """Raised when a metadata index has an incompatible or incomplete schema.

    The message gives the next step for that store.
    """


class UnrecognizedStoreError(SessionError):
    """Raised when a session's configured path is not a diffract store.

    A metadata path that names another application's database is refused
    before anything is written to it: the remedy is fixing the path, not
    upgrading the file, so this is distinct from
    :class:`IncompatibleStoreError`. The message names the identity check
    that failed.
    """
