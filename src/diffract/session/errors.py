"""Session-level error types."""


class SessionError(Exception):
    """Base exception class for session-related errors."""


class ModelNotFoundError(SessionError):
    """Raised when a model is not found in the session."""


class ModelAlreadyExistsError(SessionError):
    """Raised when a model with the same id already exists."""


class KernelNotFoundError(SessionError):
    """Raised when a kernel is not found in the registry."""


class ScopeValidationError(SessionError):
    """Raised when the active scope is incompatible with a kernel's apply level.

    For example, a binary cross-model kernel requires exactly two models in
    scope; applying it with a different number of models raises this error
    before execution rather than silently producing nothing.
    """


class InvalidIdentifierError(SessionError):
    """Raised when a model id, parameter name, or field name is not accepted.

    Identity strings are validated at every ingest boundary against a documented
    alphabet; the message names the offending character.
    """


class IncompatibleStoreError(SessionError):
    """Raised when a session's store was written at a different schema version.

    Opening a persistent store whose metadata index predates the running
    library refuses fast rather than reading it with mismatched schema
    assumptions. The message names the explicit upgrade entry point and
    recommends a backup first.
    """
