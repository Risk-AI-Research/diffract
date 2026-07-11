"""Session-level error types."""


class SessionError(Exception):
    """Base exception class for session-related errors."""


class ModelNotFoundError(SessionError):
    """Raised when a model is not found in the session."""


class ModelAlreadyExistsError(SessionError):
    """Raised when a model with the same id already exists."""


class KernelNotFoundError(SessionError):
    """Raised when a kernel is not found in the registry."""
