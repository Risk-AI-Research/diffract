"""Public session API and namespace facade."""

from __future__ import annotations

from diffract.core.data.nn.extractors.base import ParameterOverrides
from diffract.core.data.nn.params.schema import ParameterType

from .errors import (
    IncompatibleStoreError,
    InvalidIdentifierError,
    KernelNotFoundError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ScopeValidationError,
    SessionError,
)
from .session import Session
from .summaries import ApplySummary, EraseSummary

__all__ = [
    "ApplySummary",
    "EraseSummary",
    "IncompatibleStoreError",
    "InvalidIdentifierError",
    "KernelNotFoundError",
    "ModelAlreadyExistsError",
    "ModelNotFoundError",
    "ParameterOverrides",
    "ParameterType",
    "ScopeValidationError",
    "Session",
    "SessionError",
]
