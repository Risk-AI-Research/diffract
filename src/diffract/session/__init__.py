"""Public session API and namespace facade."""

from __future__ import annotations

from diffract.core.data.identity import DuplicateParameterError
from diffract.core.data.nn.extractors.base import ParameterOverrides
from diffract.core.data.nn.extractors.report import ExtractionItem, ExtractionReport
from diffract.core.data.nn.params.schema import ParameterType

from .errors import (
    EmptyExtractionError,
    FieldNotFoundError,
    IncompatibleStoreError,
    InvalidIdentifierError,
    KernelNotFoundError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ScopeValidationError,
    SessionError,
    UnrecognizedStoreError,
)
from .resolver import FieldSelector
from .session import Session
from .summaries import (
    ApplySummary,
    EraseSummary,
    PairCoverage,
    PairMismatch,
    ShapeMismatch,
)

__all__ = [
    "ApplySummary",
    "DuplicateParameterError",
    "EmptyExtractionError",
    "EraseSummary",
    "ExtractionItem",
    "ExtractionReport",
    "FieldNotFoundError",
    "FieldSelector",
    "IncompatibleStoreError",
    "InvalidIdentifierError",
    "KernelNotFoundError",
    "ModelAlreadyExistsError",
    "ModelNotFoundError",
    "PairCoverage",
    "PairMismatch",
    "ParameterOverrides",
    "ParameterType",
    "ScopeValidationError",
    "Session",
    "SessionError",
    "ShapeMismatch",
    "UnrecognizedStoreError",
]
