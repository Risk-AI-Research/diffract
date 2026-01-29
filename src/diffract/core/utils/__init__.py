"""Utility helpers used across diffract core.

This package provides essential utilities for:
- Object construction with default value handling
- Hash generation for short deterministic identifiers
- Optional dependency management and lazy imports
- Math utilities

Main components:
- build: Object construction helper with defaults
- hashing: Short deterministic IDs and unique ID generation
- imports: Optional dependency and lazy import helpers
- math: Mathematical utilities (mean, etc.)
"""

from .build import build_with_defaults
from .hashing import HashUtils, get_unique_id
from .imports import (
    LazyImport,
    OptionalDependencyError,
    available,
    get_module,
    is_available,
    lazy_import,
    optional_import,
    require,
    requires_package,
)
from .math import mean

__all__ = [
    "HashUtils",
    "LazyImport",
    "OptionalDependencyError",
    "available",
    "build_with_defaults",
    "get_module",
    "get_unique_id",
    "is_available",
    "lazy_import",
    "mean",
    "optional_import",
    "require",
    "requires_package",
]
