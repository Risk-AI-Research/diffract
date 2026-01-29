"""Model parameter extraction and management module.

This module provides comprehensive tools for extracting, organizing, and managing
neural network model parameters across different deep learning frameworks.
It supports efficient parameter storage, lazy loading, and flexible filtering
capabilities for large-scale model analysis.

Features:
    - Framework-agnostic parameter extraction
    - Hierarchical parameter organization with metadata
    - Lazy loading and intelligent caching for memory efficiency
    - Rich filtering by name, type, model ID, and fields
    - Batch operations for parameter prefetching and field management
    - Persistent storage with compression and deduplication

Core Components:
    - ParameterDataProxy: Individual parameter with lazy loading
    - ParameterCollection: Mutable collection with filtering capabilities
    - ParameterMetadata: Parameter metadata and type information
    - IParameterView: Protocol for parameter view implementations

Example:
    >>> # Prefer working via Session which owns the parameter repository.
    >>> # Views (IParameterView) are the primary interface for batch operations.
"""

from .containers import ModelParametersContainer
from .params import (
    IParameterView,
    ParameterDataProxy,
    ParameterMetadata,
    ParameterType,
    ParameterView,
)

__all__ = [
    "IParameterView",
    "ModelParametersContainer",
    "ParameterDataProxy",
    "ParameterMetadata",
    "ParameterType",
    "ParameterView",
]
