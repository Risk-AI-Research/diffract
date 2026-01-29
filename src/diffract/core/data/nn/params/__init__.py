"""Parameter management module.

This package contains the core parameter management components:
- schema: Domain type definitions and enumerations
- interface: Protocol definitions for repositories, views, and proxies
- proxy: Parameter proxy implementation
- view: Parameter view implementation
"""

from .interface import IParameterProxy, IParameterRepository, IParameterView
from .metadata import ParameterMetadata
from .proxy import ParameterDataProxy
from .schema import FieldName, ParameterIndex, ParameterType, ParameterUID
from .view import ParameterView

__all__ = [
    "FieldName",
    "IParameterProxy",
    "IParameterRepository",
    "IParameterView",
    "ParameterDataProxy",
    "ParameterIndex",
    "ParameterMetadata",
    "ParameterType",
    "ParameterUID",
    "ParameterView",
]
