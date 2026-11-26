"""Diffract: neural network weight analysis and evolution tracking.

Quick start::

    from diffract import Session

    # In-memory session (no persistence)
    session = Session(profile="ram")

    # Persistent local session (SQLite in .diffract/)
    session = Session(profile="local")

    # Or with a custom config file
    session = Session(config_path="my_config.ini")

Available profiles: "ram", "local", "hybrid".
Use ``diffract.list_profiles()`` to see all options.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

from .containers import list_profiles
from .core.data.metadata import upgrade_metadata_index
from .core.data.nn.params.schema import ParameterType
from .session import FieldSelector, ParameterOverrides, Session
from .session.analysis import AnalysisResult, analyze, compare

if TYPE_CHECKING:
    from . import viz as viz

__all__ = [
    "AnalysisResult",
    "FieldSelector",
    "ParameterOverrides",
    "ParameterType",
    "Session",
    "__version__",
    "analyze",
    "compare",
    "list_profiles",
    "upgrade_metadata_index",
    "viz",
]


def __getattr__(name: str) -> Any:
    if name == "viz":
        return importlib.import_module(".viz", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
