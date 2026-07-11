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

__version__ = "0.2.0"

from .containers import list_profiles
from .core.data.nn.params.schema import ParameterType
from .session import ParameterOverrides, Session

if TYPE_CHECKING:
    from . import viz as viz

__all__ = [
    "ParameterOverrides",
    "ParameterType",
    "Session",
    "__version__",
    "list_profiles",
    "viz",
]


def __getattr__(name: str) -> Any:
    if name == "viz":
        return importlib.import_module(".viz", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
