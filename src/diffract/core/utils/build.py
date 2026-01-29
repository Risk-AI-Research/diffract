"""Object construction utilities.

Provides helpers for building objects from configuration mappings with
default value handling and parameter merging capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["build_with_defaults"]


def build_with_defaults[T](
    cls: type[T], params: Mapping[str, Any] | None, **kwargs: Any
) -> T:
    """Construct an object filtering out None values from params.

    Merges non-None items from params with explicit keyword overrides and
    calls the class constructor with the resulting mapping.

    Args:
        cls: Class to instantiate.
        params: Mapping of parameter names to values; None values are ignored.
            If None, only kwargs are used.
        **kwargs: Explicit overrides to merge atop filtered params.

    Returns:
        Newly constructed instance of cls.

    Example:
        >>> class Config:
        ...     def __init__(self, a: int, b: str = "default"):
        ...         self.a = a
        ...         self.b = b
        >>> params = {"a": 1, "b": None, "c": "ignored"}
        >>> obj = build_with_defaults(Config, params, b="override")
    """
    if params is None:
        return cls(**kwargs)
    clean = {k: v for k, v in params.items() if v is not None}
    clean.update(**kwargs)
    return cls(**clean)
