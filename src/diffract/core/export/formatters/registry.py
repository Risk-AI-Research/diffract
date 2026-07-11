"""Formatter registry and factory for result export.

Maintains a mapping from format names to formatter instances and provides
helpers to register custom formatters and retrieve formatters by name.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

import diffract.core.utils.imports as import_utils

from .dict_formatter import DictFormatter
from .json_formatter import JsonFormatter
from .list_formatter import ListFormatter

if TYPE_CHECKING:
    from diffract.core.export.interface import IResultFormatter

# Formats backed by optional dependencies, mapped to the extra providing them
_OPTIONAL_FORMAT_EXTRAS: dict[str, str] = {
    "pandas": "pandas",
    "polars": "polars",
}

# Built-in formatter instances
FORMATTERS: dict[str, IResultFormatter] = {
    "dict": DictFormatter(),
    "json": JsonFormatter(),
    "list": ListFormatter(),
}

if import_utils.is_available("pandas"):
    from .pandas_formatter import PandasFormatter

    FORMATTERS["pandas"] = PandasFormatter()  # type: ignore[arg-type]

if import_utils.is_available("polars"):
    from .polars_formatter import PolarsFormatter

    FORMATTERS["polars"] = PolarsFormatter()  # type: ignore[arg-type]


def register_formatter(name: str, formatter: IResultFormatter) -> None:
    """Register a custom formatter under a given name.

    Overwrites an existing formatter if the name is already used.

    Args:
        name: Unique format name.
        formatter: Formatter instance to register.
    """
    FORMATTERS[name] = formatter


def get_formatter(name: str) -> IResultFormatter:
    """Return a formatter by name or raise ValueError if unknown.

    Args:
        name: Format name to lookup.

    Returns:
        Formatter instance associated with the name.

    Raises:
        ValueError: If name is not registered.
    """
    try:
        return FORMATTERS[name]
    except KeyError as e:
        if name in _OPTIONAL_FORMAT_EXTRAS:
            extra = _OPTIONAL_FORMAT_EXTRAS[name]
            msg = (
                f"Export format '{name}' requires the optional dependency "
                f"'{extra}'. Install it with: uv sync --extra {extra} "
                f"(or: pip install {extra})."
            )
        else:
            known = ", ".join(sorted(FORMATTERS))
            close = difflib.get_close_matches(name, FORMATTERS, n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            msg = f"Unsupported format '{name}'. Known: {known}.{hint}"
        raise ValueError(msg) from e
