"""Formatter registry and factory for result export.

Maintains a mapping from format names to formatter instances and provides
helpers to register custom formatters and retrieve formatters by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import diffract.core.utils.imports as import_utils

from .dict_formatter import DictFormatter
from .json_formatter import JsonFormatter

if TYPE_CHECKING:
    from diffract.core.export.interface import IResultFormatter

# Built-in formatter instances
FORMATTERS: dict[str, IResultFormatter] = {
    "dict": DictFormatter(),
    "json": JsonFormatter(),
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
    except KeyError as e:  # pragma: no cover - defensive
        known = ", ".join(sorted(FORMATTERS))
        msg = f"Unsupported format '{name}'. Known: {known}"
        raise ValueError(msg) from e
