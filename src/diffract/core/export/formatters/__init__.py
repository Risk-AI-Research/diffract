"""Formatters for exporting results to different output formats.

Includes JSON, dictionary, and optional pandas/polars DataFrame formatters along
with a small registry/factory for runtime selection by name.
"""

from .dict_formatter import DictFormatter
from .json_formatter import JsonFormatter
from .list_formatter import ListFormatter
from .pandas_formatter import PandasFormatter  # type: ignore[F401]
from .polars_formatter import PolarsFormatter  # type: ignore[F401]
from .registry import FORMATTERS, get_formatter, register_formatter

__all__ = [
    "FORMATTERS",
    "DictFormatter",
    "JsonFormatter",
    "ListFormatter",
    "PandasFormatter",
    "PolarsFormatter",
    "get_formatter",
    "register_formatter",
]
