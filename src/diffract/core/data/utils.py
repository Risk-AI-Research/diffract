"""Utility functions for the data module."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from diffract.core.constants import REGEX_PREFIX


def build_matcher(patterns: Iterable[str]) -> Callable[[str], bool]:
    """Build a matcher function from exact strings and regex patterns.

    Args:
        patterns: Strings to match. Prefix with REGEX_PREFIX ("re:") for regex.

    Returns:
        Function that returns True if input matches any pattern.
    """
    exact: set[str] = set()
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if pattern.startswith(REGEX_PREFIX):
            compiled.append(re.compile(pattern.removeprefix(REGEX_PREFIX)))
        else:
            exact.add(pattern)

    def _matches(value: str) -> bool:
        if value in exact:
            return True
        return any(pattern.fullmatch(value) is not None for pattern in compiled)

    return _matches
