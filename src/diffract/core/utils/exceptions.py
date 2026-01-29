"""Exception formatting helpers for concise, consistent logs."""

from __future__ import annotations


def format_exception_message(exc: BaseException) -> str:
    """Return a compact exception description for logs."""
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__
