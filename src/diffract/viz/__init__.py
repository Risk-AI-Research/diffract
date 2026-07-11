"""Refactored visualization stack for diffract.

Submodules that need plotting backends are imported lazily, so the package
and its data helpers stay importable without the ``viz`` extra installed.
"""

from typing import Any

__all__ = [
    "DARK_THEME",
    "DEFAULT_THEME",
    "MINIMAL_THEME",
    "Plot",
    "Theme",
    "apply_theme",
    "load_theme",
    "plots",
    "render",
    "render_from_config",
]

_RENDERER_EXPORTS = frozenset({"Plot", "load_theme", "render", "render_from_config"})
_STYLING_EXPORTS = frozenset(
    {"DARK_THEME", "DEFAULT_THEME", "MINIMAL_THEME", "Theme", "apply_theme"}
)


def __getattr__(name: str) -> Any:
    if name in _RENDERER_EXPORTS:
        from . import renderer

        return getattr(renderer, name)
    if name in _STYLING_EXPORTS:
        from . import styling

        return getattr(styling, name)
    if name == "plots":
        from . import plots

        return plots
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
