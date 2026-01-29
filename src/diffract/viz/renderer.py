"""Hydra-configurable rendering on top of the public Session API.

This module intentionally avoids touching Session private fields. All data
access goes through `Session.compute()` + `Session.get_results(export_format="dict")`.

Supports theming via:
- `theme` parameter (a Theme instance)
- `theme_path` parameter (path to a YAML file with theme config)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from diffract.core.utils import imports as import_utils
from diffract.session import Session

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


class Plot(Protocol):
    """A configurable plot that can render itself given a Session."""

    def render(self, session: Session) -> go.Figure:
        """Render the plot using data from the session.

        Args:
            session: Session providing data access.

        Returns:
            A Plotly Figure object.
        """
        ...


def _require_plotly() -> Any:
    """Import and return plotly.graph_objects module."""
    return import_utils.require("plotly.graph_objects")


def render(
    plot: Plot,
    *,
    session: Session,
    theme: Theme | None = None,
) -> go.Figure:
    """Render a Plotly figure for an already-constructed plot object.

    Args:
        plot: A Plot instance to render.
        session: Session providing data access.
        theme: Optional theme to apply after rendering.

    Returns:
        A Plotly Figure object.
    """
    from diffract.viz.themes import apply_theme

    _require_plotly()
    fig = plot.render(session)

    if theme is not None:
        fig = apply_theme(fig, theme)

    return fig


def load_theme(theme_path: str | Path) -> Theme:
    """Load a Theme from a YAML file.

    The YAML file should contain theme fields at the top level:
        width: 1200
        height: 600
        font_family: "Times New Roman"
        ...
    """
    from diffract.viz.themes import theme_from_dict

    yaml = import_utils.require("yaml")

    path = Path(theme_path).expanduser().resolve()
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Theme YAML must be a mapping, got {type(data).__name__}")

    return theme_from_dict(data)


def render_from_config(
    *,
    session: Session,
    config_path: str | Path,
    overrides: list[str] | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Load a Hydra YAML plot config and render it.

    The config file is expected to contain a `plot` node with `_target_` pointing
    to a Plot implementation (e.g. `diffract.viz.plots.scalar.BoxPlot`).
    An optional `config` node may contain Plotly update dict compatible with
    `Figure.update(...)` to apply layout/trace tweaks.

    Args:
        session: Session providing data access.
        config_path: Path to YAML config file.
        overrides: Hydra overrides to apply.
        theme: Theme to apply after rendering.
        theme_path: Path to a YAML file containing theme config.
                    If both `theme` and `theme_path` are provided,
                    `theme` takes precedence.

    Returns:
        A Plotly Figure object.
    """
    from diffract.viz.themes import apply_theme

    go = _require_plotly()

    compose = import_utils.require("hydra").compose
    initialize_config_dir = import_utils.require("hydra").initialize_config_dir
    instantiate = import_utils.require("hydra.utils").instantiate
    omega_conf = import_utils.require("omegaconf").OmegaConf

    cfg_path = Path(config_path).expanduser().resolve()
    config_dir = str(cfg_path.parent)
    config_name = cfg_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    omega_conf.resolve(cfg)

    plot = instantiate(cfg.get("plot"), _convert_="object")
    fig = plot.render(session)

    fig_cfg = cfg.get("config")
    if fig_cfg is not None:
        # OmegaConf nodes (DictConfig/ListConfig) are not accepted by Plotly.
        fig_cfg_plain = omega_conf.to_container(fig_cfg, resolve=True)
        if not isinstance(fig_cfg_plain, dict):
            raise TypeError("Config node 'config' must be a dict-like mapping")
        fig.update(fig_cfg_plain)

    if not isinstance(fig, go.Figure):
        raise TypeError("Plot.render() must return plotly.graph_objects.Figure")

    # Apply theme
    resolved_theme = theme
    if resolved_theme is None and theme_path is not None:
        resolved_theme = load_theme(theme_path)

    if resolved_theme is not None:
        fig = apply_theme(fig, resolved_theme)

    return fig
