"""Visualization namespace for Session."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from diffract.session.session import Session, SessionContext
from diffract.viz.renderer import render, render_from_config

from .box import box
from .grid import bound_grid, grid
from .heatmap import heatmap
from .scatter import scatter
from .sparkline import sparkline
from .violin import violin

if TYPE_CHECKING:  # pragma: no cover
    from diffract.viz.styling import Theme


class VizNamespace:
    """Visualization API for Session.

    Provides multiple ways to create visualizations:

    1. Simple methods (recommended for quick exploration):
        >>> session.viz.box(y="stable_rank", x="model_id")
        >>> session.viz.scatter(x="frob_norm", y="stable_rank")

    2. Plot objects (for Hydra configs and advanced customization):
        >>> from diffract.viz.plots.boxplot import BoxPlot
        >>> session.viz.draw(plot=BoxPlot(y="stable_rank", x="model_id"))

    3. Config files (for reproducible workflows):
        >>> session.viz.draw(config_path="plots/boxplot.yaml")

    4. Grid helpers (for subplot dashboards and bound grids):
        >>> session.viz.grid(subplots=[...])
        >>> session.viz.bound_grid(plot_template=..., row=..., col=...)
    """

    def __init__(
        self,
        session_or_context: Session | SessionContext,
    ) -> None:
        self.__session_or_context = session_or_context

    def draw(
        self,
        *,
        plot: Any = None,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
        theme: Theme | None = None,
        theme_path: str | Path | None = None,
    ) -> Any:
        """Render a Plotly figure using the viz module.

        Provide either `plot` (a Plot object) or `config_path` (a Hydra YAML file).

        Args:
            plot: A Plot instance to render.
            config_path: Path to a Hydra YAML config file.
            overrides: Hydra overrides to apply (only with config_path).
            theme: A Theme instance to apply.
            theme_path: Path to a YAML file with theme config.
        """
        with self.__session_or_context:
            if (plot is None) == (config_path is None):
                raise ValueError("Provide exactly one of: plot=... or config_path=...")

            if config_path is not None:
                return render_from_config(
                    session=self.__session_or_context,
                    config_path=config_path,
                    overrides=overrides,
                    theme=theme,
                    theme_path=theme_path,
                )

            return render(plot, session=self.__session_or_context, theme=theme)

    box = box
    grid = grid
    bound_grid = bound_grid
    violin = violin
    scatter = scatter
    heatmap = heatmap
    sparkline = sparkline
    line = sparkline
