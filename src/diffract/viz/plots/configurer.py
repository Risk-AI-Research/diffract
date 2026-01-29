"""Figure configurators (layout/theme wrappers) similar to notebooks_src."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from diffract.session import Session
from diffract.viz.renderer import Plot

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.viz.themes import Theme


@dataclass(slots=True)
class UpdateFigure:
    """Wrap another plot and apply Plotly update calls.

    This is the main escape hatch for "full Plotly customization" without
    forcing every Plot implementation to expose a huge kwargs surface.

    Can also apply a theme after other updates.

    Example:
        >>> wrapped = UpdateFigure(
        ...     plot=BoxPlot(field="frob_norm"),
        ...     layout={"title": "Custom Title", "showlegend": False},
        ...     traces={"marker_opacity": 0.5},
        ... )
        >>> fig = session.draw(plot=wrapped)
    """

    plot: Plot
    # Back-compat: older configs used `config:` meaning `Figure.update(config)`.
    config: dict[str, Any] | None = None
    update: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    traces: dict[str, Any] | None = None
    xaxes: dict[str, Any] | None = None
    yaxes: dict[str, Any] | None = None

    # Theming (applied after all other updates)
    theme: Theme | None = None

    def render(self, session: Session) -> go.Figure:
        """Render the wrapped plot and apply configuration overrides."""
        from diffract.viz.themes import apply_theme

        fig = self.plot.render(session)
        if self.config is not None:
            fig.update(self.config)
        if self.update is not None:
            fig.update(self.update)
        if self.layout is not None:
            fig.update_layout(**self.layout)
        if self.traces is not None:
            fig.update_traces(**self.traces)
        if self.xaxes is not None:
            fig.update_xaxes(**self.xaxes)
        if self.yaxes is not None:
            fig.update_yaxes(**self.yaxes)

        # Apply theme if provided (after other updates for override behavior)
        if self.theme is not None:
            fig = apply_theme(fig, self.theme)

        return fig
