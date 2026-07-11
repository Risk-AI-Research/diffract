from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from diffract.viz.renderer import Plot
from diffract.viz.styling import apply_theme

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.session import Session
    from diffract.viz.styling import Theme


@dataclass(kw_only=True)
class UpdateFigure:
    """Wrap another plot and apply Plotly update calls."""

    plot: Plot

    # Back-compat: older configs used `config:` meaning `Figure.update(config)`.
    config: dict[str, Any] | None = None
    update: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    traces: dict[str, Any] | None = None
    xaxes: dict[str, Any] | None = None
    yaxes: dict[str, Any] | None = None

    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render the wrapped plot and apply configuration overrides."""
        fig = self.plot.render(session, theme)

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

        if theme is not None:
            apply_theme(fig, theme)

        return fig
