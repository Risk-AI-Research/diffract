from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from .theme import Theme


def apply_theme(fig: go.Figure, theme: Theme) -> go.Figure:
    """Apply figure-level styling from theme.

    This applies layout, typography, background, legend, and colorbar styling.
    Does NOT override trace-specific settings or axes configured by configurators.

    Args:
        fig: Plotly Figure to style.
        theme: Theme to apply.

    Returns:
        The same figure (mutated in place) for chaining.
    """
    _apply_layout(fig, theme)
    _apply_background(fig, theme)
    _apply_typography(fig, theme)
    _apply_legend(fig, theme)
    _apply_colorbar_defaults(fig, theme)

    return fig


def _apply_layout(fig: go.Figure, theme: Theme) -> None:
    """Apply layout dimensions and margins."""
    if theme.layout.width is not None:
        fig.update_layout(width=theme.layout.width)

    if theme.layout.height is not None:
        fig.update_layout(height=theme.layout.height)

    fig.update_layout(margin=theme.layout.margin)


def _apply_background(fig: go.Figure, theme: Theme) -> None:
    """Apply background colors."""
    fig.update_layout(
        plot_bgcolor=theme.background.plot_bgcolor,
        paper_bgcolor=theme.background.paper_bgcolor,
    )


def _apply_typography(fig: go.Figure, theme: Theme) -> None:
    """Apply font settings."""
    fig.update_layout(
        font=dict(
            family=theme.typography.font_family,
            size=theme.typography.label_font_size,
        ),
        title=dict(
            font=dict(
                family=theme.typography.font_family,
                size=theme.typography.title_font_size,
            )
        ),
    )

    # Style annotations with consistent font
    fig.update_annotations(
        font=dict(
            family=theme.typography.font_family,
            size=theme.typography.label_font_size,
        )
    )


def _apply_legend(fig: go.Figure, theme: Theme) -> None:
    """Apply legend styling."""
    fig.update_layout(
        legend=dict(
            font=dict(
                size=theme.legend.font_size,
                family=theme.typography.font_family,
            ),
            bgcolor=theme.legend.bgcolor,
            bordercolor=theme.legend.border_color,
            borderwidth=theme.legend.border_width,
        )
    )


def _apply_colorbar_defaults(fig: go.Figure, theme: Theme) -> None:
    """Apply colorbar styling to all coloraxis."""
    colorbar_cfg = dict(
        orientation=theme.colorbar.orientation,
        x=theme.colorbar.x,
        y=theme.colorbar.y,
        xanchor=theme.colorbar.xanchor,
        yanchor=theme.colorbar.yanchor,
        thickness=theme.colorbar.thickness,
        len=theme.colorbar.len,
        tickfont=dict(
            size=theme.typography.tick_font_size,
            family=theme.typography.font_family,
        ),
    )

    for attr in dir(fig.layout):
        if attr.startswith("coloraxis"):
            coloraxis = getattr(fig.layout, attr)
            colorbar = getattr(coloraxis, "colorbar", None)
            if colorbar is None:
                coloraxis.update(dict(colorbar=colorbar_cfg))
            else:
                colorbar.update(colorbar_cfg)
