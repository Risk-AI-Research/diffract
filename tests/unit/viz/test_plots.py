"""Smoke tests for plot classes."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

class TestBoxPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.scalar import BoxPlot

        plot = BoxPlot(field="stable_rank")
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_theme(self, mock_session):
        from diffract.viz.plots.scalar import BoxPlot
        from diffract.viz.themes import MINIMAL_THEME

        plot = BoxPlot(field="stable_rank", theme=MINIMAL_THEME)
        fig = plot.render(mock_session)

        assert fig.layout.width == MINIMAL_THEME.width

    def test_render_with_color_by(self, mock_session):
        from diffract.viz.plots.scalar import BoxPlot

        plot = BoxPlot(field="stable_rank", color_by="layer_id")
        fig = plot.render(mock_session)

        assert len(fig.data) > 0


class TestViolinPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.violin import ViolinPlot

        plot = ViolinPlot(field="weights_svals")
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_theme(self, mock_session):
        from diffract.viz.plots.violin import ViolinPlot
        from diffract.viz.themes import DARK_THEME

        plot = ViolinPlot(field="weights_svals", theme=DARK_THEME)
        fig = plot.render(mock_session)

        assert fig.layout.plot_bgcolor == DARK_THEME.background_color


class TestScatterPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.scatter import ScatterPlot

        plot = ScatterPlot(x_field="stable_rank", y_field="frob_norm")
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestHeatmapPivotPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.heatmap import HeatmapPivotPlot

        plot = HeatmapPivotPlot(
            value_field="stable_rank", row_by="layer_id", col_by="head_id"
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestLineByMetaPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.lines import LineByMetaPlot

        plot = LineByMetaPlot(y_field="stable_rank", x_by="layer_id")
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestClusterBarChart:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        plot = ClusterBarChart(field="weights_svals")
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestUpdateFigure:
    def test_render_applies_updates(self, mock_session):
        from diffract.viz.plots.configurer import UpdateFigure
        from diffract.viz.plots.scalar import BoxPlot

        inner = BoxPlot(field="stable_rank")
        wrapper = UpdateFigure(
            plot=inner,
            layout={"title": "Custom Title"},
        )
        fig = wrapper.render(mock_session)

        assert fig.layout.title.text == "Custom Title"


class TestGridPlot:
    def test_render_creates_subplots(self, mock_session):
        from diffract.viz.plots.scalar import BoxPlot
        from diffract.viz.plots.subplots import GridPlot, SubplotSpec

        specs = [
            SubplotSpec(row=1, col=1, title="Plot 1", plot=BoxPlot(field="stable_rank")),
            SubplotSpec(row=1, col=2, title="Plot 2", plot=BoxPlot(field="frob_norm")),
        ]
        grid = GridPlot(subplots=specs, make_subplots_kwargs={})
        fig = grid.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        # Should have traces from both subplots
        assert len(fig.data) >= 2
