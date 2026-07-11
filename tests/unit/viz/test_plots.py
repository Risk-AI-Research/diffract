"""Smoke tests for plot classes."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestBoxPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot

        plot = BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_theme(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot
        from diffract.viz.styling import MINIMAL_THEME

        plot = BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
        fig = plot.render(mock_session, theme=MINIMAL_THEME)

        assert fig.layout.width == MINIMAL_THEME.layout.width

    def test_render_with_color(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot

        plot = BoxPlot(
            y=FieldRef("stable_rank"),
            x=FieldRef("model_id"),
            marker_color=FieldRef("layer_id"),
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        assert fig.layout is not None

    def test_render_with_x_order(self, mock_session):
        from diffract.viz.data import FieldRef, as_is
        from diffract.viz.plots.boxplot import BoxPlot

        plot = BoxPlot(
            y=FieldRef("stable_rank"),
            x=FieldRef("model_id", ordering=as_is()),
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_value_filter(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot

        plot = BoxPlot(
            y=FieldRef("stable_rank"),
            x=FieldRef("model_id"),
            value_filter={"frob_norm": (">", 5.0)},
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestViolinPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.violin import ViolinPlot

        plot = ViolinPlot(y=FieldRef("weights_svals"), x=FieldRef("model_id"))
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_theme(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.violin import ViolinPlot
        from diffract.viz.styling import DARK_THEME

        plot = ViolinPlot(y=FieldRef("weights_svals"), x=FieldRef("model_id"))
        fig = plot.render(mock_session, theme=DARK_THEME)

        assert fig.layout.plot_bgcolor == DARK_THEME.background.plot_bgcolor


class TestScatterPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.scatter import ScatterPlot

        plot = ScatterPlot(x=FieldRef("stable_rank"), y=FieldRef("frob_norm"))
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_size_mapping(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.scatter import ScatterPlot

        plot = ScatterPlot(
            x=FieldRef("stable_rank"),
            y=FieldRef("frob_norm"),
            marker_size=FieldRef("greater_dim"),
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestHeatmapPlot:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.heatmap import HeatmapPlot

        plot = HeatmapPlot(
            z=FieldRef("stable_rank"),
            y=FieldRef("layer_id"),
            x=FieldRef("head_id"),
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_sorting(self, mock_session):
        from diffract.viz.data import FieldRef, as_is, numeric
        from diffract.viz.plots.heatmap import HeatmapPlot

        plot = HeatmapPlot(
            z=FieldRef("stable_rank"),
            y=FieldRef("layer_id", ordering=as_is()),
            x=FieldRef("head_id", ordering=numeric()),
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestSparkline:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots import Sparkline

        plot = Sparkline(y=FieldRef("stable_rank"), x=FieldRef("layer_id"))
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_render_with_categorical_x(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots import Sparkline

        plot = Sparkline(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
        fig = plot.render(mock_session)

        assert len(fig.data) >= 1
        assert all(isinstance(v, str) for v in fig.data[0].x)

    def test_render_with_forced_categorical_x(self, mock_session):
        from diffract.viz.data import DataType, FieldRef
        from diffract.viz.plots import Sparkline

        plot = Sparkline(
            y=FieldRef("stable_rank"),
            x=FieldRef("layer_id", data_type=DataType.CATEGORICAL),
        )
        fig = plot.render(mock_session)

        assert len(fig.data) >= 1
        assert all(isinstance(v, str) for v in fig.data[0].x)

    def test_categorical_x_rejects_rescale(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots import Sparkline

        plot = Sparkline(
            y=FieldRef("stable_rank"),
            x=FieldRef("model_id"),
            x_rescale_range=(0.0, 1.0),
        )
        with pytest.raises(ValueError, match="rescal"):
            plot.render(mock_session)

    def test_viz_namespace_line_wrapper(self, mock_session):
        from diffract.session.namespaces.viz.sparkline import sparkline

        captured: dict[str, object] = {}

        class FakeViz:
            def draw(self, *, plot, theme=None, theme_path=None):
                assert theme_path is None
                captured["plot"] = plot
                return plot.render(mock_session, theme=theme)

        fig = sparkline(
            FakeViz(),
            y="stable_rank",
            x="layer_id",
            group_by="model_id",
            x_axis_mode="categorical",
            x_categoryorder="category descending",
        )

        assert len(fig.data) >= 1
        assert captured["plot"].x_axis_mode == "categorical"
        assert all(isinstance(v, str) for v in fig.data[0].x)
        assert fig.layout.xaxis.categoryorder == "category descending"

    def test_viz_namespace_heatmap_wrapper_forwards_categoryorder(self, mock_session):
        from diffract.session.namespaces.viz.heatmap import heatmap

        class FakeViz:
            def draw(self, *, plot, theme=None, theme_path=None):
                assert theme_path is None
                return plot.render(mock_session, theme=theme)

        fig = heatmap(
            FakeViz(),
            z="stable_rank",
            x="head_id",
            y="layer_id",
            x_categoryorder="category ascending",
            y_categoryorder="category descending",
        )

        assert fig.layout.xaxis.categoryorder == "category ascending"
        assert fig.layout.yaxis.categoryorder == "category descending"


class TestClusterBarChart:
    def test_render_returns_figure(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        plot = ClusterBarChart(field="weights_svals", group_by=["model_id"])
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_render_with_full_config(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        plot = ClusterBarChart(
            field="weights_svals",
            title="Cluster bar chart (weights_svals)",
            parameter_names=["re:.*"],
            parameter_types=["weight"],
            parameter_uids=None,
            model_ids=None,
            group_by=["model_id", "layer_id", "head_id"],
            aggregate_by="head_id",
            color_by="layer_id",
            dash_by="model_id",
            marker_by="model_id",
            num_bins=18,
            binning="exponential",
            left_bound=None,
            right_bound=None,
            draw_statistics=True,
            mode="lines+markers",
            legend_format=None,
            legend_keys=None,
        )
        fig = plot.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Cluster bar chart (weights_svals)"

    def test_render_binning_produces_expected_bins(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        plot = ClusterBarChart(
            field="weights_svals",
            group_by=["model_id"],
            num_bins=10,
            binning="linear",
        )
        fig = plot.render(mock_session)

        # One lines+markers trace per model (model_a, model_b), each with 10 bins.
        main_traces = [t for t in fig.data if not t.name.endswith("(std)")]
        assert len(main_traces) == 2
        for trace in main_traces:
            assert len(trace.y) == 10

    def test_render_statistics_draws_std_trace(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        # Aggregating model_a's two heads yields a group with >1 member, so a
        # std trace (drawn below zero) is emitted for it.
        plot = ClusterBarChart(
            field="weights_svals",
            group_by=["model_id", "head_id"],
            aggregate_by="head_id",
            draw_statistics=True,
        )
        fig = plot.render(mock_session)

        std_traces = [t for t in fig.data if t.name.endswith("(std)")]
        assert len(std_traces) >= 1
        assert all(min(t.y) <= 0 for t in std_traces)

    def test_render_with_model_filter(self, mock_session):
        from diffract.viz.plots.cluster import ClusterBarChart

        plot = ClusterBarChart(
            field="weights_svals",
            group_by=["model_id"],
            model_ids=["model_a"],
        )
        fig = plot.render(mock_session)

        names = [t.name for t in fig.data]
        assert "model_id=model_a" in names
        assert "model_id=model_b" not in names


class TestUpdateFigure:
    def test_render_applies_updates(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.base import UpdateFigure
        from diffract.viz.plots.boxplot import BoxPlot

        inner = BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
        wrapper = UpdateFigure(
            plot=inner,
            layout={"title": "Custom Title"},
        )
        fig = wrapper.render(mock_session)

        assert fig.layout.title.text == "Custom Title"


class TestGridPlot:
    def test_render_creates_subplots(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot
        from diffract.viz.plots.subplots import GridPlot, SubplotSpec

        specs = [
            SubplotSpec(
                row=1,
                col=1,
                title="Plot 1",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
            ),
            SubplotSpec(
                row=1,
                col=2,
                title="Plot 2",
                plot=BoxPlot(y=FieldRef("frob_norm"), x=FieldRef("model_id")),
            ),
        ]
        grid = GridPlot(subplots=specs, make_subplots_kwargs={})
        fig = grid.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        assert fig.layout is not None

    def test_render_with_per_subplot_filter(self, mock_session):
        from diffract.viz.data import FieldRef
        from diffract.viz.plots.boxplot import BoxPlot
        from diffract.viz.plots.subplots import GridPlot, SubplotSpec

        specs = [
            SubplotSpec(
                row=1,
                col=1,
                title="Model A",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
                filter={"model_ids": ["model_a"]},
            ),
            SubplotSpec(
                row=1,
                col=2,
                title="Model B",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
                filter={"model_ids": ["model_b"]},
            ),
        ]
        grid = GridPlot(subplots=specs, make_subplots_kwargs={})
        fig = grid.render(mock_session)

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


class TestValueFilter:
    def test_apply_value_filter(self, sample_results):
        from diffract.viz.data import apply_filter

        filtered = apply_filter(sample_results, {"metric": (">", 1.5)})
        assert len(filtered) == 2
        assert "p1" not in filtered

    def test_apply_value_filter_all_operators(self, sample_results):
        from diffract.viz.data import apply_filter

        # Greater than
        assert len(apply_filter(sample_results, {"metric": (">", 1.0)})) == 2
        # Less than
        assert len(apply_filter(sample_results, {"metric": ("<", 2.0)})) == 1
        # Equal
        assert len(apply_filter(sample_results, {"metric": ("==", 2.0)})) == 1
        # Not equal
        assert len(apply_filter(sample_results, {"metric": ("!=", 2.0)})) == 2
