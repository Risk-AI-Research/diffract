"""Built-in plots for `diffract.viz`.

This module mirrors (approximately) the public surface of `diffract.viz.plots`
for the refactored plotting stack.
"""

try:
    import plotly
except ImportError as e:
    _msg = (
        "diffract.viz requires the viz extra. "
        'Install it with: pip install "diffract-core[viz]"'
    )
    raise ImportError(_msg) from e

from .base import UpdateFigure
from .boxplot import BoxPlot
from .cluster import ClusterBarChart
from .heatmap import HeatmapPlot
from .scatter import ScatterPlot
from .sparkline import SparklinePlot
from .subplots import GridPlot, SubplotSpec
from .violin import ViolinPlot

Sparkline = SparklinePlot

__all__ = [
    "BoxPlot",
    "ClusterBarChart",
    "GridPlot",
    "HeatmapPlot",
    "ScatterPlot",
    "Sparkline",
    "SparklinePlot",
    "SubplotSpec",
    "ViolinPlot",
]
