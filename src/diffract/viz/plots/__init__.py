"""Built-in plots for diffract.viz.

All plots follow the Plot protocol and support:
- `theme: Theme | None` for consistent styling
- `color_by: str | None` for coloring by metadata keys

Available plots:
- BoxPlot: Box plots for scalar fields
- ViolinPlot: Violin plots for scalar/array-like fields
- ScatterPlot: Scatter plots for two scalar fields
- HeatmapPivotPlot: Heatmaps pivoted by metadata keys
- LineByMetaPlot: Line plots with x-axis from metadata
- ClusterBarChart: Binned clustered line charts for array-like fields
- UpdateFigure: Wrapper for Plotly customization
- GridPlot: Subplot composition
"""

from .cluster import ClusterBarChart
from .configurer import UpdateFigure
from .heatmap import HeatmapPivotPlot
from .lines import LineByMetaPlot
from .scalar import BoxPlot
from .scatter import ScatterPlot
from .subplots import GridPlot, SubplotSpec
from .violin import ViolinPlot

__all__ = [
    "BoxPlot",
    "ClusterBarChart",
    "GridPlot",
    "HeatmapPivotPlot",
    "LineByMetaPlot",
    "ScatterPlot",
    "SubplotSpec",
    "UpdateFigure",
    "ViolinPlot",
]
