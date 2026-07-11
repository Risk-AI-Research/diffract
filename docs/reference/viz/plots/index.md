# Plot Types

Reference documentation for all available plot classes.

## Overview

| Plot | Class | Best for |
|------|-------|----------|
| [Box](boxplot.md) | `BoxPlot` | Scalar/vector field distributions by category |
| [Violin](violin.md) | `ViolinPlot` | Distributions with density visualization |
| [Scatter](scatter.md) | `ScatterPlot` | Relationships between two numeric fields |
| [Heatmap](heatmap.md) | `HeatmapPlot` | 2D grids pivoted by two categorical fields |
| [Line](line.md) | `SparklinePlot` | A numeric field vs another field (layers, steps) |
| [Cluster](cluster.md) | `ClusterBarChart` | Binned profiles of array fields (e.g. singular value spectra), one trace per group |
| [Grid](grid.md) | `GridPlot` | Multi-plot layouts |

`SparklinePlot` is also exported under the alias `Sparkline`.

## Fields as `FieldRef`

Plot constructors take fields as `FieldRef` objects, not bare strings. A
`FieldRef` names the field and (optionally) how its values are ordered:

```python
from diffract.viz.data import FieldRef, numeric, custom

FieldRef("stable_rank")                      # default (as-is) ordering
FieldRef("layer_id", ordering=numeric())     # numeric ordering
FieldRef("model_id", ordering=custom(["baseline", "finetuned", "pruned"]))
```

The `session.viz.*` convenience methods accept plain strings and wrap them in
`FieldRef` for you (see [Session viz methods](#session-viz-methods) below).

## Common parameters

Most plots share these parameters (defined on the `Plot` base class and its
axis mixins).

### Titles

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | `str \| None` | Figure title |
| `x_title` | `str \| None` | X-axis title |
| `y_title` | `str \| None` | Y-axis title |

### Ordering

Ordering is expressed via the `ordering` attribute of each `FieldRef`, using
the helpers from `diffract.viz.data`:

```python
from diffract.viz.data import FieldRef, as_is, lexicographic, numeric, by_key, custom

FieldRef("model_id", ordering=lexicographic())
FieldRef("layer_id", ordering=numeric(descending=True))
FieldRef("model_id", ordering=custom(["baseline", "finetuned", "pruned"]))
```

Categorical axes additionally expose `x_categoryorder` / `x_categoryarray`
(and `y_*` for heatmaps) for Plotly-level control.

### Filtering

| Parameter | Type | Description |
|-----------|------|-------------|
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | Filter entries by field value |

### Marker / line mappings

Visual encodings are configured through the marker and line mixin fields, for
example `marker_color`, `marker_symbol`, `marker_size`, `marker_opacity`,
`line_color`, `line_dash`, and `line_width`. Each accepts either a constant or
a `FieldRef` to map the property from a data field.

## Import paths

```python
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.violin import ViolinPlot
from diffract.viz.plots.scatter import ScatterPlot
from diffract.viz.plots.heatmap import HeatmapPlot
from diffract.viz.plots.sparkline import SparklinePlot  # alias: Sparkline
from diffract.viz.plots.cluster import ClusterBarChart
from diffract.viz.plots.subplots import GridPlot, SubplotSpec
from diffract.viz.plots.base import UpdateFigure
```

All of the above are also re-exported directly from `diffract.viz.plots`:

```python
from diffract.viz.plots import BoxPlot, ViolinPlot, ScatterPlot, HeatmapPlot
from diffract.viz.plots import SparklinePlot, Sparkline, ClusterBarChart
from diffract.viz.plots import GridPlot, SubplotSpec, UpdateFigure
```

## Plot protocol

All plots implement the `Plot` protocol:

```python
from typing import Protocol
import plotly.graph_objects as go
from diffract.session import Session
from diffract.viz.styling import Theme

class Plot(Protocol):
    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render the plot using data from the session."""
        ...
```

This allows any plot to be used with `session.viz.draw(plot=...)`.

## Customizing rendered figures

`UpdateFigure` (in `diffract.viz.plots.base`) wraps any plot and applies Plotly
`update_*` calls after rendering — it is the escape hatch for figure tweaks the
plot classes do not expose directly:

```python
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.base import UpdateFigure

wrapped = UpdateFigure(
    plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
    layout={"title": "Custom Title", "showlegend": False},
    xaxes={"tickangle": -45},
)
fig = session.viz.draw(plot=wrapped)
```

`UpdateFigure` fields: `plot` (required), `config`, `update`, `layout`,
`traces`, `xaxes`, `yaxes`. A `theme` may be passed to `draw(...)` and is
applied after the updates.

## Session viz methods

For quick exploration, the session exposes convenience methods that accept
plain strings and build the corresponding plot for you:

```python
session.viz.box(y="stable_rank", x="model_id")
session.viz.violin(y="esd", x="model_id")
session.viz.scatter(x="frob_norm", y="stable_rank", group_by="model_id")
session.viz.heatmap(z="stable_rank", x="head_id", y="layer_id")
session.viz.sparkline(y="stable_rank", x="layer_id", group_by="model_id")
session.viz.line(...)   # alias of sparkline
session.viz.grid(subplots=[...])
session.viz.bound_grid(plot_template=..., row=..., col=...)
```

```{toctree}
:hidden:
:maxdepth: 1

boxplot
violin
scatter
heatmap
line
cluster
grid
```
