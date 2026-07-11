# Visualization

Diffract provides a Plotly-based visualization system for creating publication-ready plots from computed fields. The `viz` module supports multiple workflows: simple one-liners, configurable plot objects, and YAML-based reproducible configs.

## Prerequisites

Install the visualization extra:

```bash
uv sync --extra viz
# or
pip install "diffract[viz]"
```

## Quick start

The simplest way to visualize is via `session.viz` wrapper methods:

```python
from diffract import Session

session = Session(profile="local")

with session:
    session.models.add(model, model_id="my-model")
    session.compute.apply("frob_norm", "stable_rank")
    
    # One-liner box plot
    fig = session.viz.box(y="stable_rank", x="model_id")
    fig.show()
```

## Three ways to create plots

### 1. Simple methods (recommended for exploration)

```python
with session:
    fig = session.viz.box(y="stable_rank", x="model_id", marker_color="layer_id")
    fig = session.viz.scatter(x="frob_norm", y="stable_rank", group_by="model_id")
    fig = session.viz.violin(y="esd", x="model_id", jitter_enabled=True)
    fig = session.viz.heatmap(z="stable_rank", x="head_id", y="layer_id")
    fig = session.viz.line(y="frob_norm", x="in_model_idx", group_by="model_id")
```

### 2. Plot objects (recommended for Hydra configs)

```python
from diffract.viz.plots import BoxPlot
from diffract.viz.data import FieldRef

with session:
    plot = BoxPlot(
        y=FieldRef("stable_rank"),
        x=FieldRef("model_id"),
        jitter_enabled=True,
        jitter_color=FieldRef("layer_id"),
    )
    fig = session.viz.draw(plot=plot)
```

### 3. YAML config files (recommended for reproducibility)

```python
with session:
    fig = session.viz.draw(
        config_path="configs/boxplot_stable_rank.yaml",
        overrides=["plot.plot.x=layer_id"],  # Hydra overrides
    )
```

## Available plot types

| Plot | Class | `session.viz` method | Description |
|------|-------|----------------------|-------------|
| Box | `BoxPlot` | `session.viz.box` | Box plot for scalar field distributions |
| Violin | `ViolinPlot` | `session.viz.violin` | Violin plot with optional KDE |
| Scatter | `ScatterPlot` | `session.viz.scatter` | 2D scatter plot for two scalar fields |
| Heatmap | `HeatmapPlot` | `session.viz.heatmap` | Heatmap pivoted by two metadata keys |
| Line / sparkline | `SparklinePlot` | `session.viz.line` / `session.viz.sparkline` | Line plot of a field vs metadata |
| Cluster | `ClusterBarChart` | — (config / `session.viz.draw`) | Clustered line chart of binned array-like fields (e.g., singular values) |
| Grid | `GridPlot` | `session.viz.grid` | Multi-plot grid layout |

All plot classes are re-exported from `diffract.viz.plots`. `ClusterBarChart`
(defined in `diffract.viz.plots.cluster`) has no dedicated wrapper method:
construct it directly and render with `session.viz.draw(plot=...)`, or drive it
from a YAML config.

## Dimension mappings

Most `session.viz` methods map fields or metadata keys to visual dimensions.
Values are field/metadata names (converted to `FieldRef` internally):

| Dimension | Applies to | Description | Example |
|-----------|------------|-------------|---------|
| `x` / `y` | all | Axis values / categories | `x="model_id"` |
| `z` | heatmap | Cell values | `z="stable_rank"` |
| `group_by` | scatter, line | Split points/series into traces | `group_by="model_id"` |
| `marker_color` | box, violin, scatter | Color markers by a field | `marker_color="layer_id"` |
| `marker_symbol` | box, violin, scatter | Marker symbol by a field | `marker_symbol="kind"` |
| `marker_size` | box, violin, scatter | Marker size by a field | `marker_size="frob_norm"` |
| `line_color` | line | Color lines by a field | `line_color="layer_id"` |
| `line_dash` | line | Line dash pattern by a field | `line_dash="model_id"` |

## Value filtering

Filter data before plotting without modifying the session:

```python
with session:
    # Only parameters where stable_rank > 10
    fig = session.viz.box(
        y="stable_rank",
        x="model_id",
        value_filter={"stable_rank": (">", 10.0)}
    )
    
    # Multiple conditions
    fig = session.viz.scatter(
        x="frob_norm",
        y="stable_rank",
        value_filter={
            "frob_norm": (">", 1.0),
            "stable_rank": ("<=", 100),
        }
    )
```

Supported operators: `>`, `<`, `>=`, `<=`, `==`, `!=`.

## Axis types and category ordering

Heatmap axes are categorical and accept Plotly-style category ordering on both
axes (`x_categoryorder` / `x_categoryarray`, `y_categoryorder` /
`y_categoryarray`), through the `session.viz.heatmap` wrapper and the
`HeatmapPlot` class alike:

```python
from diffract.viz.plots import HeatmapPlot
from diffract.viz.data import FieldRef

with session:
    # Sort heatmap rows
    fig = session.viz.heatmap(
        z="stable_rank", x="head_id", y="layer_id",
        y_categoryorder="category ascending",
    )
    
    # Order both axes via the plot object
    plot = HeatmapPlot(
        z=FieldRef("stable_rank"), x=FieldRef("head_id"), y=FieldRef("layer_id"),
        x_categoryorder="category ascending",
        y_categoryorder="category descending",
    )
    fig = session.viz.draw(plot=plot)
```

`categoryorder` accepts the usual Plotly values (`"trace"`,
`"category ascending"`, `"category descending"`, `"array"`, etc.).

Line/sparkline plots infer the x-axis data type from the data: numeric values
(e.g., `in_model_idx`) produce a numeric axis, string values (e.g., `model_id`)
a categorical one. Pass `x_axis_mode="numeric"` or `"categorical"` to override
the inference. On a categorical x axis, `x_categoryorder` / `x_categoryarray`
control the Plotly-level ordering; alternatively, attach an `Ordering` to the
x `FieldRef` to reorder the data itself:

```python
from diffract.viz.data import FieldRef, Ordering, OrderMode

with session:
    # Numeric x inferred from data
    fig = session.viz.line(y="frob_norm", x="in_model_idx", group_by="model_id")
    
    # Force categorical treatment of a numeric key
    fig = session.viz.line(y="frob_norm", x="in_model_idx", x_axis_mode="categorical")
    
    # Plotly-level ordering of categorical x values
    fig = session.viz.line(
        y="frob_norm", x="model_id",
        x_categoryorder="category descending",
    )
    
    # Custom explicit order of categorical x values
    fig = session.viz.line(
        y="frob_norm",
        x=FieldRef(
            "model_id",
            ordering=Ordering(mode=OrderMode.CUSTOM, custom_order=["run-2", "run-1"]),
        ),
    )
```

See `Ordering` / `OrderMode` in `diffract.viz.data` for the available ordering
modes.

## Jitter overlays

Box and violin plots support jitter overlays for showing individual points:

```python
with session:
    fig = session.viz.box(
        y="stable_rank",
        x="model_id",
        jitter_enabled=True,
        jitter_color="layer_id",  # Color points by layer
        jitter_width=0.15,
        jitter_density_scale=True,  # Wider spread in dense regions
        jitter_colorscale="Viridis",
    )
```

## Theming

Apply consistent styling with themes:

```python
from diffract.viz.styling import DEFAULT_THEME, DARK_THEME, MINIMAL_THEME

with session:
    # Use a predefined theme
    fig = session.viz.box(y="stable_rank", x="model_id", theme=DARK_THEME)
    
    # Custom theme (composed of nested style groups)
    from diffract.viz.styling import (
        Theme,
        LayoutStyle,
        TypographyStyle,
        BackgroundStyle,
    )
    
    my_theme = Theme(
        layout=LayoutStyle(width=1000, height=500),
        typography=TypographyStyle(font_family="Arial"),
        background=BackgroundStyle(plot_bgcolor="#f0f0f0", paper_bgcolor="#f0f0f0"),
    )
    fig = session.viz.box(y="stable_rank", x="model_id", theme=my_theme)
```

Load themes from YAML:

```python
with session:
    fig = session.viz.draw(
        config_path="plots/my_plot.yaml",
        theme_path="themes/publication.yaml",
    )
```

## Subplots and grids

Combine multiple plots in a grid:

```python
from diffract.viz.plots import BoxPlot, SubplotSpec
from diffract.viz.data import FieldRef

with session:
    fig = session.viz.grid(
        subplots=[
            SubplotSpec(
                row=1, col=1,
                title="Model A",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
                session_filter={"model_ids": ["model_a"]},
            ),
            SubplotSpec(
                row=1, col=2,
                title="Model B",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
                session_filter={"model_ids": ["model_b"]},
            ),
        ],
        make_subplots_kwargs={"shared_yaxes": True},
    )
```

Each subplot can carry its own `session_filter` (valid keys: `param_ids`,
`param_names`, `param_types`, `model_ids`) and/or `value_filter` for per-subplot
data selection.

To generate a grid from a plot template plus row/column binds instead of
listing every subplot by hand, use `session.viz.bound_grid` — see
[Grid plots](../../reference/viz/plots/grid.md) for the full parametric grid
API.

## Advanced: Plotly customization

Wrap any plot with `UpdateFigure` for full Plotly customization:

```python
from diffract.viz.plots import BoxPlot, UpdateFigure
from diffract.viz.data import FieldRef

with session:
    wrapped = UpdateFigure(
        plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
        layout={"title": "Custom Title", "showlegend": False},
        xaxes={"tickangle": -45},
        traces={"marker_opacity": 0.7},
    )
    fig = session.viz.draw(plot=wrapped)
```

## Working with array-like fields

Some fields are arrays (e.g., singular values, eigenvalues). Violin plots flatten
arrays into samples for distribution views:

```python
with session:
    fig = session.viz.violin(y="weights_svals", x="model_id")
```

## Next steps

- [Plot configs](../../examples/plot_configs.md) — YAML configuration reference
- [Visualization showcase](../../examples/viz_showcase.md) — Full examples with output
- [Themes reference](../../reference/viz/styling.md) — Theme options

