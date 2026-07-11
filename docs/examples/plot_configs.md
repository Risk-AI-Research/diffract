# YAML Plot Configurations

Diffract visualizations can be configured via YAML files using Hydra-style instantiation.

## Basic usage

```python
from diffract import Session

session = Session(profile="local")

with session:
    fig = session.viz.draw(config_path="examples/configs/boxplot_stable_rank.yaml")
    fig.show()
```

## Configuration structure

A plot config file has two sections:

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  plot:
    _target_: diffract.viz.plots.boxplot.BoxPlot
    y: stable_rank
    x: model_id

config:
  layout:
    width: 800
    height: 400
```

- **`plot`**: Hydra-style specification of the plot class and its parameters
- **`config`**: Additional Plotly layout options

### Strings: field names vs literals

Data axes (`x`, `y`, `z`, `group_by`) always interpret bare strings as field
names. Style properties (`marker_color`, `line_color`, `jitter_color`,
`marker_symbol`, `line_dash`) accept either a field name or a Plotly literal.
The rule is deterministic and applied at config load: a string that is a valid
Plotly literal of that kind (`"#1f77b4"`, `"steelblue"`, `"circle"`, `"dash"`)
stays a literal, and anything else refers to a field. To style by a field
whose name collides with a literal, or to control ordering and data type
explicitly, use a `FieldRef`:

```yaml
plot:
  _target_: diffract.viz.plots.boxplot.BoxPlot
  y: stable_rank
  x: model_id
  marker_color:
    _target_: diffract.viz.data.FieldRef
    field: red
```

Coerced bare strings use first-occurrence ordering and auto-detected data
type; `FieldRef` (optionally with `ordering`/`data_type`) is the explicit
form.

## Example: Box plot

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  plot:
    _target_: diffract.viz.plots.boxplot.BoxPlot
    y: stable_rank
    title: "Stable Rank by Model"
    x: model_id
    marker_color: model_id

config:
  layout:
    width: 800
    height: 400
```

## Example: Scatter plot

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  plot:
    _target_: diffract.viz.plots.scatter.ScatterPlot
    x: frob_norm
    y: stable_rank
    group_by: model_id
    marker_color: model_id
    title: "Frobenius Norm vs Stable Rank"

config:
  layout:
    width: 800
    height: 600
```

## Example: Grid layout

Combine multiple plots in a grid:

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  plot:
    _target_: diffract.viz.plots.subplots.GridPlot
    make_subplots_kwargs:
      rows: 1
      cols: 2
      shared_yaxes: false
      horizontal_spacing: 0.08
    subplots:
      - _target_: diffract.viz.plots.subplots.SubplotSpec
        row: 1
        col: 1
        title: "Greater Dim"
        plot:
          _target_: diffract.viz.plots.boxplot.BoxPlot
          y: greater_dim
          x: model_id
      - _target_: diffract.viz.plots.subplots.SubplotSpec
        row: 1
        col: 2
        title: "Stable Rank"
        plot:
          _target_: diffract.viz.plots.boxplot.BoxPlot
          y: stable_rank
          x: model_id

config:
  layout:
    title: "Comparison Grid"
    width: 1000
    height: 400
```

## Themes

Apply consistent styling with theme files:

```yaml
# theme.yaml
width: 800
height: 400

font_family: "Times New Roman"
title_font_size: 16
label_font_size: 14
tick_font_size: 12

background_color: "white"
grid_color: "lightgrey"
border_color: "black"

show_borders: true
show_x_grid: true
show_y_grid: true

discrete_colormap:
  - "navy"
  - "crimson"
  - "green"
  - "chocolate"

margin:
  l: 80
  r: 40
  t: 60
  b: 80
```

Apply the theme:

```python
with session:
    fig = session.viz.draw(
        config_path="my_plot.yaml",
        theme_path="my_theme.yaml"
    )
```

## Available plot types

| Class | Import path | Description |
|-------|-------------|-------------|
| `BoxPlot` | `diffract.viz.plots.boxplot.BoxPlot` | Box plot for scalar fields |
| `ViolinPlot` | `diffract.viz.plots.violin.ViolinPlot` | Violin plot for distributions |
| `ScatterPlot` | `diffract.viz.plots.scatter.ScatterPlot` | 2D scatter plot |
| `SparklinePlot` | `diffract.viz.plots.sparkline.SparklinePlot` | Line/sparkline plot over a field |
| `HeatmapPlot` | `diffract.viz.plots.heatmap.HeatmapPlot` | Heatmap pivoted by two categorical fields |
| `ClusterBarChart` | `diffract.viz.plots.cluster.ClusterBarChart` | Binned profiles of array fields, one trace per group |
| `GridPlot` | `diffract.viz.plots.subplots.GridPlot` | Multi-plot grid |

## Bundled examples

The repository includes ready-to-use configs in `examples/configs/`:

- `boxplot_stable_rank.yaml` — basic box plot
- `boxplot_stable_rank_jitter.yaml` — box plot with jittered points
- `violin_weights_svals_jitter.yaml` — violin plot with overlay
- `grid_example.yaml` — multi-panel layout
- `theme_example.yaml` — custom theme
