# YAML Plot Configurations

Diffract visualizations can be configured via YAML files using Hydra-style instantiation.

## Basic usage

```python
from diffract import Session

session = Session(profile="local")

with session:
    fig = session.draw(config_path="examples/configs/boxplot_stable_rank.yaml")
    fig.show()
```

## Configuration structure

A plot config file has two sections:

```yaml
plot:
  _target_: diffract.viz.plots.configurer.UpdateFigure
  plot:
    _target_: diffract.viz.plots.scalar.BoxPlot
    field: stable_rank
    group_by: model_id

config:
  layout:
    width: 800
    height: 400
```

- **`plot`**: Hydra-style specification of the plot class and its parameters
- **`config`**: Additional Plotly layout options

## Example: Box plot

```yaml
plot:
  _target_: diffract.viz.plots.configurer.UpdateFigure
  plot:
    _target_: diffract.viz.plots.scalar.BoxPlot
    field: stable_rank
    title: "Stable Rank by Model"
    group_by: model_id
    color_by: model_id

config:
  layout:
    width: 800
    height: 400
```

## Example: Scatter plot

```yaml
plot:
  _target_: diffract.viz.plots.configurer.UpdateFigure
  plot:
    _target_: diffract.viz.plots.scatter.ScatterPlot
    x_field: frob_norm
    y_field: stable_rank
    color_by: model_id
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
  _target_: diffract.viz.plots.configurer.UpdateFigure
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
          _target_: diffract.viz.plots.scalar.BoxPlot
          field: greater_dim
          group_by: model_id
      - _target_: diffract.viz.plots.subplots.SubplotSpec
        row: 1
        col: 2
        title: "Stable Rank"
        plot:
          _target_: diffract.viz.plots.scalar.BoxPlot
          field: stable_rank
          group_by: model_id

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
    fig = session.draw(
        config_path="my_plot.yaml",
        theme_path="my_theme.yaml"
    )
```

## Available plot types

| Class | Import path | Description |
|-------|-------------|-------------|
| `BoxPlot` | `diffract.viz.plots.scalar.BoxPlot` | Box plot for scalar fields |
| `ViolinPlot` | `diffract.viz.plots.violin.ViolinPlot` | Violin plot for distributions |
| `ScatterPlot` | `diffract.viz.plots.scatter.ScatterPlot` | 2D scatter plot |
| `ClusterBarChart` | `diffract.viz.plots.cluster.ClusterBarChart` | Clustered bar chart |
| `GridPlot` | `diffract.viz.plots.subplots.GridPlot` | Multi-plot grid |

## Bundled examples

The repository includes ready-to-use configs in `examples/configs/`:

- `boxplot_stable_rank.yaml` — basic box plot
- `boxplot_stable_rank_jitter.yaml` — box plot with jittered points
- `violin_weights_svals_jitter.yaml` — violin plot with overlay
- `cluster_bar_chart_weights_svals.yaml` — clustered bar chart
- `grid_example.yaml` — multi-panel layout
- `theme_example.yaml` — custom theme
