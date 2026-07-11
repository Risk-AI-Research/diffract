# ScatterPlot

2D scatter plot for visualizing relationships between two numeric fields.

Class: `diffract.viz.plots.scatter.ScatterPlot` (also
`diffract.viz.plots.ScatterPlot`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.scatter import ScatterPlot

with session:
    plot = ScatterPlot(x=FieldRef("frob_norm"), y=FieldRef("stable_rank"))
    fig = session.viz.draw(plot=plot)
```

Or using the convenience method:

```python
with session:
    fig = session.viz.scatter(x="frob_norm", y="stable_rank")
```

## Example: Group and color by model

`group_by` splits entries into separate traces (legend entries); `marker_color`
controls point color:

```python
plot = ScatterPlot(
    x=FieldRef("frob_norm"),
    y=FieldRef("stable_rank"),
    group_by=FieldRef("model_id"),
    marker_color=FieldRef("model_id"),
)
```

## Example: Multi-dimensional mapping

```python
plot = ScatterPlot(
    x=FieldRef("frob_norm"),
    y=FieldRef("stable_rank"),
    marker_color=FieldRef("layer_id"),      # color by layer
    marker_size=FieldRef("effective_rank"), # size by effective rank
    marker_symbol=FieldRef("kind"),         # symbol by parameter kind
    marker_opacity=FieldRef("head_id"),     # opacity by head
)
```

## Example: With filtering

```python
plot = ScatterPlot(
    x=FieldRef("frob_norm"),
    y=FieldRef("stable_rank"),
    value_filter={
        "frob_norm": (">", 1.0),
        "stable_rank": ("<", 100.0),
    },
)
```

## Parameters

### Fields (required, keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `FieldRef` | Field for X-axis values |
| `y` | `FieldRef` | Field for Y-axis values |

### Grouping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_by` | `FieldRef \| None` | `None` | Field to split points into separate traces (legend entries) |

### Rescaling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_rescale_range` | `tuple[float, float] \| None` | `None` | Rescale x-values into this range |
| `y_rescale_range` | `tuple[float, float] \| None` | `None` | Rescale y-values into this range |
| `x_rescale_traces_separately` | `bool` | `False` | Rescale x per trace |
| `y_rescale_traces_separately` | `bool` | `False` | Rescale y per trace |

### Titles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `None` | Figure title (defaults to `"{y.field} vs {x.field}"`) |
| `x_title` | `str \| None` | `None` | X-axis title |
| `y_title` | `str \| None` | `None` | Y-axis title |

### Marker styling (from `SupportsMarker`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_color` | `FieldRef \| str \| None` | `None` | Point color (constant or by field) |
| `marker_symbol` | `FieldRef \| str \| None` | `None` | Point symbol |
| `marker_size` | `FieldRef \| float \| None` | `6` | Point size |
| `marker_size_range` | `tuple[float, float] \| None` | `None` | Range to scale sizes into |
| `marker_opacity` | `FieldRef \| float \| None` | `0.7` | Point opacity |
| `marker_opacity_range` | `tuple[float, float] \| None` | `None` | Range to scale opacity into |

Continuous `marker_color` mapping uses a coloraxis; control it with
`marker_colorscale`, `marker_cmin`, `marker_cmax`, `marker_showscale`,
`marker_colorbar_title`.

### Value filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | `None` | Filter conditions |

### Theming

Pass a `Theme` at render time:

```python
from diffract.viz.styling import DARK_THEME
fig = session.viz.draw(plot=plot, theme=DARK_THEME)
```

## YAML configuration

```yaml
plot:
  _target_: diffract.viz.plots.scatter.ScatterPlot
  x: frob_norm
  y: stable_rank
  title: "Frobenius Norm vs Stable Rank"
  group_by: model_id
  marker_color: layer_id
  marker_size: effective_rank
```

## Dimension mapping details

### Color

`marker_color` can map to:

1. **Categorical field** (discrete): each category gets a distinct palette
   color.
2. **Numeric field** (continuous): values are mapped through a colorscale via a
   coloraxis (`marker_colorscale`, `marker_cmin`, `marker_cmax`).

```python
# Discrete: each model gets a distinct color
plot = ScatterPlot(x=FieldRef("x"), y=FieldRef("y"), marker_color=FieldRef("model_id"))

# Continuous: color gradient by layer
plot = ScatterPlot(x=FieldRef("x"), y=FieldRef("y"), marker_color=FieldRef("layer_id"))
```

### Size

Maps a field value to marker size, optionally scaled into `marker_size_range`:

```python
plot = ScatterPlot(
    x=FieldRef("frob_norm"),
    y=FieldRef("stable_rank"),
    marker_size=FieldRef("effective_rank"),
    marker_size_range=(6, 20),  # min → 6, max → 20
)
```

### Symbol

Maps a categorical field to marker symbols (cycling through the theme's symbol
palette):

```python
plot = ScatterPlot(
    x=FieldRef("frob_norm"),
    y=FieldRef("stable_rank"),
    marker_symbol=FieldRef("kind"),
)
```

### Opacity

Constant or mapped to a field (optionally scaled into `marker_opacity_range`):

```python
# Constant opacity
plot = ScatterPlot(x=FieldRef("x"), y=FieldRef("y"), marker_opacity=0.5)

# Variable opacity by field
plot = ScatterPlot(
    x=FieldRef("x"),
    y=FieldRef("y"),
    marker_opacity=FieldRef("layer_id"),
    marker_opacity_range=(0.1, 1.0),
)
```

## How it works

1. Fetches data via `DataProvider.fetch([...], ...)`.
2. Keeps only entries with non-None x **and** y.
3. Groups entries by the `group_by` field (one trace per group).
4. Creates a `go.Scatter` trace per group.
5. Applies marker mappings (color, size, symbol, opacity).
6. Applies the theme when one is passed to `render`.

## Notes

- Only entries with valid (non-None) x and y values are plotted.
- Each group appears as a separate legend entry.
- Color mapping uses the theme's `PaletteBundle` (categorical) or a coloraxis
  (continuous).

## See also

- [BoxPlot](boxplot.md) — For distributions of a single field
- [Sparkline](line.md) — For a numeric field vs an ordered field
- [Styling](../styling.md) — Palettes and color mapping
