# HeatmapPlot

Heatmap visualization that pivots a scalar field over two categorical fields.

Class: `diffract.viz.plots.heatmap.HeatmapPlot` (also
`diffract.viz.plots.HeatmapPlot`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.heatmap import HeatmapPlot

with session:
    plot = HeatmapPlot(
        z=FieldRef("stable_rank"),
        x=FieldRef("head_id"),
        y=FieldRef("layer_id"),
    )
    fig = session.viz.draw(plot=plot)
```

Or using the convenience method:

```python
with session:
    fig = session.viz.heatmap(z="stable_rank", x="head_id", y="layer_id")
```

## Example: Layer x Head heatmap

Visualize attention head metrics across layers:

```python
plot = HeatmapPlot(
    z=FieldRef("stable_rank"),
    x=FieldRef("head_id"),
    y=FieldRef("layer_id"),
    title="Stable Rank by Layer and Head",
    heatmap_colorscale="Viridis",
)
```

## Example: With text annotations

```python
plot = HeatmapPlot(
    z=FieldRef("frob_norm"),
    x=FieldRef("head_id"),
    y=FieldRef("layer_id"),
    show_text=True,
    text_format=".1f",
    text_font_size=8,
)
```

## Example: Custom axis order

Ordering is expressed on each `FieldRef`:

```python
from diffract.viz.data import FieldRef, lexicographic, numeric

plot = HeatmapPlot(
    z=FieldRef("stable_rank"),
    x=FieldRef("model_id", ordering=lexicographic()),
    y=FieldRef("layer_id", ordering=numeric()),
)
```

## Parameters

### Fields (required, keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `z` | `FieldRef` | Field for cell values (color intensity) |
| `x` | `FieldRef` | Categorical field for X-axis (columns) |
| `y` | `FieldRef` | Categorical field for Y-axis (rows) |

### Rescaling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z_rescale_range` | `tuple[float, float] \| None` | `None` | Rescale z-values into this range |

### Titles and axes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `None` | Figure title (defaults to `"{z.field} by {y.field} x {x.field}"`) |
| `x_title` | `str \| None` | `None` | X-axis title |
| `y_title` | `str \| None` | `None` | Y-axis title |
| `x_categoryorder` | `str \| None` | `None` | Plotly category order (columns) |
| `y_categoryorder` | `str \| None` | `None` | Plotly category order (rows) |

### Display options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fill_value` | `float` | `NaN` | Value for missing cells |
| `show_text` | `bool` | `False` | Show values as text in cells |
| `text_format` | `str` | `".2f"` | Format string for text (e.g., `.1f`, `.0%`) |
| `text_font_size` | `int` | `10` | Font size for cell text |
| `reverse_y` | `bool` | `True` | Reverse y-axis so the first row is at the top |

### Colorscale (from `SupportsColoraxis("heatmap")`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heatmap_colorscale` | `str` | `"Viridis"` | Plotly colorscale |
| `heatmap_showscale` | `bool` | `True` | Show the colorbar |
| `heatmap_cmin` | `float \| None` | `None` | Minimum for color mapping |
| `heatmap_cmax` | `float \| None` | `None` | Maximum for color mapping |
| `heatmap_colorbar_title` | `str \| None` | `None` | Colorbar title |

### Value filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | `None` | Filter conditions |

### Theming

Pass a `Theme` at render time (not a constructor field):

```python
from diffract.viz.styling import DARK_THEME
fig = session.viz.draw(plot=plot, theme=DARK_THEME)
```

## YAML configuration

```yaml
plot:
  _target_: diffract.viz.plots.heatmap.HeatmapPlot
  z: stable_rank
  x: head_id
  y: layer_id
  title: "Stable Rank Heatmap"
  heatmap_colorscale: "RdBu"
  show_text: true
  text_format: ".1f"
```

## How it works

1. Fetches data via `DataProvider.fetch([...], ...)`.
2. Collects unique x values (columns) and y values (rows), ordered by their
   `FieldRef` ordering.
3. Builds a 2D matrix: `matrix[row, col] = z value`.
4. Missing cells get `fill_value` (NaN by default).
5. Creates a `go.Heatmap` trace with optional text annotations.
6. Reverses the y-axis when `reverse_y=True` (so row 0 is at the top).
7. Applies the theme when one is passed to `render`.

## Metadata requirements

Each entry must carry both the `x` and `y` fields. When only one value exists
per `(x, y)` pair, that value fills the cell; when multiple entries map to the
same cell, the **last** value wins (no aggregation). Entries missing either
field do not contribute a cell.

## Colorscales

Any Plotly colorscale works via `heatmap_colorscale`:

| Colorscale | Description |
|------------|-------------|
| `"Viridis"` | Perceptually uniform, colorblind-friendly |
| `"Plasma"` | Similar to Viridis, warmer |
| `"RdBu"` | Red-Blue diverging (good for centered data) |
| `"Blues"` | Single-hue blue |
| `"Hot"` | Black -> Red -> Yellow -> White |
| `"Greys"` | Grayscale |

## Text formatting

The `text_format` parameter uses Python's format specification:

| Format | Example | Description |
|--------|---------|-------------|
| `.2f` | `3.14` | 2 decimal places |
| `.1f` | `3.1` | 1 decimal place |
| `.0f` | `3` | No decimals |
| `.2e` | `3.14e+00` | Scientific notation |
| `.1%` | `31.4%` | Percentage |

NaN values display as empty strings.

## Handling missing data

By default, missing cells are `NaN` and appear as gaps (no color). To fill:

```python
plot = HeatmapPlot(
    z=FieldRef("stable_rank"),
    x=FieldRef("head_id"),
    y=FieldRef("layer_id"),
    fill_value=0.0,  # fill missing with zero
)
```

## Notes

- Only one value per `(x, y)` pair is stored (last wins).
- For multiple values per cell, precompute an aggregate field.
- The y-axis is reversed by default so row index 0 appears at the top.

## See also

- [ScatterPlot](scatter.md) — For relationships between two numeric fields
- [Styling](../styling.md) — Colorscale and palette details
