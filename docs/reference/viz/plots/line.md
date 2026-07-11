# SparklinePlot

Line/sparkline plot for visualizing a numeric field as a function of another
field.

Class: `diffract.viz.plots.sparkline.SparklinePlot` (also
`diffract.viz.plots.SparklinePlot`, and the alias `Sparkline`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.sparkline import SparklinePlot

with session:
    plot = SparklinePlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id"))
    fig = session.viz.draw(plot=plot)
```

Or using the convenience method (aliased as `session.viz.line`):

```python
with session:
    fig = session.viz.sparkline(y="stable_rank", x="layer_id")
```

## Example: Compare models across layers

```python
plot = SparklinePlot(
    y=FieldRef("frob_norm"),
    x=FieldRef("layer_id"),
    group_by=FieldRef("model_id"),
    line_color=FieldRef("model_id"),
    title="Frobenius Norm by Layer",
)
```

## Example: With error bands

When multiple entries share the same `(group_by, x)` pair, the plot shows
mean +/- std as a shaded band (`show_bands=True` by default):

```python
# If multiple heads per layer, this shows mean across heads with a std band
plot = SparklinePlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("layer_id"),
    group_by=FieldRef("model_id"),
)
```

## Example: Different line styles

```python
plot = SparklinePlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("layer_id"),
    group_by=FieldRef("model_id"),
    line_color=FieldRef("model_id"),
    line_dash=FieldRef("kind"),      # different dash patterns per kind
    marker_symbol=FieldRef("kind"),  # different markers per kind
)
```

## Parameters

### Fields (required, keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `FieldRef` | Field for Y-axis values |
| `x` | `FieldRef` | Field for X-axis values |

### Grouping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_by` | `FieldRef \| None` | `None` | Field to split into separate line traces |

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
| `title` | `str \| None` | `None` | Figure title (defaults to `"{y.field} by {x.field}"`) |
| `x_title` | `str \| None` | `None` | X-axis title |
| `y_title` | `str \| None` | `None` | Y-axis title |

### Mode and bands

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `"lines" \| "markers" \| "lines+markers"` | `"lines"` | Plotly draw mode |
| `show_bands` | `bool` | `True` | Draw mean +/- std bands |
| `band_opacity` | `float` | `0.3` | Band fill opacity |
| `band_line_width` | `float` | `0.5` | Band edge line width |

### Line styling (from `SupportsLine`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_color` | `FieldRef \| str \| None` | `None` | Line color (constant or by field) |
| `line_dash` | `FieldRef \| str \| None` | `None` | Line dash pattern |
| `line_width` | `FieldRef \| float \| None` | `2` | Line width |

### Marker styling (from `SupportsMarker`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_color` | `FieldRef \| str \| None` | `None` | Marker color |
| `marker_symbol` | `FieldRef \| str \| None` | `None` | Marker symbol |
| `marker_size` | `FieldRef \| float \| None` | `6` | Marker size |
| `marker_opacity` | `FieldRef \| float \| None` | `0.7` | Marker opacity |

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
  _target_: diffract.viz.plots.sparkline.SparklinePlot
  y: stable_rank
  x: layer_id
  group_by: model_id
  line_color: model_id
  mode: "lines+markers"
  title: "Stable Rank by Layer"
```

## How it works

1. Fetches data via `DataProvider.fetch([...], ...)`.
2. Groups entries by the `group_by` field (one line per group).
3. Within each group, computes mean and std per unique x value.
4. Orders x values by the `x` `FieldRef` ordering.
5. Creates a `go.Scatter` trace for the mean line.
6. Adds fill-between band traces for mean +/- std when `show_bands=True` and
   any std is nonzero.
7. Applies line/marker mappings and the theme when one is passed to `render`.

## Aggregation within groups

When multiple entries share the same `(group_by, x)` combination:

```
Model A:
  layer_id=0: [head0=45.2, head1=43.1, head2=47.3]  -> mean=45.2, std=2.1
  layer_id=1: [head0=38.5, head1=40.2, head2=39.1]  -> mean=39.3, std=0.8

Result: Line with mean values, shaded band showing +/-1 std
```

## Mode options

| Mode | Description |
|------|-------------|
| `"lines"` | Lines only (default) |
| `"markers"` | Points only |
| `"lines+markers"` | Lines with points |

If `marker_symbol` is set and `mode` does not include markers, markers are
added automatically.

## Dash patterns

Available dash patterns (used by `line_dash` as a constant, or supplied by the
theme's dash palette when mapping from a field):

- `"solid"` — Solid line
- `"dot"` — Dotted
- `"dash"` — Dashed
- `"dashdot"` — Dash-dot
- `"longdash"` — Long dashes
- `"longdashdot"` — Long dash-dot

## Common use cases

### Layer-wise analysis

```python
plot = SparklinePlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("layer_id"),
    group_by=FieldRef("kind"),  # separate lines for attn, ffn
    line_color=FieldRef("kind"),
)
```

### Cross-model comparison

```python
plot = SparklinePlot(
    y=FieldRef("effective_rank"),
    x=FieldRef("layer_id"),
    group_by=FieldRef("model_id"),
    line_color=FieldRef("model_id"),
    line_dash=FieldRef("model_id"),
)
```

## Notes

- X values are ordered according to the `x` `FieldRef` ordering before plotting.
- Error bands use fill-between with reduced opacity (`band_opacity`).
- The legend shows one entry per group (not per band).

## See also

- [ScatterPlot](scatter.md) — For X/Y scatter of two numeric fields
- [BoxPlot](boxplot.md) — For distributions at each X value
