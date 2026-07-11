# BoxPlot

Box plot for visualizing scalar or vector field distributions grouped by a
categorical field.

Class: `diffract.viz.plots.boxplot.BoxPlot` (also `diffract.viz.plots.BoxPlot`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot

with session:
    plot = BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
    fig = session.viz.draw(plot=plot)
```

Or using the convenience method (plain strings are wrapped in `FieldRef`):

```python
with session:
    fig = session.viz.box(y="stable_rank", x="model_id")
```

## Example: Color markers by layer

Coloring is applied to the marker points (the `boxpoints`) via `marker_color`:

```python
plot = BoxPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id"),
    boxpoints="all",
    marker_color=FieldRef("layer_id"),
    title="Stable Rank by Model (points colored by layer)",
)
```

## Example: With jitter overlay

```python
plot = BoxPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id"),
    jitter_enabled=True,
    jitter_color=FieldRef("layer_id"),
    jitter_showscale=True,
)
```

## Example: Custom category order

Ordering is expressed on the `FieldRef` for the categorical `x` axis:

```python
from diffract.viz.data import FieldRef, custom

plot = BoxPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id", ordering=custom(["baseline", "finetuned", "pruned"])),
)
```

## Example: With value filtering

```python
plot = BoxPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id"),
    value_filter={
        "stable_rank": (">", 10.0),
        "frob_norm": ("<=", 100.0),
    },
)
```

## Parameters

### Fields (required, keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `FieldRef` | Field for Y-axis values (scalar or vector) |
| `x` | `FieldRef` | Categorical field for X-axis grouping |

### Rescaling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_rescale_range` | `tuple[float, float] \| None` | `None` | Rescale y-values into this range |
| `y_rescale_traces_separately` | `bool` | `False` | Rescale each trace independently |

### Box styling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box_width` | `float` | `0.5` | Box width |
| `boxpoints` | `"all" \| "outliers" \| False` | `"outliers"` | Which points to show |

### Titles and axes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `None` | Figure title (defaults to `"{y.field} by {x.field}"`) |
| `x_title` | `str \| None` | `None` | X-axis title |
| `y_title` | `str \| None` | `None` | Y-axis title |
| `x_categoryorder` | `str \| None` | `None` | Plotly category order |
| `x_categoryarray` | `list[str] \| None` | `None` | Explicit category array |

(Plus the common axis fields: `x_tickangle`, `x_showgrid`, `y_range`, `y_dtick`,
`y_tickformat`, etc.)

### Marker styling (from `SupportsMarker`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_color` | `FieldRef \| str \| None` | `None` | Point color (constant or by field) |
| `marker_symbol` | `FieldRef \| str \| None` | `None` | Point symbol |
| `marker_size` | `FieldRef \| float \| None` | `6` | Point size |
| `marker_opacity` | `FieldRef \| float \| None` | `0.7` | Point opacity |

Continuous `marker_color` mapping uses a coloraxis; control it with
`marker_colorscale`, `marker_cmin`, `marker_cmax`, `marker_showscale`,
`marker_colorbar_title`.

### Jitter overlay (from `SupportsJitter`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jitter_enabled` | `bool` | `False` | Enable jitter overlay |
| `jitter_width` | `float` | `0.12` | Maximum jitter spread |
| `jitter_offset` | `float` | `-0.35` | Horizontal offset from box center |
| `jitter_seed` | `int` | `42` | Random seed |
| `jitter_density_scale` | `bool` | `True` | Scale jitter by local density |
| `jitter_color` | `FieldRef \| str \| None` | `None` | Color for jitter points |

The jitter overlay also has its own coloraxis fields
(`jitter_colorscale`, `jitter_showscale`, `jitter_cmin`, `jitter_cmax`, ...).
Size, opacity and symbol are inherited from the parent marker.

### Value filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | `None` | Filter conditions |

### Theming

Themes are not a constructor field. Pass a `Theme` when rendering:

```python
from diffract.viz.styling import DARK_THEME

fig = session.viz.draw(plot=plot, theme=DARK_THEME)
# or
fig = plot.render(session, theme=DARK_THEME)
```

## YAML configuration

```yaml
plot:
  _target_: diffract.viz.plots.boxplot.BoxPlot
  y: stable_rank
  x: model_id
  title: "Stable Rank Distribution"
  boxpoints: all
  marker_color: layer_id
  jitter_enabled: true
  jitter_color: layer_id
```

String values annotated as `FieldRef` are coerced automatically by the
renderer.

With an `UpdateFigure` wrapper:

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  plot:
    _target_: diffract.viz.plots.boxplot.BoxPlot
    y: stable_rank
    x: model_id
  layout:
    width: 800
    height: 400
```

## How it works

1. Fetches data via `DataProvider.fetch([...], value_filter=...)`.
2. Groups y-values by the `x` category (vector `y` values are exploded into
   individual observations).
3. Orders categories according to the `x` `FieldRef` ordering.
4. Creates a `go.Box` trace per category.
5. Applies marker styling and, if `jitter_enabled`, a scatter overlay.
6. Applies the theme when one is passed to `render`.

## Notes

- Vector `y` fields (e.g. an ESD spectrum) are flattened into per-element
  observations within each category.
- The `boxpoints` parameter controls Plotly's built-in point display; use
  `jitter_enabled=True` for a separate, customizable point overlay.
- `jitter_offset` shifts points away from the box for clarity.

## See also

- [ViolinPlot](violin.md) — Similar but with density visualization
- [ScatterPlot](scatter.md) — For plotting two fields against each other
- [Jitter utilities](../jitter.md) — How jitter works
