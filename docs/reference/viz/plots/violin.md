# ViolinPlot

Violin plot for visualizing distributions with kernel density estimation.

Class: `diffract.viz.plots.violin.ViolinPlot` (also
`diffract.viz.plots.ViolinPlot`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.violin import ViolinPlot

with session:
    plot = ViolinPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
    fig = session.viz.draw(plot=plot)
```

Or using the convenience method:

```python
with session:
    fig = session.viz.violin(y="stable_rank", x="model_id")
```

## Key feature: Vector-field expansion

Like `BoxPlot`, `ViolinPlot` handles vector (array-like) fields by expanding
each array into individual observations:

```python
# For a field like ESD (eigenvalue spectral density) with one array per entry
plot = ViolinPlot(
    y=FieldRef("esd"),  # each parameter has an array of eigenvalues
    x=FieldRef("model_id"),
)
# All eigenvalues from all parameters are combined per group
```

## Example: With jitter overlay

```python
plot = ViolinPlot(
    y=FieldRef("esd"),
    x=FieldRef("model_id"),
    jitter_enabled=True,
    jitter_color=FieldRef("layer_id"),
    jitter_showscale=True,
)
```

## Example: Half violin (one-sided)

```python
plot = ViolinPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id"),
    side="positive",  # show only the right side
    box_visible=True,
    meanline_visible=True,
)
```

## Example: Custom bandwidth

```python
plot = ViolinPlot(
    y=FieldRef("esd"),
    x=FieldRef("model_id"),
    bandwidth=0.5,  # override automatic bandwidth
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

### Titles and axes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `None` | Figure title (defaults to `"{y.field} by {x.field}"`) |
| `x_title` | `str \| None` | `None` | X-axis title |
| `y_title` | `str \| None` | `None` | Y-axis title |
| `x_categoryorder` | `str \| None` | `None` | Plotly category order |
| `x_categoryarray` | `list[str] \| None` | `None` | Explicit category array |

### Visual options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `"all" \| "outliers" \| False` | `"outliers"` | Built-in point display |
| `box_visible` | `bool` | `True` | Show box inside violin |
| `meanline_visible` | `bool` | `True` | Show mean line |
| `side` | `"positive" \| "negative" \| "both"` | `"positive"` | Which sides to draw |
| `bandwidth` | `float \| None` | `None` | KDE bandwidth (`None` = auto) |

### Marker styling (from `SupportsMarker`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker_color` | `FieldRef \| str \| None` | `None` | Point color |
| `marker_symbol` | `FieldRef \| str \| None` | `None` | Point symbol |
| `marker_size` | `FieldRef \| float \| None` | `6` | Point size |
| `marker_opacity` | `FieldRef \| float \| None` | `0.7` | Point opacity |

### Jitter overlay (from `SupportsJitter`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jitter_enabled` | `bool` | `False` | Enable jitter overlay |
| `jitter_width` | `float` | `0.12` | Maximum jitter spread |
| `jitter_offset` | `float` | `-0.35` | Horizontal offset |
| `jitter_seed` | `int` | `42` | Random seed |
| `jitter_density_scale` | `bool` | `True` | Scale by local density |
| `jitter_color` | `FieldRef \| str \| None` | `None` | Color for jitter points |

### Value filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | `None` | Filter conditions |

### Theming

Pass a `Theme` at render time, not as a constructor field:

```python
from diffract.viz.styling import MINIMAL_THEME
fig = session.viz.draw(plot=plot, theme=MINIMAL_THEME)
```

## YAML configuration

```yaml
plot:
  _target_: diffract.viz.plots.violin.ViolinPlot
  y: esd
  x: model_id
  title: "ESD Distribution"
  side: positive
  box_visible: true
  meanline_visible: true
  jitter_enabled: true
  jitter_color: layer_id
```

## How it works

1. Fetches data via `DataProvider.fetch([...], ...)`.
2. Groups y-values by the `x` category; vector `y` values are expanded into
   individual observations.
3. Creates a `go.Violin` trace per category.
4. Applies marker styling and, if `jitter_enabled`, a scatter overlay.
5. Applies the theme when one is passed to `render`.

### Vector handling

For each parameter entry:

```python
# Scalar field: stable_rank = 45.2
observations = [45.2]  # single observation

# Vector field: esd = [0.1, 0.5, 1.2, 2.3]
observations = [0.1, 0.5, 1.2, 2.3]  # all values become observations
```

All observations from all parameters in a group are combined for the violin.

## Side options

| Side | Description |
|------|-------------|
| `"positive"` | Right half only (default) |
| `"negative"` | Left half only |
| `"both"` | Full violin (both sides) |

One-sided violins are useful when comparing two groups side-by-side.

## Bandwidth

The bandwidth controls KDE smoothness:

- `None` (default): Plotly auto-selects
- Low value (e.g., 0.1): more detail, potentially noisy
- High value (e.g., 1.0): smoother, less detail

## Notes

- For scalar-only fields, `ViolinPlot` behaves like `BoxPlot` with KDE.
- `points="outliers"` uses Plotly's built-in outlier detection.
- Use `jitter_enabled=True` for full control over point display.
- `side="positive"` is the default because it pairs well with the jitter offset.

## See also

- [BoxPlot](boxplot.md) — Simpler box plot without KDE
- [Jitter utilities](../jitter.md) — How jitter works
