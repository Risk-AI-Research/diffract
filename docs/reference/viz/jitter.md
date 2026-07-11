# Jitter Utilities

The `diffract.viz.plots.base.jitter` module provides the density-scaled jitter
overlay used by box and violin plots.

## Overview

A jitter overlay scatters individual data points on top of an aggregated plot
(boxes, violins). This helps visualize the underlying distribution, especially
for small datasets. Jitter is enabled per-plot through the `SupportsJitter`
mixin fields — there is no separate config object.

```python
with session:
    fig = session.viz.box(
        y="stable_rank",
        x="model_id",
        jitter_enabled=True,
        jitter_color="layer_id",
    )
```

## SupportsJitter mixin

`SupportsJitter` (a mixin of `BoxPlot` and `ViolinPlot`) contributes these
fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jitter_enabled` | `bool` | `False` | Enable the jitter overlay |
| `jitter_width` | `float` | `0.12` | Maximum horizontal spread |
| `jitter_offset` | `float` | `-0.35` | Horizontal offset from the category center |
| `jitter_seed` | `int` | `42` | Random seed for reproducibility |
| `jitter_density_scale` | `bool` | `True` | Scale spread by local point density |
| `jitter_color` | `FieldRef \| str \| None` | `None` | Color source for jitter points |

Because `SupportsJitter` also mixes in a coloraxis, continuous `jitter_color`
mapping is controlled by the `jitter_colorscale`, `jitter_showscale`,
`jitter_cmin`, `jitter_cmax`, `jitter_coloraxis_id`, and
`jitter_colorbar_title` fields.

Marker **size, opacity, and symbol** of the jitter points are inherited from
the parent plot's `marker_size`, `marker_opacity`, and `marker_symbol`. The
jitter color falls back to `marker_color` when `jitter_color` is unset.

## density_scaled_jitter

The standalone `density_scaled_jitter` helper scales base jitter values so that
points in denser regions get a wider spread:

```python
import numpy as np
from diffract.viz.plots.base import density_scaled_jitter

y = np.array([1.0, 1.0, 1.0, 2.0, 3.0])  # clustered around 1.0
rng = np.random.default_rng(42)
base_jitter = rng.uniform(-0.12, 0.12, size=y.size)

scaled = density_scaled_jitter(y, base_jitter, n_bins=20)
```

Signature:

```python
def density_scaled_jitter(
    y: np.ndarray,
    jitter: np.ndarray,
    *,
    n_bins: int = 20,
) -> np.ndarray:
    ...
```

Algorithm:

1. Bin the y-values into `n_bins` equal-width bins between `min(y)` and
   `max(y)`.
2. Count how many points fall in each bin.
3. Scale each point's jitter by `count(bin) / max_count`.

Points in sparse bins get `scale` near 0 (minimal jitter); points in the
densest bin get `scale = 1` (full jitter). Empty or single-value inputs are
returned unchanged.

## Usage in BoxPlot

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot

plot = BoxPlot(
    y=FieldRef("stable_rank"),
    x=FieldRef("model_id"),

    # Enable jitter
    jitter_enabled=True,

    # Jitter configuration
    jitter_width=0.12,
    jitter_offset=-0.35,
    jitter_seed=42,
    jitter_density_scale=True,

    # Color jitter points by a field
    jitter_color=FieldRef("layer_id"),
    jitter_colorscale="Viridis",
    jitter_showscale=True,
)
```

## Usage in ViolinPlot

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.violin import ViolinPlot

plot = ViolinPlot(
    y=FieldRef("esd"),
    x=FieldRef("model_id"),

    jitter_enabled=True,
    jitter_width=0.12,
    jitter_offset=-0.35,
    jitter_color=FieldRef("head_id"),
    jitter_density_scale=True,
)
```

## Jitter with vector fields

When `y` is a vector field and jitter is enabled, each array is expanded into
individual observations (matching the box/violin expansion). A vector
`jitter_color` field is flattened the same way so each observation gets its own
color; a scalar color field applies one color per parameter.
