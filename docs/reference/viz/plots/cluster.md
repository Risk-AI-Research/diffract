# ClusterBarChart

Clustered line chart for array-like fields (e.g. singular value spectra). For
each parameter, the array field is binned into `num_bins` histogram counts over
the global value range; parameters are grouped by the `group_by` metadata keys,
aggregated (mean, and std when `draw_statistics` is set) across the parameters
in each group, and drawn as one lines+markers trace per group.

Class: `diffract.viz.plots.cluster.ClusterBarChart` (also
`diffract.viz.plots.ClusterBarChart`).

## Basic usage

```python
from diffract.viz.plots.cluster import ClusterBarChart

with session:
    plot = ClusterBarChart(field="esd", group_by=["model_id"])
    fig = session.viz.draw(plot=plot)
```

Unlike most plots, `ClusterBarChart` takes its field as a plain string
(`field="esd"`), not a `FieldRef` — the field must resolve to array-like
values, and `group_by` / `color_by` / `dash_by` / `marker_by` name metadata
keys directly.

## Example: Compare spectra across models

```python
from diffract.viz.plots.cluster import ClusterBarChart

with session:
    fig = session.viz.draw(
        plot=ClusterBarChart(
            field="esd",
            group_by=["model_id"],
            num_bins=30,
            binning="exponential",
            color_by="model_id",
            legend_format="{model_id}",
        )
    )
```

## Parameters

### Field (required, keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `field` | `str` | Array-like field to bin (e.g. `esd`, `weights_svals`) |

### Grouping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_by` | `list[str]` | `["model_id"]` | Metadata keys that define one trace per distinct combination |
| `aggregate_by` | `str \| None` | `None` | Extra key whose values are averaged out within each group |

### Parameter/model filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameter_uids` | `list[str] \| None` | `None` | Restrict to these parameter UIDs |
| `parameter_names` | `list[str] \| None` | `None` | Restrict to these parameter names (`re:` prefix for regex) |
| `parameter_types` | `list[str] \| None` | `None` | Restrict to these parameter types |
| `model_ids` | `list[str] \| None` | `None` | Restrict to these model IDs |

These are applied via `Session.filter(...)` before fetching data, so the plot
carries its own filtering (useful in YAML configs).

### Binning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_bins` | `int` | `20` | Number of histogram bins |
| `binning` | `"linear" \| "exponential"` | `"exponential"` | Bin edge spacing |
| `left_bound` | `float \| None` | `None` | Left edge of the binning range (defaults to the data minimum) |
| `right_bound` | `float \| None` | `None` | Right edge of the binning range (defaults to the data maximum) |

### Statistics and trace styling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `draw_statistics` | `bool` | `False` | Draw per-group std below zero as a dashed trace |
| `mode` | `"lines" \| "markers" \| "lines+markers"` | `"lines+markers"` | Plotly trace mode |
| `color_by` | `str \| None` | `None` | Metadata key mapped to line color |
| `dash_by` | `str \| None` | `None` | Metadata key mapped to line dash |
| `marker_by` | `str \| None` | `None` | Metadata key mapped to marker symbol |

### Legend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `legend_format` | `str \| None` | `None` | Format string over group keys, e.g. `"{model_id}"` |
| `legend_keys` | `list[str] \| None` | `None` | Subset of group keys to show as `key=value` pairs |

If neither is set, legend entries list all group keys as `key=value` pairs.

### Common fields (from `Plot`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str \| None` | `None` | Figure title (defaults to `"Cluster bar chart ({field})"`) |
| `value_filter` | `dict[str, tuple[str, Any]] \| None` | `None` | Filter conditions |

### Theming

Themes are not a constructor field. Pass a `Theme` when rendering:

```python
from diffract.viz.styling import DARK_THEME

fig = session.viz.draw(plot=plot, theme=DARK_THEME)
```

## YAML configuration

Adapted from `notebooks/configs/plots/cluster_bar_charts/compare_checkpoints.yaml`:

```yaml
plot:
  _target_: diffract.viz.plots.base.UpdateFigure
  layout:
    width: 1000
    height: 480
  yaxes:
    title: "count per bin (mean over heads)"
    type: log
    dtick: 1
  plot:
    _target_: diffract.viz.plots.cluster.ClusterBarChart
    field: weights_svals
    title: "Singular value spectrum: 16k vs 128k pre-train steps"
    parameter_names:
      - re:.*(q|k|v|o)_proj_head.*
    group_by: [model_id]
    color_by: model_id
    dash_by: model_id
    legend_format: "{model_id}"
    num_bins: 40
    left_bound: 0.0
    binning: linear
    draw_statistics: false
```

## How it works

1. Applies the plot-level parameter/model filter (if any) via `Session.filter`.
2. Fetches the array field for every remaining parameter.
3. Computes shared bin edges over the global value range (`np.linspace` for
   `linear`, `np.geomspace` for `exponential`).
4. Bins each parameter's array into histogram counts.
5. Groups counts by the `group_by` keys (minus `aggregate_by`) and averages
   them within each group.
6. Adds one `go.Scatter` trace per group at the numeric bin centers
   (geometric means of the edges for `exponential` binning).

## Notes

- Bin centers are numeric, so the x axis stays quantitative — a `type: log`
  axis (e.g. via `UpdateFigure`) positions the bins honestly.
- With `draw_statistics: true`, each group's std is drawn as a dashed trace
  mirrored below zero, with a horizontal reference line at zero.
- Entries missing any `group_by` key are skipped.

## See also

- [Plot Types](index.md) — Overview and the `UpdateFigure` wrapper
- [Line/Sparkline](line.md) — A scalar field vs another field
