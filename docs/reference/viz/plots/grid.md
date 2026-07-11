# GridPlot

Compose multiple plots into a grid layout with independent or shared axes.

Class: `diffract.viz.plots.subplots.GridPlot` (also
`diffract.viz.plots.GridPlot`).

## Basic usage

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.subplots import GridPlot, SubplotSpec

with session:
    grid = GridPlot(
        subplots=[
            SubplotSpec(
                row=1, col=1, title="Stable Rank",
                plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
            ),
            SubplotSpec(
                row=1, col=2, title="Frob Norm",
                plot=BoxPlot(y=FieldRef("frob_norm"), x=FieldRef("model_id")),
            ),
        ],
        make_subplots_kwargs={"shared_yaxes": False},
    )
    fig = session.viz.draw(plot=grid)
```

## Example: Side-by-side model comparison

```python
grid = GridPlot(
    subplots=[
        SubplotSpec(
            row=1, col=1,
            title="Model A",
            plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id")),
            session_filter={"model_ids": ["model_a"]},
        ),
        SubplotSpec(
            row=1, col=2,
            title="Model B",
            plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id")),
            session_filter={"model_ids": ["model_b"]},
        ),
    ],
    make_subplots_kwargs={"shared_yaxes": True},
)
```

## Example: 2x2 grid with mixed plot types

```python
from diffract.viz.plots.scatter import ScatterPlot
from diffract.viz.plots.violin import ViolinPlot

grid = GridPlot(
    subplots=[
        SubplotSpec(
            row=1, col=1, title="Stable Rank",
            plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
        ),
        SubplotSpec(
            row=1, col=2, title="Frob Norm",
            plot=BoxPlot(y=FieldRef("frob_norm"), x=FieldRef("model_id")),
        ),
        SubplotSpec(
            row=2, col=1, title="Scatter",
            plot=ScatterPlot(x=FieldRef("frob_norm"), y=FieldRef("stable_rank")),
        ),
        SubplotSpec(
            row=2, col=2, title="ESD",
            plot=ViolinPlot(y=FieldRef("esd"), x=FieldRef("model_id")),
        ),
    ],
    make_subplots_kwargs={
        "horizontal_spacing": 0.08,
        "vertical_spacing": 0.12,
    },
)
```

## SubplotSpec

Specification for a single subplot
(`diffract.viz.plots.subplots.SubplotSpec`):

```python
from diffract.viz.plots.subplots import SubplotSpec

spec = SubplotSpec(
    row=1,                 # row position (1-indexed)
    col=2,                 # column position (1-indexed)
    title="My Plot",       # subplot title
    plot=BoxPlot(...),     # any Plot object
    session_filter=None,   # optional per-subplot session filter
    value_filter=None,     # optional per-subplot value filter
)
```

`SubplotSpec` fields: `row`, `col`, `title`, `plot`, `session_filter`,
`value_filter`, and `filter` (a backward-compatible alias for
`session_filter`).

### Per-subplot session filtering

Each subplot can filter session data independently via `session_filter`
(forwarded to `Session.filter(...)`):

```python
SubplotSpec(
    row=1, col=1,
    title="Attention Layers",
    plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id")),
    session_filter={
        "model_ids": ["model_a"],
        "param_types": ["attn"],
    },
)
```

Valid `session_filter` keys:

| Key | Description |
|-----|-------------|
| `param_ids` | List of parameter UIDs |
| `param_names` | List of parameter names |
| `param_types` | List of parameter types |
| `model_ids` | List of model IDs |

### Per-subplot value filtering

`value_filter` is merged into the child plot's `value_filter` for that subplot
only:

```python
SubplotSpec(
    row=1, col=1,
    title="Filtered",
    plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
    value_filter={"stable_rank": (">", 10.0)},
)
```

## GridPlot parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `subplots` | `list[SubplotSpec]` | List of subplot specifications |
| `make_subplots_kwargs` | `dict[str, Any]` | Arguments for `plotly.subplots.make_subplots` |

`GridPlot` has no `theme` field. Pass a `Theme` when rendering:

```python
from diffract.viz.styling import DARK_THEME
fig = session.viz.draw(plot=grid, theme=DARK_THEME)
```

## make_subplots_kwargs

Common options for `plotly.subplots.make_subplots` (the `rows`, `cols`, and
`subplot_titles` keys are ignored — they are derived from the specs):

| Option | Type | Description |
|--------|------|-------------|
| `shared_xaxes` | `bool \| str` | Share X axes (`True`, `"all"`, `"columns"`) |
| `shared_yaxes` | `bool \| str` | Share Y axes (`True`, `"all"`, `"rows"`) |
| `horizontal_spacing` | `float` | Horizontal space between subplots |
| `vertical_spacing` | `float` | Vertical space between subplots |
| `column_widths` | `list[float]` | Relative widths of columns |
| `row_heights` | `list[float]` | Relative heights of rows |

```python
make_subplots_kwargs={
    "shared_yaxes": True,
    "horizontal_spacing": 0.05,
    "vertical_spacing": 0.1,
    "column_widths": [0.6, 0.4],  # first column wider
}
```

## YAML configuration

```yaml
plot:
  _target_: diffract.viz.plots.subplots.GridPlot
  make_subplots_kwargs:
    shared_yaxes: true
    horizontal_spacing: 0.08
  subplots:
    - _target_: diffract.viz.plots.subplots.SubplotSpec
      row: 1
      col: 1
      title: "Model A"
      plot:
        _target_: diffract.viz.plots.boxplot.BoxPlot
        y: stable_rank
        x: layer_id
      session_filter:
        model_ids:
          - model_a
    - _target_: diffract.viz.plots.subplots.SubplotSpec
      row: 1
      col: 2
      title: "Model B"
      plot:
        _target_: diffract.viz.plots.boxplot.BoxPlot
        y: stable_rank
        x: layer_id
      session_filter:
        model_ids:
          - model_b
```

## Parametric grids

Instead of listing every `SubplotSpec` by hand, a grid can be generated from a
plot template plus row/column axis binds. The building blocks live in
`diffract.viz.plots.subplots`:

- `build_bound_grid(*, plot_template, row=None, col=None, cell_rules=None,
  title_template=None, make_subplots_kwargs=None, theme=None,
  base_session_filter=None, base_value_filter=None)` returns a `GridPlot`
  with auto-generated `SubplotSpec` entries.
- `GridAxisBind(target=..., key=..., values=..., labels=None, op=None)` maps a
  sequence of values onto rows or columns. `target` selects what each value
  overrides: `"plot"` (a plot attribute), `"session_filter"` (a
  `SubplotSpec.session_filter` key), or `"value_filter"` (a
  `SubplotSpec.value_filter` key; `op` turns bare thresholds into
  `(op, threshold)` conditions).
- `GridCellRule(selector=..., plot=None, plot_format=None, session_filter=None,
  value_filter=None)` applies extra overrides to selected cells; its
  `CellSelector` targets `"all"`, `"first_row"`, `"last_row"`, `"first_col"`,
  `"last_col"`, or explicit `row`/`col` indices.

At least one axis bind (`row` or `col`) is required. For each cell the plot
template is deep-copied, the row and column binds are applied, matching
`cell_rules` are applied in order (last write wins), and a `SubplotSpec` is
created.

The session wrapper `session.viz.bound_grid(...)` takes the same arguments
(plus `theme`/`theme_path`) and renders the grid in one call. For `plot` binds
targeting `FieldRef` plot attributes, it converts plain string bind values into
`FieldRef` objects automatically:

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.plots.subplots import GridAxisBind

with session:
    fig = session.viz.bound_grid(
        plot_template=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
        row=GridAxisBind(
            target="plot",
            key="y",
            values=["stable_rank", "frob_norm"],
            labels=["stable_rank", "frob_norm"],
        ),
        col=GridAxisBind(
            target="session_filter",
            key="model_ids",
            values=[["baseline"], ["finetuned"]],
            labels=["baseline", "finetuned"],
        ),
        make_subplots_kwargs={"shared_yaxes": "rows"},
    )
```

This produces a 2×2 grid — one metric per row, one model per column — with
titles like `"stable_rank | baseline"` (the default `title_template` is
`"{row_label} | {col_label}"` when both binds are present).

The repository contains worked YAML examples of bound grids under
`notebooks/configs/plots/`: the `grid_*.yaml` configs, the plot templates in
`templates/`, and the reusable `FieldRef` definitions in `refs/`.

## How it works

1. Determines grid dimensions from the max row/col across specs.
2. Creates a Plotly subplots grid via `make_subplots()`.
3. For each `SubplotSpec`:
   - Applies `session_filter` if specified (creates a filtered session view).
   - Merges `value_filter` into the child plot for that subplot.
   - Renders the child plot.
   - Remaps its coloraxis to avoid conflicts across subplots.
   - Adds traces to the grid at the specified position.
4. Deduplicates legend entries.
5. Applies the theme (when passed to `render`) to the whole figure.

## Coloraxis handling

Each subplot's coloraxis is automatically remapped to a unique name, preventing
color conflicts when multiple subplots use continuous color mapping (e.g.,
jitter coloring).

## Notes

- Row/column indices are 1-based, matching the Plotly `make_subplots` convention.
- Grid dimensions are computed automatically from the specs.
- `make_subplots_kwargs` may include `rows`/`cols`, but they are overridden.
- Each child plot is rendered independently, then composed.

## Common patterns

### Compare models side-by-side

```python
models = ["baseline", "finetuned", "pruned"]
subplots = [
    SubplotSpec(
        row=1, col=i + 1, title=model,
        plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id")),
        session_filter={"model_ids": [model]},
    )
    for i, model in enumerate(models)
]
grid = GridPlot(subplots=subplots, make_subplots_kwargs={"shared_yaxes": True})
```

### Dashboard-style layout

```python
from diffract.viz.plots.sparkline import SparklinePlot
from diffract.viz.plots.heatmap import HeatmapPlot

grid = GridPlot(
    subplots=[
        SubplotSpec(
            row=1, col=1, title="Overview",
            plot=BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id")),
        ),
        SubplotSpec(
            row=1, col=2, title="Scatter",
            plot=ScatterPlot(x=FieldRef("frob_norm"), y=FieldRef("stable_rank")),
        ),
        SubplotSpec(
            row=2, col=1, title="By Layer",
            plot=SparklinePlot(y=FieldRef("stable_rank"), x=FieldRef("layer_id")),
        ),
        SubplotSpec(
            row=2, col=2, title="Heatmap",
            plot=HeatmapPlot(
                z=FieldRef("stable_rank"),
                x=FieldRef("head_id"),
                y=FieldRef("layer_id"),
            ),
        ),
    ],
    make_subplots_kwargs={"horizontal_spacing": 0.1, "vertical_spacing": 0.15},
)
```

## See also

- [Plot Types](index.md) — Overview and the `UpdateFigure` wrapper
- [Styling](../styling.md) — Grid-wide theming
