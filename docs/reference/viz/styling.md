# Styling

The styling system provides consistent, publication-ready styling for all
Plotly figures. The API lives in `diffract.viz.styling`; the core symbols
(`Theme`, `apply_theme`, and the presets) are also re-exported from
`diffract.viz`.

## Composed Theme

`Theme` is composed of nested style objects, each of which can be constructed
and overridden independently:

```python
from diffract.viz.styling import (
    Theme,
    LayoutStyle,
    TypographyStyle,
    BackgroundStyle,
    AxesStyle,
    LegendStyle,
    ColorbarStyle,
)
from diffract.viz.styling.palettes import PaletteBundle

theme = Theme(
    layout=LayoutStyle(width=800, height=400, margin={"l": 80, "r": 40, "t": 60, "b": 80}),
    typography=TypographyStyle(
        font_family="Times New Roman",
        title_font_size=16,
        label_font_size=14,
        tick_font_size=12,
    ),
    background=BackgroundStyle(plot_bgcolor="white", paper_bgcolor="white"),
    axes=AxesStyle(
        grid_color="lightgrey",
        line_color="black",
        show_grid=True,
        show_line=True,
        mirror=True,
    ),
    legend=LegendStyle(
        bgcolor="rgba(255,255,255,0.9)",
        border_color="gray",
        border_width=1,
        font_size=12,
    ),
    colorbar=ColorbarStyle(orientation="h", x=0.5, y=-0.15, thickness=15, len=0.5),
    palettes=PaletteBundle(),
)
```

Every nested component has sensible defaults, so you only construct the ones
you want to change (see [Creating custom themes](#creating-custom-themes)).

### Theme components

| Component | Fields |
|-----------|--------|
| `LayoutStyle` | `width`, `height`, `margin` |
| `TypographyStyle` | `font_family`, `title_font_size`, `label_font_size`, `tick_font_size` |
| `BackgroundStyle` | `plot_bgcolor`, `paper_bgcolor` |
| `AxesStyle` | `grid_color`, `line_color`, `show_grid`, `show_line`, `mirror` |
| `LegendStyle` | `bgcolor`, `border_color`, `border_width`, `font_size` |
| `ColorbarStyle` | `orientation`, `x`, `y`, `xanchor`, `yanchor`, `thickness`, `len` |
| `PaletteBundle` | `color`, `symbols`, `dashes` |

## Palettes

Colors, marker symbols, and line dashes come from a `PaletteBundle`
(`diffract.viz.styling.palettes`):

```python
from diffract.viz.styling.palettes import (
    PaletteBundle,
    DefaultColorPalette,
    DefaultSymbolPalette,
    DefaultDashPalette,
)

bundle = PaletteBundle(
    color=DefaultColorPalette(_colors=["#1f77b4", "#ff7f0e", "#2ca02c"]),
    symbols=DefaultSymbolPalette(),
    dashes=DefaultDashPalette(),
)

theme = Theme(palettes=bundle)
```

The default discrete colors are:

```python
["navy", "crimson", "green", "chocolate", "orange",
 "violet", "purple", "blue", "grey", "teal"]
```

Default marker symbols cycle through circle, square, diamond, cross, x,
triangle-up, and so on; default line dashes cycle through solid, dot, dash,
longdash, dashdot, and longdashdot.

Plots draw categorical colors/symbols/dashes from the active theme's palettes;
continuous color mapping uses a Plotly colorscale via each plot's coloraxis
fields (e.g. `marker_colorscale`, `heatmap_colorscale`).

## Predefined themes

### DEFAULT_THEME

Classic publication style: Times New Roman font, white background, visible
grid.

```python
from diffract.viz.styling import DEFAULT_THEME

fig = session.viz.box(y="stable_rank", x="model_id", theme=DEFAULT_THEME)
```

### DARK_THEME

Dark mode with dark backgrounds and a brighter color palette.

```python
from diffract.viz.styling import DARK_THEME

fig = session.viz.scatter(x="frob_norm", y="stable_rank", theme=DARK_THEME)
```

### MINIMAL_THEME

Clean, minimal style: Arial font, no grid lines, no axis lines, reduced
margins.

```python
from diffract.viz.styling import MINIMAL_THEME

fig = session.viz.violin(y="esd", x="model_id", theme=MINIMAL_THEME)
```

## Applying themes

### With convenience methods

```python
with session:
    fig = session.viz.box(y="stable_rank", x="model_id", theme=my_theme)
```

### With plot objects

Themes are passed at render time, not as a plot constructor field:

```python
from diffract.viz.data import FieldRef
from diffract.viz.plots.boxplot import BoxPlot
from diffract.viz.styling import DARK_THEME

plot = BoxPlot(y=FieldRef("stable_rank"), x=FieldRef("model_id"))
fig = session.viz.draw(plot=plot, theme=DARK_THEME)
# or directly:
fig = plot.render(session, theme=DARK_THEME)
```

### With config files

```python
with session:
    fig = session.viz.draw(
        config_path="plots/my_plot.yaml",
        theme_path="themes/publication.yaml",
    )
```

## Loading themes from YAML

`load_theme` reads a YAML file into a `Theme`. The structured format mirrors the
composed dataclasses:

```yaml
# themes/publication.yaml
layout:
  width: 1200
  height: 600
  margin:
    l: 100
    r: 50
    t: 80
    b: 100

typography:
  font_family: "Times New Roman"
  title_font_size: 18
  label_font_size: 14
  tick_font_size: 12

background:
  plot_bgcolor: "white"
  paper_bgcolor: "white"

axes:
  grid_color: "#e0e0e0"
  line_color: "black"
  show_grid: true
  show_line: true
  mirror: true

palettes:
  color:
    - "#1f77b4"
    - "#ff7f0e"
    - "#2ca02c"
    - "#d62728"
    - "#9467bd"
```

Load programmatically:

```python
from diffract.viz.renderer import load_theme

theme = load_theme("themes/publication.yaml")
fig = session.viz.box(y="stable_rank", x="model_id", theme=theme)
```

`load_theme` also accepts a flat layout (top-level keys such as `width`,
`font_family`, `grid_color`, `discrete_colormap`, ...), which it maps onto the
structured components.

## Creating custom themes

### Extend a predefined theme

Use `dataclasses.replace` to override individual **components**:

```python
from dataclasses import replace
from diffract.viz.styling import DEFAULT_THEME, LayoutStyle, TypographyStyle

my_theme = replace(
    DEFAULT_THEME,
    layout=LayoutStyle(width=1000, height=500),
    typography=replace(DEFAULT_THEME.typography, font_family="Arial"),
)
```

Note that `replace` swaps whole components — use a nested `replace` (as above)
to change a single field while keeping the rest of a component.

## apply_theme function

Apply a theme's figure-level styling to any Plotly figure:

```python
from diffract.viz.styling import apply_theme, DARK_THEME
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6])])
apply_theme(fig, DARK_THEME)
```

`apply_theme` applies layout (dimensions/margins), background, typography,
legend, and colorbar defaults. It does **not** override trace-specific settings
or the axes configured by plot configurators. It mutates the figure in place
and returns it for chaining; it is safe to call more than once.

## Theme component reference

### LayoutStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `width` | `int \| None` | `None` | Figure width in pixels |
| `height` | `int \| None` | `None` | Figure height in pixels |
| `margin` | `dict[str, int]` | `{"l": 80, "r": 40, "t": 60, "b": 80}` | Margins |

### TypographyStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `font_family` | `str` | `"Times New Roman"` | Font family |
| `title_font_size` | `int` | `16` | Title font size |
| `label_font_size` | `int` | `14` | Axis label font size |
| `tick_font_size` | `int` | `12` | Tick label font size |

### BackgroundStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `plot_bgcolor` | `str` | `"white"` | Plot area background |
| `paper_bgcolor` | `str` | `"white"` | Figure background |

### AxesStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `grid_color` | `str` | `"lightgrey"` | Grid line color |
| `line_color` | `str` | `"black"` | Axis line color |
| `show_grid` | `bool` | `True` | Show grid lines |
| `show_line` | `bool` | `True` | Show axis lines |
| `mirror` | `bool` | `True` | Mirror axes on the opposite side |

### LegendStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `bgcolor` | `str` | `"rgba(255,255,255,0.9)"` | Legend background |
| `border_color` | `str` | `"gray"` | Legend border color |
| `border_width` | `int` | `1` | Border width |
| `font_size` | `int` | `12` | Font size |

### ColorbarStyle

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `orientation` | `str` | `"h"` | `"h"` or `"v"` |
| `x` | `float` | `0.5` | X position |
| `y` | `float` | `-0.15` | Y position |
| `xanchor` | `str` | `"center"` | X anchor |
| `yanchor` | `str` | `"top"` | Y anchor |
| `thickness` | `int` | `15` | Thickness |
| `len` | `float` | `0.5` | Length |

### PaletteBundle

| Property | Type | Description |
|----------|------|-------------|
| `color` | `ColorPalette` | Discrete color palette |
| `symbols` | `SymbolPalette` | Marker symbol palette |
| `dashes` | `DashPalette` | Line dash palette |
