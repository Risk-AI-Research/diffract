# Visualization Reference

Comprehensive API reference for the `diffract.viz` module.

## Overview

The visualization system is built around several key concepts:

- **Plot classes**: Dataclass-based configurations for each plot type
- **DataProvider**: Unified data fetching from a session
- **Styling**: Composable, publication-ready themes
- **Renderer**: Hydra-configurable rendering pipeline

## Module structure

```
diffract.viz
├── renderer.py          # render(), render_from_config(), load_theme()
├── data/
│   ├── extraction.py    # get_field_value(), get_field_values(), get_field_data()
│   ├── detection.py     # detect_data_type(), detect_data_shape()
│   ├── filtering.py     # apply_filter()
│   ├── ordering.py      # Ordering, OrderMode, as_is/lexicographic/numeric/...
│   ├── provider.py      # DataProvider
│   └── types.py         # FieldRef, Entry, DataType, DataShape
├── styling/
│   ├── theme/           # Theme + AxesStyle/LayoutStyle/... and presets
│   ├── palettes/        # PaletteBundle, ColorPalette, SymbolPalette, DashPalette
│   ├── resolvers/       # ColorResolver, CategoricalPropertyResolver, ...
│   └── sources.py       # ColorSource/SymbolSource/DashSource annotations
└── plots/
    ├── boxplot.py       # BoxPlot
    ├── violin.py        # ViolinPlot
    ├── scatter.py       # ScatterPlot
    ├── heatmap.py       # HeatmapPlot
    ├── sparkline.py     # SparklinePlot (alias: Sparkline)
    ├── cluster.py       # ClusterBarChart
    ├── subplots/        # GridPlot, SubplotSpec
    └── base/            # Plot, mixins (SupportsMarker, SupportsJitter, ...),
                         # UpdateFigure, density_scaled_jitter
```

Plot classes are re-exported from `diffract.viz.plots`, and the `Theme`,
`DEFAULT_THEME`, `DARK_THEME`, `MINIMAL_THEME`, and `apply_theme` symbols are
re-exported from both `diffract.viz.styling` and `diffract.viz`.

## Documentation

```{toctree}
:maxdepth: 2

plots/index
styling
extraction
jitter
```
