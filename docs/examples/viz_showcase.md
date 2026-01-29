# Visualization showcase

This page is a documentation-friendly version of `examples/viz_showcase.ipynb`.
It shows how to:

- Compute a few fields via the public `Session` API.
- Add custom per-parameter metadata (so plots can group/color by it).
- Render several Plotly figures via `diffract.viz`, both from Python objects and
  from YAML configs (Hydra-style).

## Prerequisites

Install the visualization extra (and a framework extra if you want to run the
example end-to-end):

```bash
uv sync --extra viz
uv sync --extra torch  # optional, only needed if you run the toy model code
```

## 1) Create a session and add models

Diffract stores parameters and computed fields in the configured backend. For an
interactive visualization workflow you typically want persistence:

```python
from diffract import Session

session = Session(profile="local")  # or "hybrid" for large models
```

The notebook uses a small PyTorch model and attaches metadata like `layer_id` and
`head_id` using `ParameterOverrides`. That metadata becomes available to plots as
columns you can group/color by.

```python
import re

import torch.nn as nn
from diffract import ParameterOverrides


def build_overrides(model: nn.Module) -> dict[str, ParameterOverrides]:
    overrides: dict[str, ParameterOverrides] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        m = re.match(r"^layers\.(\d+)\.heads\.(\d+)\.proj$", name)
        if m:
            overrides[name] = ParameterOverrides(
                other_meta={
                    "layer_id": int(m.group(1)),
                    "head_id": int(m.group(2)),
                    "kind": "attn_proj",
                }
            )
            continue
        m = re.match(r"^layers\.(\d+)\.ffn$", name)
        if m:
            overrides[name] = ParameterOverrides(
                other_meta={
                    "layer_id": int(m.group(1)),
                    "head_id": None,
                    "kind": "ffn",
                }
            )
    return overrides
```

Add one or more models:

```python
with session:
    session.add(model_small, model_id="toy_small", parameter_overrides=build_overrides(model_small))
    session.add(model_big, model_id="toy_big", parameter_overrides=build_overrides(model_big))
```

## 2) Compute fields

Compute a few scalar fields that plots can consume:

```python
with session:
    session.compute("shape", "frob_norm", "effective_rank", "stable_rank")
```

## 3) Plot from Python objects

### Box plot (grouped by model)

```python
from diffract.viz.plots.scalar import BoxPlot

with session:
    fig = session.draw(
        plot=BoxPlot(
            field="stable_rank",
            title="stable_rank by model_id",
            group_by="model_id",
        )
    )
    fig.show()
```

### Scatter plot (two scalar fields)

```python
from diffract.viz.plots.scatter import ScatterPlot

with session:
    fig = session.draw(
        plot=ScatterPlot(
            x_field="frob_norm",
            y_field="stable_rank",
            title="stable_rank vs frob_norm",
            group_by="model_id",
        )
    )
    fig.show()
```

## 4) Plot from YAML configs (Hydra-style)

For reproducible and shareable plots, you can keep plot definitions in YAML and
render them via `Session.draw(config_path=...)`.

All YAML configs used in the notebook live in `examples/configs/`. For example:

```python
from pathlib import Path

CONFIGS_DIR = Path("examples/configs")

with session:
    fig = session.draw(
        config_path=CONFIGS_DIR / "boxplot_stable_rank.yaml",
        overrides=[],  # optional Hydra overrides
    )
    fig.show()
```

See [Plot configs](plot_configs.md) for the YAML structure and available plot types.

## 5) Themes and coloring by metadata

If you attach metadata via `other_meta` (like `layer_id`), you can color by it:

```python
from diffract.viz.plots.scalar import BoxPlot
from diffract.viz.themes import DARK_THEME, MINIMAL_THEME

with session:
    fig = session.draw(
        plot=BoxPlot(
            field="stable_rank",
            title="Stable Rank by Model (Dark theme, color by layer_id)",
            group_by="model_id",
            color_by="layer_id",
            theme=DARK_THEME,
        )
    )
    fig.show()

with session:
    fig2 = session.draw(
        plot=BoxPlot(
            field="stable_rank",
            title="Stable Rank (Minimal theme)",
            group_by="model_id",
            theme=MINIMAL_THEME,
        )
    )
    fig2.show()
```

You can also load a theme from YAML (see `examples/configs/theme_example.yaml` in the repo):

```python
from pathlib import Path

with session:
    fig = session.draw(
        config_path=Path("examples/configs/boxplot_greater_dim.yaml"),
        theme_path=Path("examples/configs/theme_example.yaml"),
    )
    fig.show()
```

## 6) Export results (e.g. pandas)

```python
with session:
    df = session.get_results(
        "shape",
        "frob_norm",
        "effective_rank",
        "stable_rank",
        export_format="pandas",
    )
```

