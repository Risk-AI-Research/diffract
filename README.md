# Diffract: Deep Neural Network Weight Analysis Library

Diffract is a Python package for analyzing deep neural network weights and tracking their evolution over the course of training.

With a straightforward API and a functional design centered around reusable *kernels*, Diffract automatically resolves dependencies, builds computation graphs, and schedules calculations. Parameters and results are persisted across sessions.

It works seamlessly with popular frameworks such as PyTorch, TensorFlow, Flax, and ONNX.

<br>

## 🚀 Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone ...
cd diffract
uv sync --extra dev
```

### Optional extras

- `uv sync --extra viz` installs Plotly, Kaleido, Hydra, OmegaConf, and PyYAML so the visualization helpers work. This extra is also pulled in by `make test`, which passes `--extra viz` already.
- `uv sync --extra notebooks` installs notebook-focused tooling (`jupyter`, `matplotlib`, `ipywidgets`) without duplicating the viz-specific packages now centralized in the `viz` extra.

Then use it in your code:

```python
from diffract import Session

# Quick experiments (in-memory, no persistence)
session = Session(profile="ram")

# Persistent local storage (SQLite in .diffract/)
session = Session(profile="local")

with session:
    session.add(torch_model, model_id="bert-base")
    session.compute("frob_norm", "stable_rank")
    # Returns StructuredExportResult with .scalars and .aggregates attributes
    results = session.get_results("frob_norm", "stable_rank", export_format="pandas")
    scalars_df = results.scalars  # DataFrame with per-parameter metrics
    aggregates_df = results.aggregates  # DataFrame with aggregation results
```

Check out [example notebooks](examples/) and [plot configurations](examples/configs/) for more examples.

<br>

## 🤔 Why Diffract?

Neural networks often feel like black boxes. Diffract provides tools to analyze their internal structure:

- **Training Insights**: Track how weights evolve across training epochs.
- **Architecture Analysis**: Compare different model architectures objectively.
- **Initialization Studies**: Evaluate the impact of initialization methods.
- **Spectral Analysis**: Compute empirical spectral distributions, ranks, and norms.
- **Heavy-Tailed Distributions**: Detect power-law and exponential tails in weight spectra.

<br>

## 🔑 Key Features

- **Session-based API**: Simple `add`, `compute`, and `get_results` workflow.

- **Kernels**: Reusable functions that compute model characteristics—ranks, norms, spectral properties. Dependencies are resolved automatically.

- **Persistent Storage**: Parameters and results survive between sessions. Supports HDF5, SQLite, Zarr, and hybrid backends.

- **Flexible Aggregation Levels**: Kernels can work at multiple levels:

  - **PARAMETER** - Operate on individual weight matrices.
  - **MODEL** - Aggregate across all parameters in a model.

- **Built-in Visualization**: Publication-ready Plotly plots with theming support.

- **Export Formats**: Get results in pandas, polars, dict, or JSON formats.

<br>

## ✨ Core Functionality

### Adding Models

Add models from various frameworks to a session:

```python
from diffract import Session

session = Session()

with session:
    session.add(torch_model)  # torch.nn.Module
    session.add(torch_state_dict, model_id="checkpoint")  # Dict[str, torch.Tensor]
    session.add(onnx_model, model_id="onnx-model")  # onnx.ModelProto
    session.add(flax_model, model_id="flax-model")  # flax.linen.Module
    session.add(tf_model, model_id="tf-model")  # TensorFlow model
```

### Computing Metrics

Dependencies are resolved automatically:

```python
session.compute("frob_norm", "stable_rank")
session.compute("pl_ks")  # has many dependencies—all resolved automatically
```

### Filtering Parameters

Filter computations by model, parameter type, or name:

```python
from diffract import Session, ParameterOverrides
from diffract.core.data.nn.parameter import ParameterType

session = Session()

# Assign custom types and names during extraction
overrides = {
    "model.layers.0.attn.q_proj.weight": ParameterOverrides(name="q", ptype="attn"),
    "model.layers.0.attn.k_proj.weight": ParameterOverrides(name="k", ptype="attn"),
    "model.layers.0.mlp.fc1.weight": ParameterOverrides(ptype="mlp"),
}

with session:
    session.add(model, model_id="gpt", parameter_overrides=overrides)
    session.compute("frob_norm", model_ids=["gpt"])
    session.compute("frob_norm", parameter_types=[ParameterType.from_string("attn")])
    session.compute("frob_norm", parameter_names=["q", "k"])
```

### Retrieving Results

Export results in various formats (pandas, polars, dict, or json):

```python
results = session.get_results("stable_rank", export_format="pandas")
# Returns StructuredExportResult with separate scalars and aggregates:
# - scalars: DataFrame with per-parameter metrics
# - aggregates: DataFrame with aggregation/cross-entity results
scalars_df = results.scalars
aggregates_df = results.aggregates

# Other formats work the same way
results = session.get_results("stable_rank", export_format="polars")
results = session.get_results("stable_rank", export_format="dict")
results = session.get_results("stable_rank", export_format="json")
```

### Visualization

Create publication-ready Plotly plots:

```python
from diffract.viz.plots import BoxPlot, ScatterPlot
from diffract.viz.themes import DEFAULT_THEME

# Create and render plots using the session.draw() method
session.draw(plot=BoxPlot(field="stable_rank", theme=DEFAULT_THEME)).show()
session.draw(plot=ScatterPlot(x_field="frob_norm", y_field="stable_rank")).show()
```

### YAML-Driven Plotting

Define complex visualizations via Hydra configs:

```python
session.draw(config_path="examples/configs/boxplot_stable_rank.yaml").show()
```

### Kernel Configuration

List and configure kernels at runtime:

```python
session.list_kernels(verbose=True)
session.list_fields_can_compute(verbose=True)
session.configure_kernel("hard_rank", threshold=1e-6)
```

### Session Management

Data persists automatically—parameters and results survive between runs:

```python
from diffract import Session

session = Session()

# First run: add models and compute
with session:
    session.add(model, model_id="my-model")
    session.compute("frob_norm")

# Later: data persists across runs
with session:
    results = session.get_results("frob_norm", export_format="pandas")
    session.list_models()
    session.erase_models("old-model")
```

### Custom Kernels

Implement your own research metrics using the session kernel decorator:

```python
from diffract import Session

session = Session()

with session:
    # Define and register a custom kernel
    @session.kernel()
    def my_custom_metric(frob_norm: float, *, scaling_factor: float = 1.0) -> float:
        """Custom metric that scales frobenius norm."""
        return frob_norm * scaling_factor

    session.add(my_model)
    session.configure_kernel("my_custom_metric", scaling_factor=2.0)
    session.compute("my_custom_metric")
```

You can also customize kernel parameters:

```python
with session:
    @session.kernel(name="scaled_metric", produce_fields=["scaled_result"])
    def custom_analysis(frob_norm: float, stable_rank: float, *, weight: float = 0.5) -> float:
        """Custom analysis combining multiple metrics."""
        return weight * frob_norm + (1 - weight) * stable_rank
```

### Available Kernels

Diffract includes kernels for norms, ranks, spectral analysis, heavy-tailed fits, and more. Run `session.list_kernels(verbose=True)` to list them all.

### Merging Sessions

Merge parameters and results from another session:

```python
from diffract import Session

session1 = Session(config_path="config1.ini")
session2 = Session(config_path="config2.ini")

with session1:
    session1.add(model1, model_id="model-a")
    session1.compute("frob_norm")

with session2:
    session2.add(model2, model_id="model-b")
    session2.merge(session1, fields=["frob_norm"])
```

### Configuration

Diffract offers built-in **profiles** for common setups:

| Profile | Storage | Cache | Use case |
|---------|---------|-------|----------|
| `ram` | RAM | None | Quick experiments, no persistence |
| `local` | SQLite | Simple LRU | Local development, persistent |
| `hybrid` | SQLite + HDF5 | Simple LRU | Large models, optimized arrays |

```python
from diffract import Session

# Use a profile (recommended for most users)
session = Session(profile="ram")      # fast, temporary
session = Session(profile="local")    # persistent, simple
session = Session(profile="hybrid")   # persistent, optimized for large arrays

# Or use a custom config file for full control
session = Session(config_path="my_config.ini")
```

**Tip**: Start with a profile, then switch to a config file when you need reproducibility or custom settings.

#### Advanced Configuration

For production or reproducible experiments, use INI config files. See `configs/` for examples:

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[cache]
backend = "simple"

[compute.executor]
max_workers = 4
```

#### Storage Backends

- **RAM**: In-memory (no persistence)
- **SQLite**: Lightweight database for metadata and arrays
- **HDF5**: Optimized for large numerical arrays with compression
- **Zarr**: Cloud-optimized array storage for large-scale data
- **Hybrid**: SQLite (metadata) + HDF5/Zarr (arrays) — best of both

#### Cache Backends

- **Simple**: In-memory LRU cache
- **Redis**: Distributed caching (requires `redis` extra)
- **None**: Disable caching

## 📚 Documentation

The documentation site is sourced from `docs/` and built with Sphinx + MyST.
Install the tooling via `uv sync --extra docs` and run `make docs` to render the HTML locally.

<br>

## ❤️ Contributions

Contributions are welcome! Fork the repo, create a feature branch, and submit a PR. Use `make lint` and `make test` to validate your changes.

<br>

## License

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Risk AI Research.
