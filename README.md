# Diffract: Deep Neural Network Weight Analysis Library

[![CI](https://github.com/Risk-AI-Research/diffract/actions/workflows/ci.yml/badge.svg)](https://github.com/Risk-AI-Research/diffract/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/diffract-core)](https://pypi.org/project/diffract-core/)
[![Python](https://img.shields.io/pypi/pyversions/diffract-core)](https://pypi.org/project/diffract-core/)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://risk-ai-research.github.io/diffract/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official library of
["Diffract: Spectral View of LLM Domain Adaptation"](https://openreview.net/forum?id=XBUHoiAGDE)
(**ICML 2026 Oral**).

Diffract is a Python package for analyzing deep neural network weights and tracking
their evolution over the course of training.

With a straightforward API and a functional design centered on reusable _kernels_,
Diffract automatically resolves dependencies, builds computation graphs, and schedules
calculations. Parameters and results are persisted across sessions.

It accepts models from PyTorch, TensorFlow, Flax, and ONNX, as well as plain
dictionaries of NumPy weight matrices.

<br>

## 🚀 Quick Start

The core package accepts a dictionary of named weight matrices.

```python
import numpy as np
from diffract import analyze, compare

source = {"encoder": np.diag([4.0, 2.0, 1.0]), "head": np.eye(2)}
target = {"encoder": np.diag([3.0, 2.0, 1.0]), "head": np.eye(2)}

analysis = analyze(source)
assert analysis.summary["stable_rank"]["count"] == 2

rows = compare(source, target, fields=("relative_spectrum_change",))
measurements = {row["aggregate_uid"]: row["value_scalar"] for row in rows}
assert len(measurements) == 2 and len(rows) == 4
assert np.isclose(np.mean(list(measurements.values())), 0.5 / np.sqrt(21))
assert {row["member_role"] for row in rows} == {"source", "target"}
```

Each matched parameter contributes one measurement per field. The long rows
identify both members of the pair, which share an `aggregate_uid`.

Pass `export_format="pandas"` for DataFrames with the pandas extra. Use a
`Session` to choose checkpoints or configure kernels. A
`Session(profile="local")` keeps results on disk.

Diffract requires Python 3.12 or newer. The core package installs without any deep
learning framework; heavy dependencies are opt-in extras:

```bash
pip install diffract-core                # core: extraction, spectral metrics, storage
pip install "diffract-core[torch]"       # + PyTorch model loading (CUDA wheels on Linux, ~2-3 GB)
pip install "diffract-core[viz]"         # + Plotly visualization and YAML plot configs
pip install "diffract-core[taichi]"      # + accelerated heavy-tailed fits and p-value kernels
pip install "diffract-core[all]"         # torch + viz + taichi + pandas/polars exports
```

Further extras: `frameworks` (TensorFlow, Flax, ONNX), `pandas` / `polars` (DataFrame
exports), `zarr` (cloud storage), `redis` (shared cache), `notebooks` (tooling for the
example notebooks). The quotes around `"diffract-core[...]"` matter in zsh.

### Development Install

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install (uv provisions the right Python automatically)
git clone https://github.com/Risk-AI-Research/diffract.git
cd diffract
uv sync --extra dev --extra torch
```

Then use it in your code:

```python
from diffract import Session

# Quick experiments (in-memory, no persistence)
session = Session(profile="ram")

# Persistent local storage (SQLite in .diffract/)
session = Session(profile="local")

with session:
    session.models.add(torch_model, model_id="bert-base")
    session.compute.apply("frob_norm", "stable_rank", "log_norm")
    # per-parameter metrics
    metrics_df = session.results.export_metrics(
        "frob_norm", "stable_rank", export_format="pandas"
    )
    # model-level aggregates
    aggregates_df = session.results.export_aggregates(
        "log_norm", export_format="pandas"
    )
```

Check out
[example notebooks](https://github.com/Risk-AI-Research/diffract/tree/main/examples) and
[plot configurations](https://github.com/Risk-AI-Research/diffract/tree/main/examples/configs)
for more examples. The
[notebooks](https://github.com/Risk-AI-Research/diffract/tree/main/notebooks) directory
contains `compare_two_checkpoints.ipynb`, an end-to-end walkthrough.

<br>

## 🤔 Why Diffract?

Neural networks often feel like black boxes. Diffract provides tools to analyze their
internal structure:

- **Training Insights**: Track how weights evolve across training epochs.
- **Architecture Analysis**: Compare different model architectures objectively.
- **Initialization Studies**: Evaluate the impact of initialization methods.
- **Spectral Analysis**: Compute empirical spectral distributions, ranks, and norms.
- **Heavy-Tailed Distributions**: Detect power-law and exponential tails in weight
  spectra.

<br>

## Key Features

- **Session-based API**: Simple `models.add`, `compute.apply`, and
  `results.export_metrics` workflow.

- **Kernels**: Reusable functions that compute model characteristics—ranks, norms,
  spectral properties—stored as named _fields_ on each parameter. Dependencies are
  resolved automatically.

- **Persistent Storage**: Parameters and results survive between sessions. Supports
  HDF5, SQLite, Zarr, and hybrid backends.

- **Kernel Apply Levels**: Kernels can work at multiple levels:

  - **PARAMETER** - Operate on individual weight matrices.
  - **IN_MODEL** - Aggregate within a single model.
  - **CROSS_MODEL** - Compare or aggregate across models. Pairwise comparisons
    (the alignment/overlap family) compare ordered checkpoints. Use
    `session.compute.apply("l_overlap", pairs=[(("net", 1000), ("net", 2000))])` to
    compare two checkpoints of one model.

- **Built-in Visualization**: Publication-ready Plotly plots with theming support.

- **Export Formats**: Get results as `pandas`, `polars`, `dict`, `json`, or `list`.

<br>

## ✨ Core Functionality

### Adding Models

Add models from various frameworks to a session:

<!-- skip: next -->

```python
from diffract import Session

session = Session()

with session:
    session.models.add(torch_model)  # torch.nn.Module
    session.models.add(
        torch_state_dict, model_id="checkpoint"
    )  # Dict[str, torch.Tensor]
    session.models.add(numpy_weights, model_id="raw-weights")  # Dict[str, np.ndarray]
    session.models.add(onnx_model, model_id="onnx-model")  # onnx.ModelProto
    session.models.add(flax_model, model_id="flax-model")  # flax.linen.Module
    session.models.add(tf_model, model_id="tf-model")  # TensorFlow model
```

### Computing Metrics

Dependencies are resolved automatically:

```python
session.compute.apply("frob_norm", "stable_rank")
session.compute.apply("pl_ks")  # has many dependencies—all resolved automatically
```

Every built-in metric—its formula, apply level, required inputs, and configuration—is
catalogued in the
[metrics reference](https://risk-ai-research.github.io/diffract/reference/metrics/).

### Filtering Parameters

Filter computations by model, parameter type, or name:

```python
from diffract import ParameterOverrides, ParameterType, Session

session = Session()

# Assign custom types and names during extraction
overrides = {
    "model.layers.0.attn.q_proj.weight": ParameterOverrides(name="q", ptype="attn"),
    "model.layers.0.attn.k_proj.weight": ParameterOverrides(name="k", ptype="attn"),
    "model.layers.0.mlp.fc1.weight": ParameterOverrides(ptype="mlp"),
}

with session:
    session.models.add(model, model_id="gpt", parameter_overrides=overrides)
    session.compute.apply("frob_norm")

# Scope work to a subset with session.filter(...)
gpt = session.filter(model_ids=["gpt"])
with gpt:
    gpt.compute.apply("frob_norm")

attn = session.filter(param_types=[ParameterType.from_string("attn")])
with attn:
    attn.results.export_metrics("frob_norm")
```

Pass numeric training steps to `session.filter` to select checkpoints.

```python
import numpy as np
from diffract import Session

trajectory = Session(profile="ram")
for step in (0, 80, 160):
    trajectory.models.add({"w": np.diag([step + 1.0, 1.0])}, model_id="net", step=step)

ends = trajectory.filter(model_ids=["net"], steps=[0, 160])
ends.compute.apply("stable_rank")
ranks = ends.results.export_metrics("stable_rank", export_format="list")
assert sorted(row["step"] for row in ranks) == [0, 160]
```

### Retrieving Results

Export results in various formats (`pandas`, `polars`, `dict`, `json`, or `list`):

```python
with session:
    session.compute.apply("stable_rank", "log_norm")

    # PARAMETER-level fields come from export_metrics
    scalars_df = session.results.export_metrics("stable_rank", export_format="pandas")
    # IN_MODEL / CROSS_MODEL fields come from export_aggregates
    aggregates_df = session.results.export_aggregates(
        "log_norm", export_format="pandas"
    )

    # Other formats work the same way
    results = session.results.export_metrics("stable_rank", export_format="polars")
    results = session.results.export_metrics("stable_rank", export_format="dict")
    results = session.results.export_metrics("stable_rank", export_format="json")
    results = session.results.export_metrics("stable_rank", export_format="list")
```

### Visualization

Create publication-ready Plotly plots:

<!-- skip: next -->

```python
from diffract.viz import DEFAULT_THEME

# Ergonomic helpers on session.viz accept field names directly
session.viz.box(y="stable_rank", x="model_id", theme=DEFAULT_THEME).show()
session.viz.scatter(x="frob_norm", y="stable_rank").show()
```

### YAML-Driven Plotting

Define complex visualizations via Hydra configs:

<!-- skip: next -->

```python
session.viz.draw(config_path="examples/configs/boxplot_stable_rank.yaml").show()
```

### Kernel Configuration

List and configure kernels at runtime:

```python
session.compute.list_available_kernels(verbose=True)
session.compute.list_available_metrics(verbose=True)
session.compute.configure_kernel("hard_rank", rtol=1e-6)
```

The next `apply` checks stored results against the effective configuration and
their inputs before reusing them. Pass `recompute=True` to force the requested
kernel even when its stored outputs are fresh.

```python
session.compute.apply("hard_rank")
session.results.status("hard_rank")
```

See [kernel configuration](https://risk-ai-research.github.io/diffract/guide/recipes/kernels_and_compute.html)
for freshness, force recomputation, and kernel output groups.

### Session Management

Parameters and results stay available whenever the session is reopened. `Session()`
defaults to the in-memory `ram` profile; pick a persistent profile such as `local` or
`hybrid` to keep data across separate runs:

```python
from diffract import Session

session = Session()  # in-memory; use profile="local" to persist across runs

# Add models and compute
with session:
    session.models.add(model, model_id="my-model")
    session.compute.apply("frob_norm")

# Reopen the session: earlier results are still there
with session:
    results = session.results.export_metrics("frob_norm", export_format="pandas")
    session.models.list()
    session.models.erase("my-model")
```

### Custom Kernels

Implement your own research metrics using the session kernel decorator:

```python
from diffract import Session

session = Session()

with session:
    # Define and register a custom kernel
    @session.compute.kernel()
    def my_custom_metric(frob_norm: float, *, scaling_factor: float = 1.0) -> float:
        """Custom metric that scales the Frobenius norm."""
        return frob_norm * scaling_factor

    session.models.add(my_model)
    session.compute.configure_kernel("my_custom_metric", scaling_factor=2.0)
    session.compute.apply("my_custom_metric")
```

You can also override the registered name and output fields:

```python
with session:

    @session.compute.kernel(name="scaled_metric", produce_fields=["scaled_result"])
    def custom_analysis(
        frob_norm: float, stable_rank: float, *, weight: float = 0.5
    ) -> float:
        """Custom analysis combining multiple metrics."""
        return weight * frob_norm + (1 - weight) * stable_rank
```

Diffract infers input fields from arguments without defaults. Arguments with
defaults configure the kernel. Use `weights` for the stored weight matrix.
To compute from a one-dimensional feature, attach the vector to the parameter
at the relevant checkpoint and run the kernel in that scope.

```python
initial = trajectory.filter(model_ids=["net"], steps=[0], param_names=["w"])
(owner,) = initial.models.parameters.list()
prefix_scores = np.arange(24)
initial.results.ingest_metrics({owner["uid"]: {"prefix_scores": prefix_scores}})


@trajectory.compute.kernel()
def prefix_mean(prefix_scores):
    return float(np.mean(prefix_scores))


initial.compute.apply("prefix_mean")
(row,) = initial.results.export_metrics("prefix_mean", export_format="list")
assert row["prefix_mean"] == 11.5
```

### Available Kernels

Diffract includes kernels for norms, ranks, spectral analysis, heavy-tailed fits, and
more. Run `session.compute.list_available_kernels(verbose=True)` to list them all.

### Merging Sessions

Merge parameters and results from another session:

<!-- skip: next -->

```python
from diffract import Session

session1 = Session(config_path="config1.ini")
session2 = Session(config_path="config2.ini")

with session1:
    session1.models.add(model1, model_id="model-a")
    session1.compute.apply("frob_norm")

with session2:
    session2.models.add(model2, model_id="model-b")
    session2.utils.merge_other_session(session1, fields=["frob_norm"])
```

### Configuration

Diffract offers built-in **profiles** for common setups:

| Profile  | Storage       | Cache      | Use case                          |
| -------- | ------------- | ---------- | --------------------------------- |
| `ram`    | RAM           | None       | Quick experiments, no persistence |
| `local`  | SQLite        | Simple LRU | Local development, persistent     |
| `hybrid` | SQLite + HDF5 | Simple LRU | Large models, optimized arrays    |

```python
from diffract import Session

# Use a profile (recommended for most users)
session = Session(profile="ram")  # fast, temporary
session = Session(profile="local")  # persistent, simple
session = Session(profile="hybrid")  # persistent, optimized for large arrays

# Or use a custom config file for full control
session = Session(config_path="my_config.ini")
```

**Tip**: Start with a profile, then switch to a config file when you need
reproducibility or custom settings.

#### Advanced Configuration

For production or reproducible experiments, use INI config files. See
`src/diffract/configs/` for examples:

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[cache]
backend = "simple"

[parallel.thread_pool]
max_workers = 4
```

#### Storage Backends

- **RAM**: In-memory (no persistence)
- **SQLite**: Lightweight database for metadata and arrays
- **HDF5**: Optimized for large numerical arrays with compression
- **Zarr**: Cloud-optimized array storage for large-scale data
- **Hybrid**: SQLite (metadata) + HDF5/Zarr (arrays)

#### Cache Backends

- **Simple**: In-memory LRU cache
- **Redis**: Distributed caching (requires `redis` extra)
- **None**: Disable caching

<br>

## 📚 Documentation

For model instances, numeric checkpoints and the direction of a comparison, see
[Addressing weights and comparisons](docs/guide/identity.md).

The documentation is hosted at
[risk-ai-research.github.io/diffract](https://risk-ai-research.github.io/diffract/). It
is sourced from `docs/` and built with Sphinx + MyST; `uv sync --extra docs` and
`make docs` render the HTML locally.

<br>

## Citation

If you use Diffract or build on the paper, please cite:

```bibtex
@inproceedings{borodin2026diffract,
  title     = {Diffract: Spectral View of {LLM} Domain Adaptation},
  author    = {Nikita Borodin and Maria Krylova and Artem Zabolotnyi and Dmitry Aspisov and Egor Shikov and Nikita Tyuplyaev and Oleg Travkin and Roman Alferov and Dmitry Vinichenko},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=XBUHoiAGDE}
}
```

<br>

## Contributions

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the development
setup, the checks CI runs, and the design principles.

<br>

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Risk AI
Research.
