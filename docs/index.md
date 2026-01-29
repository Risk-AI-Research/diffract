# Diffract

Diffract is a Python library for analyzing deep neural network weights and tracking how they evolve during training.

## Installation

Install the core package:

```bash
pip install diffract
# or with uv
uv sync
```

### Optional extras

| Extra | Description |
|-------|-------------|
| `torch` | PyTorch model support |
| `tensorflow` | TensorFlow/Keras model support |
| `flax` | Flax/JAX model support |
| `onnx` | ONNX model support |
| `viz` | Plotly visualization helpers |
| `pandas` | Export to pandas DataFrames |
| `polars` | Export to polars DataFrames |
| `redis` | Redis cache backend |
| `zarr` | Zarr storage backend (cloud-native arrays via fsspec) |
| `common` | viz + pandas + polars (recommended) |
| `docs` | Build this documentation |

Install extras with:

```bash
uv sync --extra torch --extra common
# or
pip install "diffract[torch,common]"
```

## Start here

::::{grid} 2
:gutter: 2

:::{grid-item-card} Overview
:link: guide/what_is_diffract
:link-type: doc

What Diffract is, core concepts, and when to use it.
:::

:::{grid-item-card} Quickstart
:link: guide/quickstart
:link-type: doc

5-minute tour: `Session → add → compute → get_results → draw`.
:::

:::{grid-item-card} Recipes
:link: guide/recipes/index
:link-type: doc

Focused how-tos: filtering, kernels, exports, storage backends.
:::

:::{grid-item-card} API Reference
:link: reference/index
:link-type: doc

Docstring-driven reference starting from `Session`.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: User Guide

guide/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/index
writing_docs
```
