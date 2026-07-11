# Diffract

Diffract is a Python library for analyzing deep neural network weights and tracking how they evolve during training.

## Installation

Diffract requires Python 3.12 (uv provisions it automatically). Clone the
repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Risk-AI-Research/diffract.git
cd diffract
uv sync --extra dev
```

### Optional extras

| Extra | Description |
|-------|-------------|
| `torch` | PyTorch model support |
| `tensorflow` | TensorFlow/Keras model support |
| `flax` | Flax/JAX model support |
| `onnx` | ONNX model support |
| `frameworks` | TensorFlow + Flax + ONNX bundle |
| `viz` | Plotly visualization helpers |
| `pandas` | Export to pandas DataFrames |
| `polars` | Export to polars DataFrames |
| `redis` | Redis cache backend |
| `zarr` | Zarr storage backend (cloud-native arrays via fsspec) |
| `taichi` | Taichi-accelerated heavy-tailed fitting and bootstrap p-value kernels |
| `common` | viz + pandas + polars (recommended) |
| `notebooks` | Dependencies for the example notebooks |
| `docs` | Build this documentation |

Install extras with:

```bash
uv sync --extra torch --extra common
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

5-minute tour: `Session → models.add → compute.apply → results.export_metrics → viz.draw`.
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
