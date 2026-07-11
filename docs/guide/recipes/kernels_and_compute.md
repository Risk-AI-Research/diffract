# Kernels and Compute

This page explains how Diffract kernels work: configuration, dependencies, aggregation, and parallel execution.

## Fields vs kernels

- A **field** is a named value stored on each parameter (e.g., `frob_norm`, `stable_rank`).
- A **kernel** is a function that produces one or more fields.

When you call:

```python
session.compute.apply("stable_rank")
```

Diffract finds the kernel that produces `stable_rank` and executes it (plus any dependencies).

Run `session.compute.list_available_kernels(verbose=True)` to list every registered kernel with its required and produced fields.

## Configuring kernels

Some kernels accept configuration parameters. Use `session.compute.configure_kernel()` before computing:

```python
from diffract import Session

session = Session(profile="ram")

with session:
    session.compute.configure_kernel("hard_rank", threshold=1e-6)
    session.compute.apply("hard_rank")
```

Kernel configuration is process-global: the registry is a module-level singleton shared by all sessions in the process. Configuration set in one session applies to every other session and resets only when the interpreter restarts.

## Dependency resolution

Each kernel declares its **required input fields** and **output fields**. Diffract builds a dependency graph and executes kernels in topological order.

Example:

- Kernel A: `produce_fields=("x",)`
- Kernel B: `require_fields=("x",)`, `produce_fields=("y",)`

Computing `y` automatically executes A first, then B.

## Kernel signatures

Diffract infers kernel metadata from the Python function signature:

- Parameters **without defaults** → required input fields
- Parameters **with defaults** → configurable keyword arguments

```python
def hard_rank(esd, *, threshold: float = 1e-5) -> int:
    ...
```

Here, `esd` is a required field; `threshold` is configurable via `session.compute.configure_kernel("hard_rank", threshold=...)`.

**Note:** `*args` and `**kwargs` are not allowed in kernel signatures.

## Apply levels

Kernels operate at one of three levels:

| Level | Scope | Example |
|-------|-------|---------|
| `PARAMETER` | Per parameter | `frob_norm` — Frobenius norm of each layer |
| `IN_MODEL` | Per model (aggregates by `model_id`) | `param_norm` — sum of squared norms |
| `CROSS_MODEL` | Per parameter name across models | `l_overlap` — compare layers across checkpoints |

### Contextual field names

Aggregated kernels write results with a deterministic suffix to avoid collisions:

```
some_metric@models[m1,m2]@params[layer.0.weight]
```

When you call `results.export_aggregates("some_metric", ...)`, Diffract matches both the base name and contextual variants.

## Kernel outputs

Diffract normalizes kernel return values:

| Return type | Behavior |
|-------------|----------|
| `dict` | Used as-is: `{field_name: value}` |
| `tuple` | Mapped positionally to `produce_fields` |
| scalar | Stored under the single declared field |

This allows one kernel to produce multiple fields in a single pass.

## Execution flow

For per-parameter kernels:

1. Filter to parameters missing the target fields
2. Group the pending parameters into **chunks** sized by an approximate read budget, derived from the cache manager's available bytes with headroom
3. For each chunk: **prefetch** the required input fields into memory, then execute the kernel on every parameter in the chunk

Chunk sizes are estimated from stored field metadata (shape and dtype), so large fields such as weight matrices produce small chunks and scalar fields produce large ones.

Worker pool configuration:

```ini
[parallel.thread_pool]
max_workers = 8

[parallel.process_pool]
max_workers = 8
```

## Parallelism

Parallelism is controlled by:

- `parallel.thread_pool.max_workers` — threads for view filtering and field prefetching
- `parallel.process_pool.max_workers` — worker processes for kernel execution
- Each kernel's `KernelExecutionProtocol` (`SEQUENTIAL` or `PARALLEL`)

If a kernel is marked `PARALLEL`, its tasks run on a `ProcessPoolExecutor` sized by `parallel.process_pool.max_workers`.

**Practical notes:**

- Kernel implementations must be picklable (safe for multiprocessing)
- Prefer top-level functions; avoid closures with non-serializable state
- Parallelism has overhead — it helps when per-task computation is substantial

## Registering custom kernels

Built-in kernels live in `diffract/core/compute/kernels/`. Register your own with the `@session.compute.kernel()` decorator — required input fields and configurable keyword arguments are inferred from the function signature:

```python
import torch
from diffract import Session

model = torch.nn.Sequential(torch.nn.Linear(32, 64))
session = Session(profile="ram")

with session:

    @session.compute.kernel()
    def scaled_frob_norm(frob_norm: float, *, scale: float = 1.0) -> float:
        return frob_norm * scale

    session.models.add(model, model_id="m1")
    session.compute.configure_kernel("scaled_frob_norm", scale=2.0)
    session.compute.apply("scaled_frob_norm")

    metrics = session.results.export_metrics("scaled_frob_norm", export_format="dict")
```

By default the kernel is named after the function, produces a single field with the same name, runs at `PARAMETER` level, and executes sequentially. The decorator accepts `name`, `require_fields`, `produce_fields`, `apply_level`, `execution_protocol`, and `restrictions` for full control. For example, an `IN_MODEL` kernel receives each required field as a tuple of per-parameter values and writes one aggregate per model:

```python
import numpy as np

from diffract.core.compute.execution import KernelApplyLevel, KernelExecutionProtocol

with session:

    @session.compute.kernel(
        name="total_frob_norm",
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
    )
    def total_frob_norm(frob_norm: tuple[float, ...]) -> float:
        return float(np.sum(np.square(frob_norm)))

    session.compute.apply("total_frob_norm")

    aggregates = session.results.export_aggregates("total_frob_norm", export_format="dict")
```

Because the registry is process-global, a kernel registered through one session is available to every session in the process. Kernels marked `PARALLEL` must be picklable top-level functions (see [Parallelism](#parallelism)).

---

## Built-in fields reference

Diffract provides 75+ built-in fields organized by category.

### Matrix properties

| Field | Description |
|-------|-------------|
| `greater_dim` | Larger dimension |
| `lower_dim` | Smaller dimension |
| `aspect_ratio` | greater_dim / lower_dim |
| `weights_std` | Standard deviation of weights |
| `weights_rand` | Randomized weight matrix |

### Decomposition

| Field | Description |
|-------|-------------|
| `weights_lsvs` | Left singular vectors |
| `weights_svals` | Singular values |
| `weights_rsvs` | Right singular vectors |
| `esd` | Empirical spectral distribution (squared svals / greater_dim) |
| `esd_max`, `esd_min` | ESD bounds |
| `max_weights_sval`, `min_weights_sval` | Singular value bounds |
| `weights_rand_lsvs`, `weights_rand_svals`, `weights_rand_rsvs` | SVD of the randomized weight matrix |
| `max_weights_rand_sval`, `min_weights_rand_sval` | Randomized singular value bounds |
| `esd_rand`, `esd_rand_max`, `esd_rand_min` | Randomized ESD and bounds |

### Norms

| Field | Description |
|-------|-------------|
| `frob_norm` | Frobenius norm |
| `l1_norm` | L1 norm (sum of singular values) |
| `l2_norm` | Spectral norm (max singular value) |
| `log_norm` | Mean squared log Frobenius (model-level) |
| `log_spectral_norm` | Mean squared log spectral (model-level) |
| `param_norm` | Sum of squared Frobenius (model-level) |
| `prod_frob_norm` | Product of Frobenius norms (model-level) |
| `prod_spectral_norm` | Product of spectral norms (model-level) |
| `pl_alpha_norm`, `tpl_alpha_norm` | Heavy-tailed alpha norms |
| `model_pl_alpha_norm`, `model_tpl_alpha_norm` | Mean log10 alpha norm (model-level) |

### Ranks

| Field | Description |
|-------|-------------|
| `stable_rank` | (frob_norm / l2_norm)² |
| `effective_rank` | Entropy-based rank |
| `hard_rank` | Count of eigenvalues > threshold (configurable) |
| `mp_soft_rank` | MP bulk max / ESD max |

### Marchenko-Pastur (random matrix theory)

| Field | Description |
|-------|-------------|
| `mp_esd_max`, `mp_esd_min` | MP bulk bounds |
| `mp_bulk_std` | MP bulk std |
| `mp_sval_max` | Max singular value from MP |
| `mp_ks` | KS statistic for MP fit |
| `mp_concentration` | Fraction of ESD in MP bulk |
| `mp_presence` | MP bulk width / total width |
| `mp_num_spikes` | Spikes above MP bulk |

### Tracy-Widom

| Field | Description |
|-------|-------------|
| `tw_esd_bound` | Tracy-Widom spike threshold |
| `tw_num_spikes` | Spikes above TW bound |

### Heavy-tailed fits

| Field | Description |
|-------|-------------|
| `pl_alpha`, `pl_esd_xmin`, `pl_ks` | Power law fit parameters |
| `pl_p_value` | Power law bootstrap p-value (requires the `taichi` extra) |
| `pl_concentration`, `pl_presence` | Power law metrics |
| `tpl_alpha`, `tpl_lambda`, `tpl_esd_xmin`, `tpl_ks` | Truncated power law |
| `tpl_p_value`, `tpl_concentration`, `tpl_presence`, `tpl_scale` | TPL metrics (p-value requires the `taichi` extra) |
| `expon_lambda`, `expon_esd_xmin`, `expon_ks` | Exponential fit |
| `expon_p_value`, `expon_concentration`, `expon_presence`, `expon_scale` | Exponential metrics (p-value requires the `taichi` extra) |

#### Fit method

The fit kernels — `power_law_fit`, `truncated_power_law_fit`, and `exponential_fit` — accept a `fit_method` keyword argument selecting the fitting implementation:

- `"auto"` (default) — the accelerated taichi implementation when the `taichi` extra is installed and the ESD has at least 100 points; the `powerlaw` library otherwise. If taichi is installed but fails to initialize, `auto` falls back to the `powerlaw` library.
- `"powerlaw"` — always the `powerlaw` library.
- `"diffract"` — always the accelerated implementation; raises if taichi is unavailable or fails to initialize.

The 100-point floor is statistical, not performance: the accelerated fitter only considers `xmin` candidates that leave a tail of at least 50 points (the reliability bound of Clauset et al.), so below roughly twice that it cannot select a tail at all. On small ESDs (roughly 100-300 points) this constrained search yields more conservative estimates than the `powerlaw` library's unrestricted one, which minimizes KS over arbitrarily small tails; pass `fit_method="powerlaw"` for exact parity with it. For the same reason, the p-value kernels return NaN when the fitted tail has fewer than 50 points.

Select the implementation per kernel:

```python
session.compute.configure_kernel("power_law_fit", fit_method="powerlaw")
session.compute.apply("pl_alpha")
```

### Model quality

| Field | Description |
|-------|-------------|
| `pl_alpha_weighted` | Alpha weighted by log ESD max |
| `rand_distance` | Jensen-Shannon divergence to randomized ESD |

### Alignment (cross-model)

| Field | Description |
|-------|-------------|
| `l_overlap` | Left singular vector overlap between models |
| `r_overlap` | Right singular vector overlap between models |
| `l_agreement`, `r_agreement` | Diagonal of the overlap matrix (per-component agreement) |
| `max_l_agreement`, `max_r_agreement` | Max absolute overlap per row (best agreement per component) |
| `avg_l_agreement`, `avg_r_agreement` | Mean per-component agreement |
| `avg_max_l_agreement`, `avg_max_r_agreement` | Mean best agreement per component |
