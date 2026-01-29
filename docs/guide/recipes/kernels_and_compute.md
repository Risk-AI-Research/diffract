# Kernels and Compute

This page explains how Diffract kernels work: configuration, dependencies, aggregation, and parallel execution.

## Fields vs kernels

- A **field** is a named value stored on each parameter (e.g., `frob_norm`, `stable_rank`).
- A **kernel** is a function that produces one or more fields.

When you call:

```python
session.compute("stable_rank")
```

Diffract finds the kernel that produces `stable_rank` and executes it (plus any dependencies).

## Configuring kernels

Some kernels accept configuration parameters. Use `session.configure_kernel()` before computing:

```python
from diffract import Session

session = Session(profile="ram")

with session:
    session.configure_kernel("hard_rank", threshold=1e-6)
    session.compute("hard_rank")
```

Configuration is session-scoped — it resets when you create a new session.

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

Here, `esd` is a required field; `threshold` is configurable via `session.configure_kernel("hard_rank", threshold=...)`.

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

When you call `get_results("some_metric", ...)`, Diffract matches both the base name and contextual variants.

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
2. Attempt to **prefetch** required input fields into memory
3. If prefetch fails (memory limits), fall back to **chunked execution**

Chunking configuration:

```ini
[compute.executor]
max_workers = 8
chunk_size = 32
minimal_chunk_size = 1
```

The executor recursively halves chunks until prefetch succeeds or `minimal_chunk_size` is reached.

## Parallelism

Parallelism is controlled by:

- `compute.executor.max_workers` — global worker count
- Each kernel's `KernelExecutionProtocol` (`SEQUENTIAL` or `PARALLEL`)

If `max_workers > 1` and a kernel is marked `PARALLEL`, tasks run via `ProcessPoolExecutor`.

**Practical notes:**

- Kernel implementations must be picklable (safe for multiprocessing)
- Prefer top-level functions; avoid closures with non-serializable state
- Parallelism has overhead — it helps when per-task computation is substantial

## Registering custom kernels (advanced)

Built-in kernels live in `diffract/core/compute/kernels/`. For experimentation, you can register kernels at runtime using internal APIs:

```python
import numpy as np
from diffract import Session
from diffract.core.compute.execution.enums import KernelApplyLevel, KernelExecutionProtocol

session = Session(profile="ram")
registry = session._container.compute_singleton.kernel_registry()


def w_sum(weights: np.ndarray) -> float:
    return float(np.sum(weights))


_, cfg = registry._split_signature(w_sum)
registry.register_kernel(
    name="w_sum",
    require_fields=("weights",),
    produce_fields=("w_sum",),
    implementation=w_sum,
    apply_level=KernelApplyLevel.PARAMETER,
    execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
    restrictions=None,
    config=cfg,
)

with session:
    session.compute("w_sum")
```

**Warning:** This uses internal APIs that may change. For stable extensions, contribute kernels to the codebase.

---

## Built-in fields reference

Diffract provides 60+ built-in fields organized by category.

### Matrix properties

| Field | Description |
|-------|-------------|
| `shape` | Matrix shape as array |
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
| `pl_p_value` | Power law p-value |
| `pl_concentration`, `pl_presence` | Power law metrics |
| `tpl_alpha`, `tpl_lambda`, `tpl_esd_xmin`, `tpl_ks` | Truncated power law |
| `tpl_p_value`, `tpl_concentration`, `tpl_presence`, `tpl_scale` | TPL metrics |
| `expon_lambda`, `expon_esd_xmin`, `expon_ks` | Exponential fit |
| `expon_p_value`, `expon_concentration`, `expon_presence`, `expon_scale` | Exponential metrics |

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
| `svs_similarity` | Singular vector similarity |
