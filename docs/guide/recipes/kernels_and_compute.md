# Kernels and Compute

This page explains how Diffract kernels work: configuration, dependencies, aggregation,
and parallel execution.

## Fields vs kernels

- A **field** is a named value stored on each parameter (e.g., `frob_norm`,
  `stable_rank`).
- A **kernel** is a function that produces one or more fields.

When you call:

```python
session.compute.apply("stable_rank")
```

Diffract finds the kernel that produces `stable_rank` and executes it (plus any
dependencies).

Run `session.compute.list_available_kernels(verbose=True)` to list every registered
kernel with its required and produced fields.

## Configuring kernels

Some kernels accept configuration parameters. Use `session.compute.configure_kernel()`
before computing:

```python
from diffract import Session

session = Session(profile="ram")

with session:
    session.compute.configure_kernel("hard_rank", rtol=1e-6)
    session.compute.apply("hard_rank")
```

Kernel configuration is process-global: the registry is a module-level singleton shared
by all sessions in the process. Configuration set in one session applies to every other
session and resets only when the interpreter restarts.

## Dependency resolution

Each kernel declares its **required input fields** and **output fields**. Diffract
builds a dependency graph and executes kernels in topological order.

Example:

- Kernel A: `produce_fields=("x",)`
- Kernel B: `require_fields=("x",)`, `produce_fields=("y",)`

Computing `y` automatically executes A first, then B.

## Kernel signatures

Diffract infers kernel metadata from the Python function signature:

- Parameters **without defaults** → required input fields
- Parameters **with defaults** → configurable keyword arguments

```python
def hard_rank(esd, *, rtol: float = 1e-5) -> int: ...
```

Here, `esd` is a required field; `rtol` is configurable via
`session.compute.configure_kernel("hard_rank", rtol=...)`.

**Note:** `*args` and `**kwargs` are not allowed in kernel signatures.

## Apply levels

Kernels operate at one of three levels:

| Level         | Scope                                | Example                                         |
| ------------- | ------------------------------------ | ----------------------------------------------- |
| `PARAMETER`   | Per parameter                        | `frob_norm` — Frobenius norm of each layer      |
| `IN_MODEL`    | Per model (aggregates by `model_id`) | `param_norm` — sum of squared norms             |
| `CROSS_MODEL` | Per parameter name across models     | `l_overlap` — compare layers across checkpoints |

### Contextual field names

Aggregated kernels write results with a deterministic suffix to avoid collisions:

```
some_metric@models[m1,m2]@params[layer.0.weight]
```

When you call `results.export_aggregates("some_metric", ...)`, Diffract matches both the
base name and contextual variants.

## Kernel outputs

Diffract normalizes kernel return values:

| Return type | Behavior                                |
| ----------- | --------------------------------------- |
| `dict`      | Used as-is: `{field_name: value}`       |
| `tuple`     | Mapped positionally to `produce_fields` |
| scalar      | Stored under the single declared field  |

This allows one kernel to produce multiple fields in a single pass.

## Execution flow

For per-parameter kernels:

1. Filter to parameters missing the target fields
1. Group the pending parameters into **chunks** sized by an approximate read budget,
   derived from the cache manager's available bytes with headroom
1. For each chunk: **prefetch** the required input fields into memory, then execute the
   kernel on every parameter in the chunk

Chunk sizes are estimated from stored field metadata (shape and dtype), so large fields
such as weight matrices produce small chunks and scalar fields produce large ones.

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

If a kernel is marked `PARALLEL`, its tasks run on a `ProcessPoolExecutor` sized by
`parallel.process_pool.max_workers`.

**Practical notes:**

- Kernel implementations must be picklable (safe for multiprocessing)
- Prefer top-level functions; avoid closures with non-serializable state
- Parallelism has overhead — it helps when per-task computation is substantial

## Registering custom kernels

Built-in kernels live in `diffract/core/compute/kernels/`. Register your own with the
`@session.compute.kernel()` decorator — required input fields and configurable keyword
arguments are inferred from the function signature:

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

By default the kernel is named after the function, produces a single field with the same
name, runs at `PARAMETER` level, and executes sequentially. The decorator accepts
`name`, `require_fields`, `produce_fields`, `apply_level`, `execution_protocol`, and
`restrictions` for full control. For example, an `IN_MODEL` kernel receives each
required field as a tuple of per-parameter values and writes one aggregate per model:

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

    aggregates = session.results.export_aggregates(
        "total_frob_norm", export_format="dict"
    )
```

Because the registry is process-global, a kernel registered through one session is
available to every session in the process. Kernels marked `PARALLEL` must be picklable
top-level functions (see [Parallelism](#parallelism)).

______________________________________________________________________

## Built-in fields reference

Every built-in field -- its formula, apply level, required inputs, and configuration --
is documented in the [metric catalog](../../reference/metrics/index.md), generated from
the kernel registry so it never drifts from the registered kernels. The per-category
pages there give the mathematics, conventions, and pitfalls.

### Fit method

The fit kernels -- `power_law_fit`, `truncated_power_law_fit`, and `exponential_fit` --
accept a `fit_method` keyword argument selecting the fitting implementation:

- `"auto"` (default) -- the accelerated taichi implementation when the `taichi` extra is
  installed and the ESD has at least 100 points; the `powerlaw` library otherwise. If
  taichi is installed but fails to initialize, `auto` falls back to the `powerlaw`
  library.
- `"powerlaw"` -- always the `powerlaw` library.
- `"diffract"` -- always the accelerated implementation; raises if taichi is unavailable
  or fails to initialize.

The 100-point floor is statistical, not performance: the accelerated fitter only
considers `xmin` candidates that leave a tail of at least 50 points (the reliability
bound of Clauset et al.), so below roughly twice that it cannot select a tail at all. On
small ESDs (roughly 100-300 points) this constrained search yields more conservative
estimates than the `powerlaw` library's unrestricted one, which minimizes KS over
arbitrarily small tails; pass `fit_method="powerlaw"` for exact parity with it. For the
same reason, the p-value kernels return NaN when the fitted tail has fewer than 50
points.

Select the implementation per kernel:

```python
session.compute.configure_kernel("power_law_fit", fit_method="powerlaw")
session.compute.apply("pl_alpha")
```
