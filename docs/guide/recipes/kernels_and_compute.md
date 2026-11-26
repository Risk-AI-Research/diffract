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

### Versioning custom kernels

Set the implementation revision with `@session.compute.kernel(version="2")`.
The default revision is `"1"`. Bump the revision when behavior changes, including
changes to an existing default that affect results.

Use JSON-compatible configuration for a stable fingerprint. Supported values
include finite numbers, booleans, strings, lists, string-keyed dictionaries, and
`None`. Dictionary order is ignored, while numeric types such as `1` and `1.0`
retain their distinct representations. Explicitly setting a value to its default
has the same fingerprint as leaving it at that default.

### Reconfiguring after results are stored

`apply` checks the effective configuration and the change sequences of its direct
inputs before reusing stored results. Use `results.status(field)` to inspect the same
checks against the current metadata. The report contains one entry per field and
identity in scope, with a `status` of `fresh`, `stale`, or `never computed`.

```python
import numpy as np

from diffract import Session

session = Session(profile="ram")
with session:
    session.models.add({"layer.weight": np.diag([1.0, 0.2])}, model_id="example")
    session.compute.configure_kernel("hard_rank", rtol=1e-9)
    session.compute.apply("hard_rank")

    session.compute.configure_kernel("hard_rank", rtol=0.5)
    (entry,) = session.results.status("hard_rank")
    print(entry.status, entry.reason)  # stale config key rtol changed
    session.compute.apply("hard_rank")
```

An inherited result eligible for verification reports `fresh` with reason
`pre-provenance`. The first `apply` records the effective configuration for that
output group while preserving its payload. SVD producers recompute inherited
spectra and vectors whose arithmetic is unknown. See
[SVD precision and execution records](../../reference/metrics/spectral.md#precision-and-execution-records).

Use `session.compute.apply("hard_rank", recompute=True)` to run the producing
kernel even when its stored result is fresh. For a multi-output kernel, the
call runs the full output group in the active session scope. Kernels needed
only as dependencies still follow the freshness check. The returned
`ApplySummary.forced_fields` lists the fields produced by forced runs.

### Memory used by SVD

The SVD kernels load one parameter's inputs at a time and write results in
batches. For CPU execution, Diffract estimates the extra working memory
from the matrix dimensions and requested factors. It budgets eight bytes per
element for real inputs. With economy vectors, the allowance includes a term
quadratic in the shorter dimension. The estimate covers workspace, copies
and output sorting. See {py:func}`~diffract.core.compute.extensions.svd_workspace.estimate_svd_workspace_bytes`.

When available host RAM can be measured, the CPU SVD raises an error before
calling LAPACK if the estimate exceeds 80% of it. On Linux, the budget also
considers detected cgroup v2 limits. If the measurement is unavailable,
Diffract logs a warning and continues. Each check is a snapshot of available
memory, so other processes can still change memory pressure during a run.

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

| Level         | Scope                                           | Example                                             |
| ------------- | ----------------------------------------------- | --------------------------------------------------- |
| `PARAMETER`   | Per parameter                                   | `frob_norm` — Frobenius norm of each layer          |
| `IN_MODEL`    | Per model instance and training step            | `param_norm` — sum of squared norms                 |
| `CROSS_MODEL` | Per tensor name across the selected checkpoints | `l_overlap` compares two left singular-vector bases |

### Pair kernels compare two model checkpoints

The alignment family — `l_overlap`, `r_overlap`, and the agreement metrics derived
from them (`l_agreement`, `max_l_agreement`, `avg_l_agreement`,
`avg_max_l_agreement`, and their right-basis `r_*` counterparts) — is
**pairwise**: each compares one layer between two model versions. Scope the session to the pair with `filter(model_ids=[a, b])` before
applying:

```python
with session:
    pair = session.filter(model_ids=["checkpoint_1000", "checkpoint_2000"])
    pair.compute.apply("l_agreement")  # l_overlap is resolved automatically
```

Pair exports identify each operand with `model_source`, `step_source`,
`model_target`, and `step_target`. For overlap, rows index the source
basis and columns index the target basis. Swapping the pair transposes
the matrix. To compare two checkpoints of one model, select its name and
two steps with `session.filter(model_ids=["net"], steps=[0, 100])`. The following table applies to both the left and right bases.

| Quantity                          | Effect of reversing the operands |
| --------------------------------- | -------------------------------- |
| Overlap matrix                    | Transposes the matrix            |
| Diagonal agreement and its mean   | Preserves the value              |
| Best-match agreement and its mean | Takes maxima over the other axis |

In a session containing steps 0, 1 and 2 of one model, request a single
direction explicitly:

```python
session.compute.apply("l_overlap", pairs=[(("net", 0), ("net", 1))])
```

For one matched weight this computes one directed overlap, from 0 to 1. A
symmetric field such as `l_agreement` shares one stored value across both
orientations. Wide aggregate exports show its two orientations. In long
format each exported orientation has two member rows. Exporting both orientations
of this symmetric measurement gives four rows sharing one `aggregate_uid`.

Without `pairs`, a scope containing fewer or more than two checkpoints raises
`ScopeValidationError` before any work runs, naming the scope and the fix:

```python
from diffract.session import ScopeValidationError

with session:  # three checkpoints in scope
    try:
        session.compute.apply("l_overlap")
    except ScopeValidationError as err:
        print(err)
        # The error lists the selected coordinates and suggests a pair filter.
```

The requirement is transitive: `l_agreement` reads `l_overlap`, so requesting
`l_agreement` from a non-pair scope without explicit pairs raises the same error and names the offending
dependency (`l_overlap`).

To compare more than two models, apply the kernel once per pair:

```python
from itertools import combinations

with session:
    checkpoints = ["step_1000", "step_2000", "step_3000"]
    for earlier, later in combinations(checkpoints, 2):
        session.filter(model_ids=[earlier, later]).compute.apply("avg_l_agreement")
```

See the [alignment metrics](../../reference/metrics/alignment.md) reference for the
mathematics of the overlap and agreement fields.

### Contextual field names

A migrated pair with unknown direction yields one row marked
`direction_unknown`. A role-qualified selector excludes that row.

The combined export displays aggregate fields using their model and parameter
names. For pairs it also displays the source and target checkpoints:

```
max_l_agreement [source=m1 (step 0) / w, target=m2 (step 0) / w]
```

Request the base field, e.g. `results.export_aggregates("max_l_agreement")`,
for records with separate model and checkpoint columns.

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

1. Select parameters whose output group needs computation
1. Group the pending parameters into **chunks** sized by an approximate read budget,
   derived from the cache manager's available bytes with headroom
1. For each chunk, prefetch inputs for regular kernels. Streaming kernels read one
   parameter's inputs when it is executed

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
