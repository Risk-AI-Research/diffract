# Addressing weights and comparisons

A parameter has three coordinates: a model instance, a training step and a
parameter name. For example, `("pythia-70m-seed1", 1000, "block.0.weight")`
addresses one tensor at checkpoint 1000. Reusing that address in another
session selects the same logical parameter when the store contains those
coordinates. Save the coordinates with analysis code. Diffract also assigns an
opaque UID for joins within a store. Independent imports can assign different
UIDs, and their representation can change between releases.

## Instances and checkpoints

Use one instance name across a run's checkpoints. Give independently trained
runs distinct names, such as `pythia-70m-seed1` and `pythia-70m-seed2`.
Training step is the ordered axis. A seed, architecture variant or dataset
variant belongs in the instance name. The stored tensors describe the model
weights at a checkpoint. Runtime activations and gradients depend on additional
coordinates such as inputs or batches and fall outside this weight schema.

```python
import numpy as np
from diffract import Session

with Session(profile="ram") as session:
    for step in (2, 10):
        session.models.add(
            {"block.0.weight": np.diag([float(step), 1.0])},
            model_id="pythia-70m-seed1",
            step=step,
        )
    assert session.models.list(layout="coordinates") == [
        ("pythia-70m-seed1", 2),
        ("pythia-70m-seed1", 10),
    ]
    final = session.filter(model_ids=["pythia-70m-seed1"], steps="final")
    assert {p["step"] for p in final.models.parameters.list()} == {10}
```

Omitting `step` when adding a model stores it at step zero. The selector
`"final"` resolves to the greatest numeric step for each selected instance.
Adding a later checkpoint can therefore change the selection. Use a numeric
step to reproduce an earlier analysis. Calling `models.rename(old, new)`
changes the model name used for selection. Calling
`models.set_display_label(model_id, step, label)` sets the checkpoint label
displayed in results. Both operations preserve the stored parameter UIDs.

## Scope and direction

Use `session.filter(model_ids=..., steps=..., param_names=...)` to select
parameters. A comparison uses two endpoints. One instance at two steps is a
valid pair. This example computes the overlap for the same layer at both
checkpoints and selects the order from step 2 to step 10.

```python
from diffract import FieldSelector

pair = session.filter(model_ids=["pythia-70m-seed1"], steps=[2, 10])
pair.compute.apply("l_overlap")
forward = FieldSelector(
    "l_overlap",
    roles=("pythia-70m-seed1", "pythia-70m-seed1"),
    steps=(2, 10),
    params=("block.0.weight",),
)
rows = pair.results.export_aggregates(forward, export_format="list")
```

The first slot is the source and the second is the target. For a left or right
singular-vector overlap matrix, the source indexes rows and the target indexes
columns. Reversing the endpoints transposes the matrix. Best-match agreement
takes a maximum over target components for each source component, so reversing
the pair can change the result. Diagonal agreement and its mean are symmetric.
Their two operand orders share a stored value.

Use a plain field name for an unconstrained result and `FieldSelector` to
specify names, steps or roles. A string such as
`l_overlap@models[a,b]@params[block.0.weight]` selects the named members
in either order. The string carries no source or target role. Model and
parameter identifiers use ASCII letters, digits, underscores, hyphens and
periods. Parameter path components between periods must be nonempty.

## Opening an older store

Preview the upgrade with `python -m diffract.migrate /path/to/store --dry-run`.
The migration maps old identifiers to current coordinates and preserves field
values for the records it imports. Old model names are literal by default.
During migration, `--parse-step-suffix` maps a name such as `net_step1000` to
instance `net` at step 1000. A migrated comparison whose direction cannot be
recovered carries a direction-unknown marker.

[Backup and restore instructions](recipes/storage_and_cache.md#store-schema-versions)
