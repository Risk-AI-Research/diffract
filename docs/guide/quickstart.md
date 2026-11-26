# Quickstart

A 5-minute tour of the Diffract workflow.

## 1. Create a session

```python
from diffract import Session

# RAM-only (no persistence, fastest for experiments)
session = Session(profile="ram")

# Persistent storage (SQLite)
session = Session(profile="local")

# Large models (SQLite + HDF5)
session = Session(profile="hybrid")
```

See [Configuration](../reference/configuration.md) for all profile options and custom config files.

## 2. Add a model

```python
with session:
    session.models.add(model, model_id="my-model")
```

Diffract extracts parameters and stores them. Supported frameworks: PyTorch,
TensorFlow, Flax, ONNX. A plain `dict[str, numpy.ndarray]` of weight matrices
also works, without any framework installed.

(naming-rules)=

### Naming rules

A `model_id` and a parameter name may contain only ASCII letters, digits, and
the characters `_`, `-`, and `.`. This excludes the characters `@`, `[`, `]`,
and `,` (reserved for the aggregate-context grammar) and `/` (the storage-key
path separator). The same alphabet applies wherever identifiers enter the
session — `models.add`, `models.rename`, `results.ingest_aggregates`, and
`utils.merge_other_session`. A violating identifier raises
{py:exc}`~diffract.session.InvalidIdentifierError`, naming the offending
character, rather than silently corrupting downstream keys. Field names passed
to `results.ingest_metrics` are held to the narrower rule that they must not
contain `/` or the other storage-hostile characters `< > : " \ | ? *`.

(parameter-names)=

### Parameter names

The stored parameter name is the **dotted path of the extracted weight
tensor** — for PyTorch, the tensor's `state_dict` key (`encoder.0.weight`);
for a plain dict (including a reloaded safetensors file), the dict key
supplies that path before wrapper normalization. A network added as a live module and the same network added as its
`state_dict` agree on every name they share: the weight paths are
identical, `parameter_overrides` keys are those same paths in every view,
and names survive serialization round-trips. A bare root module (e.g. a
single `nn.Linear` as the whole model) stores the name `weight`.

Diffract removes leading wrapper components from tensor paths. For example,
`module._orig_mod.encoder.0.weight` becomes `encoder.0.weight`.
The recognized prefixes are `module.`, `_orig_mod.`, `_fsdp_wrapped_module.`
and `_checkpoint_wrapped_module.`. An interior component such as the one in
`encoder.module.weight` stays in the name. Parameter metadata records the
extractor and, when a prefix was removed, the original path in `raw_name`.
Keys ending in the component `_extra_state` are logged and skipped.

Each parameter gets a random UID when first extracted. An independent
extraction gets a fresh UID. Merges match the destination model instance,
training step, and tensor path. Tensor paths are
case-sensitive, so `head.weight` and `Head.weight` are distinct.
Merging a matching coordinate preserves its destination UID.

Use one model name across checkpoints and pass an integer training step. A
model added without a step uses step zero. Each model can have one checkpoint
per step. Seeds, variants and stages can have distinct model names.

```python
import numpy as np

early_weights = {"projection.weight": np.eye(2)}
late_weights = {"projection.weight": 2 * np.eye(2)}
session.models.add(early_weights, model_id="encoder", step=2)
session.models.add(late_weights, model_id="encoder", step=10, display_label="epoch 1")
checkpoints = session.filter(model_ids=["encoder"], steps=[2, 10])
final = session.filter(steps="final")
coordinates = session.models.list(layout="coordinates")
steps_by_model = session.models.list(layout="grouped")
```

Parameter listings order steps numerically within each model. The `final`
selector chooses each model's largest stored step. Chained filters intersect
with that selection. Free-form `step_label` text appears in
`models.parameters.list(verbose=True)`.

The default extractors select weights according to the input type.

| Input                | Weights selected                                                                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch `nn.Module`  | The weight of each `nn.Linear` layer.                                                                                                                |
| PyTorch `state_dict` | Two-dimensional tensors with more than one element and a name ending in `weight`, including embedding tables.                                        |
| NumPy dictionary     | Unmasked two-dimensional floating-point arrays with more than one element. The key supplies the name. Auxiliary `_extra_state` entries are excluded. |

For example, a floating-point 2 × 2 NumPy array named `floating.bias` is
extracted. A 2 × 2 integer array named `integer.weight` is reported as
`unsupported` with the reason `No matching handler for ndarray`.

`models.add` returns an extraction report. Each item records its source name,
its outcome, and either the stored name and dtype or the reason for omission.
PyTorch BF16 weights are stored as FP32. The report and the stored parameter
metadata retain the source dtype. To require specific weights and inspect
which of them have a metric, pass their exact source paths to `required_names`.

```python
from diffract import analyze

analysis, coverage = analyze(
    {"projection.weight": np.eye(2)},
    fields=("frob_norm",),
    required_names=("projection.weight",),
    return_coverage=True,
)
stored = {item.stored_name for item in coverage.items if item.status == "extracted"}
measured = {row["param_name"] for row in analysis.details}
assert stored == measured == {"projection.weight"}
```

`coverage.counts` totals the extracted, unsupported, failed and excluded inputs.
If a required name is absent or unreadable, the import fails before storing
weights. The same model and step can then be retried.
When the extractor accepts no parameters, `EmptyExtractionError.coverage`
contains the per-input reasons.

Tied tensors appear once per name present in the input. A safetensors file
carries no tied aliases: `save_file` refuses a model with shared tensors,
and `save_model` keeps one name and drops the rest. A dict reloaded from
such a file therefore presents the single surviving name.

Extraction also records `in_model_idx`, the position of each parameter in
the input's iteration order, counted over every entry the input yields —
including entries that are not extracted, so stored values can skip
numbers. It is presentation metadata for plots — never identity:
serialization formats may reorder keys (safetensors does), which changes
`in_model_idx` but leaves names untouched.

## 3. Compute fields

```python
with session:
    session.compute.apply("frob_norm", "stable_rank")
```

Diffract resolves dependencies and executes kernels. Results are stored automatically.

## 4. Get results

```python
with session:
    results = session.results.export_metrics(
        "frob_norm",
        "stable_rank",
        export_format="dict",  # or "pandas", "polars", "json", "list"
    )
```

## 5. Visualize (optional)

If you installed the `viz` extra:

```python
with session:
    fig = session.viz.box(y="stable_rank", x="model_id")
    fig.show()
```

## Complete example

```python
from diffract import Session

session = Session(profile="local")

with session:
    # Add your model
    session.models.add(model, model_id="gpt2-small")

    # Compute metrics
    session.compute.apply("frob_norm", "stable_rank", "effective_rank")

    # Export results
    df = session.results.export_metrics(
        "frob_norm", "stable_rank", "effective_rank", export_format="pandas"
    )
    print(df.head())
```

## Next steps

- [Overview](what_is_diffract.md) — core concepts explained
- [Configuration](../reference/configuration.md) — profiles, config files, backends
- [Recipes](recipes/index.md) — filtering, exports, storage setup
- [Visualization Showcase](../examples/viz_showcase.md) — plotting examples
