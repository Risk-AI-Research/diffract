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
