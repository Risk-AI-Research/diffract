# Filtering Parameters

Target computations and exports to specific parameters.

Filtering is done with `session.filter(...)`, which returns a scoped context
exposing the same namespaces (`models`, `compute`, `results`, `viz`, `utils`)
restricted to the selected parameters and aggregates. The compute and export
methods themselves do **not** take filter keyword arguments — you filter first,
then call them on the returned scope.

## Filter options

`session.filter()` (and the chained `.filter()` on a scope) accept these keyword
arguments:

| Filter        | Type                  | Description                                                   |
| ------------- | --------------------- | ------------------------------------------------------------- |
| `model_ids`   | `list[str]`           | Filter by model identifier (supports regex with `re:` prefix) |
| `param_names` | `list[str]`           | Filter by parameter name (supports regex with `re:` prefix)   |
| `param_types` | `list[ParameterType]` | Filter by parameter type                                      |
| `param_ids`   | `list[str]`           | Filter by exact parameter UID                                 |

## Filter by model

```python
from diffract import Session

session = Session(profile="local")

with session:
    session.models.add(model, model_id="gpt2-small")

    # Compute only for a specific model
    session.filter(model_ids=["gpt2-small"]).compute.apply("frob_norm")

    # Export only from specific models
    df = session.filter(model_ids=["gpt2-small"]).results.export_metrics(
        "frob_norm",
        export_format="pandas",
    )
```

Scoping to **exactly two models** is also how you run the pairwise cross-model
metrics (the `l_overlap` / `l_agreement` alignment family): they compare one layer
between two model versions and raise `ScopeValidationError` on any other scope size.
See [Apply levels](kernels_and_compute.md#apply-levels) for the pairwise workflow.

## Filter by parameter name

Exact match:

```python
with session:
    session.filter(param_names=["layer.0.weight", "layer.1.weight"]).compute.apply(
        "frob_norm"
    )
```

Regex match (prefix with `re:`):

```python
with session:
    # All attention weights
    session.filter(param_names=["re:.*attn.*weight"]).compute.apply("frob_norm")

    # All projection layers
    session.filter(param_names=["re:.*proj$"]).compute.apply("frob_norm")

    # Specific layer range
    session.filter(param_names=["re:layer\\.[0-5]\\..*"]).compute.apply("frob_norm")
```

## Filter by parameter type

Filter with built-in parameter types, or create custom types from strings:

```python
from diffract import ParameterType

with session:
    # Filter by dense layers
    session.filter(param_types=[ParameterType.DENSE]).compute.apply("frob_norm")

    # Create a custom type from a string
    custom_type = ParameterType.from_string("attention")
    session.filter(param_types=[custom_type]).compute.apply("frob_norm")
```

## Filter by UID

For precise targeting when you know exact parameter UIDs:

```python
with session:
    uids = ["abc123", "def456"]
    session.filter(param_ids=uids).compute.apply("frob_norm")
```

## Combining filters

Filters passed together are combined with AND logic:

```python
with session:
    # Attention layers in gpt2-small only
    session.filter(
        model_ids=["gpt2-small"],
        param_names=["re:.*attn.*"],
    ).compute.apply("frob_norm")
```

## Reusing and chaining scopes

`session.filter(...)` returns a scope you can hold onto and reuse, or narrow
further with a chained `.filter(...)`:

```python
with session:
    gpt2 = session.filter(model_ids=["gpt2-small"])
    gpt2.compute.apply("frob_norm", "stable_rank")

    # Narrow the existing scope to attention layers
    attn = gpt2.filter(param_names=["re:.*attn.*"])
    df = attn.results.export_metrics("frob_norm", export_format="pandas")
```
