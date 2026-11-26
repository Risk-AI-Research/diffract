# Exporting Results

Export computed fields in various formats.

## Metrics vs. aggregates: which exporter

A field's apply level decides which level-specific exporter returns it.
Use `session.compute.describe_field("l_overlap")` to inspect the producer,
dependencies, configuration, and export route.

| Apply level               | Exporter                      |
| ------------------------- | ----------------------------- |
| `PARAMETER`               | `results.export_metrics()`    |
| `IN_MODEL`, `CROSS_MODEL` | `results.export_aggregates()` |

`PARAMETER` fields are per-parameter scalars; `IN_MODEL` and `CROSS_MODEL`
fields are aggregates keyed by the field name and the UIDs of the participating
parameters. Exported labels show the current model and parameter names. Each field's
apply level is listed in the [metric catalog](../../reference/metrics/catalog.md);
the levels themselves are described under
[Apply levels](kernels_and_compute.md#apply-levels).

When the other exporter holds a field, the warning names the method that
serves it. A metrics export can still include parameter metadata.

## Available formats

| Format   | Extra required | Description                |
| -------- | -------------- | -------------------------- |
| `dict`   | —              | Nested Python dictionaries |
| `json`   | —              | JSON string                |
| `pandas` | `pandas`       | pandas DataFrame           |
| `polars` | `polars`       | polars DataFrame           |
| `list`   | —              | List of record dicts       |

## Basic usage

```python
from diffract import Session

session = Session(profile="local")

with session:
    session.compute.apply("frob_norm", "stable_rank")

    # Dictionary (nested by parameter uid)
    results = session.results.export_metrics(
        "frob_norm", "stable_rank", export_format="dict"
    )

    # JSON string
    json_str = session.results.export_metrics(
        "frob_norm", "stable_rank", export_format="json"
    )
```

## Tabular exports (pandas / polars)

Install the required extra:

```bash
uv sync --extra pandas
# or
uv sync --extra polars
```

Export to a DataFrame:

```python
with session:
    df = session.results.export_metrics(
        "frob_norm", "stable_rank", "effective_rank", export_format="pandas"
    )
    print(df.head().to_string())
```

Each row carries the parameter UID, model name, instance UID, numeric step,
optional step label, parameter name and type. Rows sort by model name, numeric
step and parameter name. Custom metadata columns use the `meta_` prefix.
A value whose field name matches an identity column or begins with `meta_`
or `field_` gains a `field_` prefix in a table. For example, a user metric
named `step` appears in `field_step` alongside the numeric checkpoint.

## Long export and joining results

Use `shape="long"` to get parameter metrics and aggregates in the same table.
Every row carries the field name and the `parameter_uid` it refers to.
An aggregate contributes one row per member, or one row with no parameter UID
when it has no members. Rows from the same stored value share an
`aggregate_uid`. Directional rows include source and target model
names, steps, and the member's role. Symmetric results can display both orders
with a shared aggregate UID. A `run_id` links the value to its recorded run
when provenance is available. Use `session.results.describe_run(run_id)` for
the requested configuration and per-entity arithmetic. See the
[SVD lookup example](../../reference/metrics/spectral.md#precision-and-execution-records).

```python
rows = session.results.export(
    "frob_norm", "l_overlap", shape="long", export_format="polars"
)
```

Floating-point scalars occupy `value_scalar` and signed 64-bit integers occupy
`value_integer`. Floating-point arrays with elements up to 64 bits are flat
lists in `value_array` with the original `value_shape` and `value_dtype`.
Boolean and text fields have separate columns. For other stored values,
`value_encoding` describes the content of `value_text`.

Long `list` exports are record lists, `dict` exports map columns to lists,
and `json` serializes the records, using `null` for non-finite numeric values.
In `dict` format with the level-specific exporters,
parameter dictionaries are keyed by UID and aggregates are lists of records.

## Filtering exports

`export_metrics()` itself accepts field names or selectors and `export_format`. To export a
subset, create a filtered scope with `session.filter(...)` and call
`export_metrics()` on it. See [Filtering Parameters](filtering.md) for all
filter options.

```python
with session:
    # Only parameters from a specific model
    df = session.filter(model_ids=["gpt2-small"]).results.export_metrics(
        "frob_norm",
        export_format="pandas",
    )

    # Only attention layers (using regex)
    df = session.filter(param_names=["re:.*attn.*"]).results.export_metrics(
        "frob_norm",
        export_format="pandas",
    )
```

## Working with contextual fields

Request a field by name for every matching aggregate in the current scope.
Use a `FieldSelector` to constrain the members or their operand order.

```python
from diffract import FieldSelector

both = FieldSelector("l_overlap", models=("m1", "m2"), params=("w",))
forward = FieldSelector("l_overlap", roles=("m1", "m2"), steps=(0, 0), params=("w",))
rows = session.results.export_aggregates(both, export_format="list")
forward_rows = session.results.export_aggregates(forward, export_format="list")
```

The input `l_overlap@models[m1,m2]@params[w]` selects the same rows as
`both`. A directed comparison records its source and target. A wide export
displays the current model names and checkpoint labels. Change a checkpoint
caption with `session.models.set_display_label("m1", 0, "initial")`.

## Ingesting and erasing results

`session.results` also covers the reverse direction:

- `ingest_metrics(fields_by_uid, force=False)` — store precomputed per-parameter
  values via a `uid -> {field_name: value}` mapping. Raises on existing fields
  unless `force=True`.
- `ingest_aggregates(aggregates, force=False)` — store precomputed aggregate
  values for parameters already in the session. `context_models` and optional
  `context_params` select the members within the current scope. Add the models
  before importing their results. If any named model or parameter is missing,
  the whole batch raises `SessionError` before storing any values. Renaming a
  model updates the exported labels and preserves the aggregate and its value.
- `erase(*fields, erase_dependent_also=False, erase_all=False)` — remove
  computed field data while keeping the parameters. Field names are resolved
  through the kernel registry, so `erase()` applies to kernel-produced fields.
  Reapplying a parameter kernel rebuilds its output group on each parameter
  with a missing output. Pass `erase_dependent_also=True` to clear downstream
  values before recomputing them.
  [Configuring kernels](kernels_and_compute.md#configuring-kernels)

```python
with session:
    df = session.results.export_metrics("frob_norm", export_format="pandas")
    uid = df["parameter_uid"].iloc[0]

    # Attach an externally computed value to a parameter
    session.results.ingest_metrics({uid: {"external_score": 0.87}})

    # Drop a computed field (parameters stay)
    session.results.erase("frob_norm")
```

## Saving exports

```python
with session:
    df = session.results.export_metrics("frob_norm", export_format="pandas")

    # CSV
    df.to_csv("results.csv", index=False)

    # Parquet (efficient for large datasets)
    df.to_parquet("results.parquet")
```

For polars:

```python
with session:
    df = session.results.export_metrics("frob_norm", export_format="polars")
    df.write_csv("results.csv")
    df.write_parquet("results.parquet")
```
