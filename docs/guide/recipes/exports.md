# Exporting Results

Export computed fields in various formats.

## Metrics vs. aggregates: which exporter

A field's apply level decides which exporter returns it:

| Apply level               | Exporter                      |
| ------------------------- | ----------------------------- |
| `PARAMETER`               | `results.export_metrics()`    |
| `IN_MODEL`, `CROSS_MODEL` | `results.export_aggregates()` |

`PARAMETER` fields are per-parameter scalars; `IN_MODEL` and `CROSS_MODEL`
fields are aggregates keyed by their model and parameter context. Each field's
apply level is listed in the [metric catalog](../../reference/metrics/catalog.md);
the levels themselves are described under
[Apply levels](kernels_and_compute.md#apply-levels).

Requesting a field from the wrong exporter returns nothing (with a warning)
rather than raising, so an unexpectedly empty result usually means the field is
served by the other exporter.

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

Output:

```
  parameter_uid  model_id parameter_name parameter_type  meta_in_model_idx meta_torch_dtype meta_original_model_id  frob_norm  stable_rank  effective_rank
0      b5c80064  my-model            fc1          DENSE                  1    torch.float32               4f0ec9e0   4.690019    11.069382       29.816245
1      ef6d4a0c  my-model            fc2          DENSE                  2    torch.float32               4f0ec9e0   2.277429     8.742566       15.563562
```

Each row is one parameter: identity columns (`parameter_uid`, `model_id`,
`parameter_name`, `parameter_type`), metadata columns prefixed with `meta_`,
and one column per requested field.

## Filtering exports

`export_metrics()` itself only takes field names and `export_format`. To export a
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

Aggregated kernels produce contextual field names like `metric@models[m1]@params[...]`. When you request the base name, Diffract matches all contextual variants:

```python
with session:
    # Matches both "overlap" and "overlap@models[m1,m2]@params[...]"
    df = session.results.export_aggregates("overlap", export_format="pandas")
```

## Ingesting and erasing results

`session.results` also covers the reverse direction:

- `ingest_metrics(fields_by_uid, force=False)` — store precomputed per-parameter
  values via a `uid -> {field_name: value}` mapping. Raises on existing fields
  unless `force=True`.
- `ingest_aggregates(aggregates, force=False)` — store precomputed aggregate
  values, each identified by `field_name`, `context_models`, and optional
  `context_params`.
- `erase(*fields, erase_dependent_also=False, erase_all=False)` — remove
  computed field data while keeping the parameters. Field names are resolved
  through the kernel registry, so `erase()` applies to kernel-produced fields.
  Erase a multi-output kernel's fields as a group, and stale dependents
  explicitly (`erase_dependent_also=True`) — see
  [Configuring kernels](kernels_and_compute.md#configuring-kernels) for why
  `apply` cannot restore a partially erased produce group.

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
