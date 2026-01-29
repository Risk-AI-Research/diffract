# Exporting Results

Export computed fields in various formats.

## Available formats

| Format | Extra required | Description |
|--------|----------------|-------------|
| `dict` | — | Nested Python dictionaries |
| `json` | — | JSON string |
| `pandas` | `pandas` | pandas DataFrame |
| `polars` | `polars` | polars DataFrame |

## Basic usage

```python
from diffract import Session

session = Session(profile="local")

with session:
    session.compute("frob_norm", "stable_rank")
    
    # Dictionary (nested by parameter uid)
    results = session.get_results("frob_norm", "stable_rank", export_format="dict")
    
    # JSON string
    json_str = session.get_results("frob_norm", "stable_rank", export_format="json")
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
    df = session.get_results(
        "frob_norm", "stable_rank", "effective_rank",
        export_format="pandas"
    )
    print(df.head())
```

Output:

```
                                    uid       name    model_id  frob_norm  stable_rank
0  a1b2c3d4-e5f6-7890-abcd-ef1234567890  layer.0.weight  my-model   12.345       64.2
1  b2c3d4e5-f6a7-8901-bcde-f12345678901  layer.1.weight  my-model   15.678       48.7
```

## Filtering exports

Apply the same filters as `compute()`:

```python
with session:
    # Only parameters from a specific model
    df = session.get_results(
        "frob_norm",
        export_format="pandas",
        model_ids=["gpt2-small"]
    )
    
    # Only attention layers (using regex)
    df = session.get_results(
        "frob_norm",
        export_format="pandas",
        parameter_names=["re:.*attn.*"]
    )
```

## Working with contextual fields

Aggregated kernels produce contextual field names like `metric@models[m1]@params[...]`. When you request the base name, Diffract matches all contextual variants:

```python
with session:
    # Matches both "overlap" and "overlap@models[m1,m2]@params[...]"
    df = session.get_results("overlap", export_format="pandas")
```

## Saving exports

```python
with session:
    df = session.get_results("frob_norm", export_format="pandas")
    
    # CSV
    df.to_csv("results.csv", index=False)
    
    # Parquet (efficient for large datasets)
    df.to_parquet("results.parquet")
```

For polars:

```python
with session:
    df = session.get_results("frob_norm", export_format="polars")
    df.write_csv("results.csv")
    df.write_parquet("results.parquet")
```
