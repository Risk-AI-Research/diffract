# Filtering Parameters

Target computations and exports to specific parameters.

## Filter options

All filtering methods (`compute()`, `get_results()`) accept these parameters:

| Filter | Type | Description |
|--------|------|-------------|
| `model_ids` | `list[str]` | Filter by model identifier |
| `parameter_names` | `list[str]` | Filter by parameter name (supports regex with `re:` prefix) |
| `parameter_types` | `list[ParameterType]` | Filter by parameter type |
| `parameter_uids` | `list[str]` | Filter by exact UID |

## Filter by model

```python
from diffract import Session

session = Session(profile="local")

with session:
    session.add(model, model_id="gpt2-small")
    
    # Compute only for specific model
    session.compute("frob_norm", model_ids=["gpt2-small"])
    
    # Export only from specific models
    df = session.get_results(
        "frob_norm",
        export_format="pandas",
        model_ids=["gpt2-small"]
    )
```

## Filter by parameter name

Exact match:

```python
with session:
    session.compute("frob_norm", parameter_names=["layer.0.weight", "layer.1.weight"])
```

Regex match (prefix with `re:`):

```python
with session:
    # All attention weights
    session.compute("frob_norm", parameter_names=["re:.*attn.*weight"])
    
    # All projection layers
    session.compute("frob_norm", parameter_names=["re:.*proj$"])
    
    # Specific layer range
    session.compute("frob_norm", parameter_names=["re:layer\\.[0-5]\\..*"])
```

## Filter by parameter type

Parameter types are created dynamically from strings:

```python
from diffract.core.data.nn.parameter import ParameterType

with session:
    # Filter by dense layers
    session.compute("frob_norm", parameter_types=[ParameterType.DENSE])
    
    # Create custom type from string
    custom_type = ParameterType.from_string("attention")
    session.compute("frob_norm", parameter_types=[custom_type])
```

## Filter by UID

For precise targeting when you know exact parameter UIDs:

```python
with session:
    uids = ["abc123", "def456"]
    session.compute("frob_norm", parameter_uids=uids)
```

## Combining filters

Filters are combined with AND logic:

```python
with session:
    # Attention layers in gpt2-small only
    session.compute(
        "frob_norm",
        model_ids=["gpt2-small"],
        parameter_names=["re:.*attn.*"]
    )
```
