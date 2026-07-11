# Merging Sessions

Copy parameters and computed fields from one session into another with
`session.utils.merge_other_session()`.

## Basic usage

```python
from diffract import Session

session1 = Session(profile="ram")
session2 = Session(profile="ram")

with session1:
    session1.models.add(model1, model_id="model-a")
    session1.compute.apply("frob_norm", "stable_rank")

with session2:
    session2.models.add(model2, model_id="model-b")
    session2.utils.merge_other_session(session1)
```

After the merge, `session2` contains the parameters of both models, and all
fields computed in `session1` are available for the `model-a` parameters:

```python
with session2:
    df = session2.results.export_metrics(
        "frob_norm", "stable_rank", export_format="pandas"
    )
```

Fields that were never computed in `session2` show up as missing values for
the `model-b` rows until you compute them there.

## Selecting fields

Pass `fields=` to restrict which computed fields are copied. Parameters are
merged regardless; only the listed fields come along:

```python
with session2:
    session2.utils.merge_other_session(session1, fields=["frob_norm"])
```

Here `frob_norm` values from `session1` are available in `session2`, while
`stable_rank` is not — it is absent from subsequent exports. With
`fields=None` (the default), all computed fields are merged.

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `fields` | `None` | Computed fields to copy; `None` copies all |
| `verify` | `True` | Check for conflicts and skip duplicate fields |
| `read_budget_bytes` | 512 MiB | Maximum bytes read from the source per chunk |

Merging works across storage backends: for example, merge a RAM session into a
persistent SQLite session to keep the results.
