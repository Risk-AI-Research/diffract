# Extraction Utilities

The `diffract.viz.data.extraction` module provides field extraction utilities
used by all plots. These functions read values out of session entries,
resolving contextual field variants along the way.

All three functions are re-exported from `diffract.viz.data`.

## Entries and contexts

An `Entry` is a `TypedDict` with a `fields` mapping and optional `field_contexts` for aggregates. Plots fetch entries
via `DataProvider` as a `dict[str, Entry]` keyed by uid. To resolve a single
field you first build an `EntryContext`, which extracts the model and parameter
context from an entry's fields:

```python
from diffract.viz.data import Entry, EntryContext

entry: Entry = {
    "fields": {"model_id": "gpt2", "name": "layer.0.weight", "stable_rank": 5.0}
}
ctx = EntryContext.from_entry(entry)
# ctx.fields, ctx.model_id == "gpt2", ctx.parameter_name == "layer.0.weight"
```

Plot selectors accept `name`, `parameter_name` (metric tables), and
`param_name` (long comparison rows) for the parameter name. For example,
`session.viz.scatter(x="step", y="stable_rank", group_by="parameter_name")`
draws one series per parameter name.

## get_field_value

Resolve one field's value from an `EntryContext`, handling contextual variants:

```python
from diffract.viz.data import EntryContext, get_field_value

ctx = EntryContext.from_entry(entry)
get_field_value(ctx, "stable_rank")  # 5.0
```

### Contextual fields

Use the same selectors for plots and result exports. For example, this
reference selects the overlap from the initial checkpoint of model `a` to
step 10 of model `b`:

```python
from diffract import FieldSelector
from diffract.viz.data import FieldRef

ref = FieldRef(FieldSelector("l_overlap", roles=("a", "b"), steps=(0, 10)))
```

The provider attaches the aggregate context to each participating parameter.
A plot slot that matches both directions raises `ValueError` and lists the
role-qualified selectors. A value filter keeps an entry if any matching
direction satisfies its predicate. For other contextual fields, the extractor
prefers the matching model and parameter, then the smallest context.

## get_field_values

Resolve a field for every entry, returning one value per entry:

```python
from diffract.viz.data import get_field_values

values = get_field_values(entries, "stable_rank")
# [5.0, 12.3, ...]  # one per entry, in entries iteration order
```

Internally this calls `get_field_value` on each entry's context.

## get_field_data

Return values plus the detected `DataType` and `DataShape`:

```python
from diffract.viz.data import get_field_data

values, data_type, data_shape = get_field_data(entries, "esd")
# data_type  -> DataType.NUMERIC
# data_shape -> DataShape.VECTOR
```

`DataType` is `NUMERIC` or `CATEGORICAL`; `DataShape` is `SCALAR` or `VECTOR`.
Both live in `diffract.viz.data`. See
[Detection](#detection) for how they are inferred.

## Detection

The `diffract.viz.data.detection` module infers a field's type and shape from
its values (also re-exported from `diffract.viz.data`):

```python
from diffract.viz.data import detect_data_type, detect_data_shape, detect_field_meta

detect_data_type([1, 2, 3])  # DataType.NUMERIC
detect_data_type(["a", "b"])  # DataType.CATEGORICAL
detect_data_shape([[1, 2], [3, 4]])  # DataShape.VECTOR

meta = detect_field_meta([1.0, 2.0])
# meta.data_type == DataType.NUMERIC, meta.data_shape == DataShape.SCALAR
```

## API summary

| Function                           | Description                                                      |
| ---------------------------------- | ---------------------------------------------------------------- |
| `get_field_value(ctx, field)`      | Resolve a field from an `EntryContext`, with contextual matching |
| `get_field_values(entries, field)` | Resolve a field for every entry                                  |
| `get_field_data(entries, field)`   | Values plus detected `DataType` and `DataShape`                  |
| `detect_data_type(values)`         | Infer `DataType` (numeric vs categorical)                        |
| `detect_data_shape(values)`        | Infer `DataShape` (scalar vs vector)                             |
| `detect_field_meta(values)`        | Infer both as a `FieldMeta`                                      |
