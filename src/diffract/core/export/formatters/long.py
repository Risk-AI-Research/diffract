"""Long result rows with explicit aggregate joins and typed value columns."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from functools import partial
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

import diffract.core.utils.imports as import_utils
from diffract.core.storage.serialization import encode_value

if TYPE_CHECKING:
    from diffract.core.export.interface import AggregateData, ResultData


_TEXT_COLUMNS = (
    "entity_kind",
    "parameter_uid",
    "aggregate_uid",
    "run_id",
    "model_id",
    "instance_uid",
    "param_name",
    "step_label",
    "field",
    "member_role",
    "model_source",
    "model_target",
    "value_text",
    "value_dtype",
    "value_encoding",
)
_INT_COLUMNS = ("step", "step_source", "step_target", "value_integer")
_BOOL_COLUMNS = ("symmetric", "direction_unknown", "value_boolean")
_COLUMNS = (
    "entity_kind",
    "parameter_uid",
    "aggregate_uid",
    "run_id",
    "model_id",
    "instance_uid",
    "step",
    "step_label",
    "param_name",
    "field",
    "member_role",
    "model_source",
    "step_source",
    "model_target",
    "step_target",
    "symmetric",
    "direction_unknown",
    "value_scalar",
    "value_integer",
    "value_boolean",
    "value_text",
    "value_array",
    "value_shape",
    "value_dtype",
    "value_encoding",
)
_EMPTY_ROW = dict.fromkeys(_COLUMNS)


def _value_columns(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"value_boolean": value}
    if isinstance(value, int) and -(2**63) <= value < 2**63:
        return {"value_integer": value}
    if isinstance(value, float):
        return {"value_scalar": value}
    if isinstance(value, str):
        return {"value_text": value}
    if (
        isinstance(value, np.ndarray)
        and value.dtype.kind == "f"
        and value.itemsize <= np.dtype(np.float64).itemsize
    ):
        return {
            "value_array": value.reshape(-1).tolist(),
            "value_shape": list(value.shape),
            "value_dtype": value.dtype.str,
        }
    payload, tag = encode_value(value)
    if tag == "json":
        return {"value_text": payload.decode("utf-8"), "value_encoding": tag}
    return {
        "value_text": base64.b64encode(payload).decode("ascii"),
        "value_encoding": f"{tag}-base64",
    }


def _coordinates(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": member["model_id"],
        "instance_uid": member["instance_uid"],
        "step": member["step"],
        "step_label": member["step_label"],
        "param_name": member["name"],
    }


def long_result_rows(
    metrics: ResultData,
    aggregates: AggregateData,
    run_ids: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Collect one row per parameter value or exported aggregate member."""
    rows = []
    for uid, result in metrics.items():
        identity = (
            _EMPTY_ROW
            | _coordinates(result["metadata"])
            | {
                "entity_kind": "parameter",
                "parameter_uid": uid,
            }
        )
        rows.extend(
            identity
            | {"field": field, "run_id": run_ids.get((uid, field))}
            | _value_columns(value)
            for field, value in sorted(result["fields"].items())
        )
    for aggregate in aggregates:
        identity = (
            _EMPTY_ROW
            | {
                "entity_kind": "aggregate",
                "aggregate_uid": aggregate["aggregate_uid"],
                "run_id": aggregate.get("run_id"),
                "field": aggregate["field"],
                "symmetric": aggregate.get("symmetric", False),
                "direction_unknown": aggregate.get("direction_unknown", False),
            }
            | _value_columns(aggregate["value"])
        )
        pair = "model_source" in aggregate
        if pair:
            identity.update(
                (key, aggregate[key])
                for key in (
                    "model_source",
                    "step_source",
                    "model_target",
                    "step_target",
                )
            )
        if not aggregate["member_uids"]:
            rows.append(identity)
            continue
        rows.extend(
            identity
            | _coordinates(member)
            | {
                "parameter_uid": uid,
                "member_role": ("source", "target")[slot] if pair else None,
            }
            for slot, (uid, member) in enumerate(
                zip(aggregate["member_uids"], aggregate["members"], strict=True)
            )
        )
    return rows


def long_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Transpose records into columns in export order, including empty columns."""
    return {key: [row[key] for row in rows] for key in _COLUMNS}


def long_json(rows: list[dict[str, Any]]) -> str:
    """Encode nonfinite numbers as JSON null."""
    records = []
    for row in rows:
        scalar, array = row["value_scalar"], row["value_array"]
        if scalar is not None and not isfinite(scalar):
            scalar = None
        if array is not None and not all(map(isfinite, array)):
            array = [item if isfinite(item) else None for item in array]
        records.append(row | {"value_scalar": scalar, "value_array": array})
    return json.dumps(records, indent=2, allow_nan=False)


def long_pandas(rows: list[dict[str, Any]], *, module: Any) -> Any:
    """Build a frame with nullable scalar columns."""
    columns = long_columns(rows)
    for key in _INT_COLUMNS:
        columns[key] = module.array(columns[key], dtype="Int64")
    for key in _TEXT_COLUMNS:
        columns[key] = module.array(columns[key], dtype="string")
    for key in _BOOL_COLUMNS:
        columns[key] = module.array(columns[key], dtype="boolean")
    columns["value_scalar"] = module.array(columns["value_scalar"], dtype="Float64")
    return module.DataFrame(columns)


def long_polars(rows: list[dict[str, Any]], *, module: Any) -> Any:
    """Build a frame with typed scalar, shape and array columns."""
    columns = long_columns(rows)
    schema = (
        dict.fromkeys(_TEXT_COLUMNS, module.String)
        | dict.fromkeys(_INT_COLUMNS, module.Int64)
        | dict.fromkeys(_BOOL_COLUMNS, module.Boolean)
        | {
            "value_scalar": module.Float64,
            "value_array": module.List(module.Float64),
            "value_shape": module.List(module.Int64),
        }
    )
    return module.DataFrame(columns, schema={key: schema[key] for key in _COLUMNS})


type LongRows = list[dict[str, Any]]
type LongFormatter = Callable[[LongRows], Any]


def _list_rows(rows: LongRows) -> LongRows:
    return rows


_LONG_FORMATTERS: dict[str, LongFormatter] = {
    "list": _list_rows,
    "dict": long_columns,
    "json": long_json,
}
_LONG_TABLE_FORMATTERS = {"pandas": long_pandas, "polars": long_polars}


LONG_FORMATS = tuple(sorted(_LONG_FORMATTERS.keys() | _LONG_TABLE_FORMATTERS.keys()))


def get_long_formatter(name: str) -> LongFormatter:
    """Resolve the long layout and its optional dependency before collecting rows."""
    if name in _LONG_FORMATTERS:
        return _LONG_FORMATTERS[name]
    if name in _LONG_TABLE_FORMATTERS:
        return partial(_LONG_TABLE_FORMATTERS[name], module=import_utils.require(name))
    raise ValueError(f"Long export requires one of: {', '.join(LONG_FORMATS)}.")


def format_long_results(
    metrics: ResultData,
    aggregates: AggregateData,
    run_ids: dict[tuple[str, str], str],
    export_format: str,
) -> Any:
    """Format one row per parameter value or exported aggregate member."""
    format_rows = get_long_formatter(export_format)
    return format_rows(long_result_rows(metrics, aggregates, run_ids))


def format_long_rows(rows: LongRows, export_format: str) -> Any:
    """Convert collected long rows to the requested format."""
    return get_long_formatter(export_format)(rows)
