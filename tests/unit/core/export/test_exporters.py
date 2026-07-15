"""Unit tests for result exporting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from diffract.core.export.exporters import ResultExporter
from diffract.core.export.formatters.dict_formatter import DictFormatter
from diffract.core.export.formatters.json_formatter import JsonFormatter

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _Meta:
    uid: str
    name: str
    model_id: str
    ptype: Any
    other_meta: dict[str, Any]


class _PType:
    def __init__(self, name: str) -> None:
        self.name = name


class _Param:
    def __init__(self, meta: _Meta, fields: dict[str, Any]) -> None:
        self.meta = meta
        self._fields = fields

    def has_field(self, field: str) -> bool:
        return field in self._fields

    def get_field(self, field: str, *, auto_prefetch: bool = False) -> Any:
        return self._fields[field]


class _Collection:
    def __init__(
        self, params: Iterable[_Param], *, fail_prefetch: bool = False
    ) -> None:
        self._params = list(params)
        self._fail_prefetch = fail_prefetch

    def __iter__(self):
        return iter(self._params)

    def list_fields_by_uid(
        self, *, parallel: object | None = None
    ) -> dict[str, list[str]]:
        return {p.meta.uid: list(p._fields.keys()) for p in self._params}

    def prefetch_fields(
        self,
        *,
        fields_by_uid: dict[str, list[str]] | None = None,
        fields: list[str] | None = None,
        verify_prefetch: bool = False,
        parallel: object | None = None,
    ) -> bool:
        if self._fail_prefetch:
            raise RuntimeError("prefetch failed")
        if fields_by_uid is not None and fields is not None:
            raise ValueError("fields_by_uid and fields are mutually exclusive.")
        return True


@dataclass(frozen=True, slots=True)
class _AggMeta:
    uid: str
    field_name: str
    context_models: tuple[str, ...]
    context_params: tuple[str, ...]


class _Aggregate:
    def __init__(self, meta: _AggMeta, value: Any) -> None:
        self.meta = meta
        self._value = value

    def has_field(self, field: str) -> bool:
        return field == "value"

    def get_field(self, field: str) -> Any:
        if field == "value":
            return self._value
        raise KeyError(field)


class _AggregateView:
    def __init__(self, aggregates: Iterable[_Aggregate]) -> None:
        self._aggregates = list(aggregates)
        self._filters: dict[str, Any] = {}

    def __iter__(self):
        return iter(self._aggregates)

    def __bool__(self):
        return bool(self._aggregates)

    def filter_by_field_name(self, *fields: str) -> _AggregateView:
        filtered = [a for a in self._aggregates if a.meta.field_name in fields]
        return _AggregateView(filtered)

    def prefetch_fields(
        self,
        *,
        fields: list[str] | None = None,
        parallel: object | None = None,
    ) -> bool:
        return True


def test_export_results_requires_fields() -> None:
    exporter = ResultExporter()
    with pytest.raises(ValueError, match="At least one field"):
        exporter.export_results(parameters=_Collection([]), formatter=DictFormatter())


def test_export_results_dict_format_collects_metadata_and_fields() -> None:
    exporter = ResultExporter()

    meta = _Meta(
        uid="u1",
        name="layer.0.weight",
        model_id="m1",
        ptype=_PType("DENSE"),
        other_meta={"layer_id": 0},
    )
    p = _Param(meta=meta, fields={"a": 1, "b": np.asarray([1, 2])})
    params = _Collection([p], fail_prefetch=True)

    result = exporter.export_results(
        "a", "b", parameters=params, formatter=DictFormatter()
    )
    got = result.scalars
    assert set(got.keys()) == {"u1"}
    assert got["u1"]["metadata"]["name"] == "layer.0.weight"
    assert got["u1"]["metadata"]["model_id"] == "m1"
    assert got["u1"]["metadata"]["parameter_type"] == "DENSE"
    assert got["u1"]["metadata"]["layer_id"] == 0
    assert got["u1"]["fields"]["a"] == 1


def test_export_results_json_serializes_arrays() -> None:
    exporter = ResultExporter()

    meta = _Meta(
        uid="u2",
        name="w",
        model_id="m",
        ptype=_PType("DENSE"),
        other_meta={},
    )
    p = _Param(meta=meta, fields={"x": np.asarray([1.0, 2.0], dtype=np.float32)})

    got = exporter.export_results(
        "x", parameters=_Collection([p]), formatter=JsonFormatter()
    )
    assert '"x"' in got
    assert '"x": [' in got
    assert "1.0" in got
    assert "2.0" in got


def test_export_results_dict_format_with_aggregates() -> None:
    """Test that DictFormatter correctly includes aggregates."""
    exporter = ResultExporter()

    meta = _Meta(
        uid="u1",
        name="layer.0.weight",
        model_id="m1",
        ptype=_PType("DENSE"),
        other_meta={},
    )
    p = _Param(meta=meta, fields={"frob_norm": 1.5})
    params = _Collection([p])

    # Aggregates from aggregate repository
    agg = _Aggregate(
        meta=_AggMeta(
            uid="r1",
            field_name="l_overlap",
            context_models=("m1", "m2"),
            context_params=("layer.0.weight",),
        ),
        value=[[0.9, 0.1]],
    )
    aggregates = _AggregateView([agg])

    result = exporter.export_results(
        "frob_norm",
        "l_overlap",
        parameters=params,
        aggregates=aggregates,
        formatter=DictFormatter(),
    )

    # Check scalars
    assert "u1" in result.scalars
    assert result.scalars["u1"]["fields"]["frob_norm"] == 1.5

    # Check aggregates
    assert len(result.aggregates) == 1
    assert result.aggregates[0]["field"] == "l_overlap"
    assert result.aggregates[0]["context_models"] == ("m1", "m2")
    assert result.aggregates[0]["value"] == [[0.9, 0.1]]


def test_export_results_json_format_includes_aggregates() -> None:
    """Test that JsonFormatter correctly serializes aggregates."""
    exporter = ResultExporter()

    meta = _Meta(
        uid="u1",
        name="w",
        model_id="m",
        ptype=_PType("DENSE"),
        other_meta={},
    )
    p = _Param(meta=meta, fields={"score": 42})

    agg = _Aggregate(
        meta=_AggMeta(
            uid="r1",
            field_name="agg",
            context_models=("m1", "m2"),
            context_params=("p",),
        ),
        value=100,
    )
    aggregates = _AggregateView([agg])

    got = exporter.export_results(
        "score",
        "agg",
        parameters=_Collection([p]),
        aggregates=aggregates,
        formatter=JsonFormatter(),
    )

    assert '"score": 42' in got
    assert '"aggregates":' in got
    assert '"field": "agg"' in got
    assert '"value": 100' in got


def test_export_results_only_scalars_without_aggregates() -> None:
    """Test exporting only scalar fields without aggregates."""
    exporter = ResultExporter()

    meta = _Meta(uid="u1", name="w", model_id="m", ptype=_PType("DENSE"), other_meta={})
    p = _Param(meta=meta, fields={"metric": 1.0, "other": 2.0})

    # No aggregates
    result = exporter.export_results(
        "metric", parameters=_Collection([p]), formatter=DictFormatter()
    )

    # Only metric should be in scalars
    assert result.scalars["u1"]["fields"]["metric"] == 1.0
    assert "other" not in result.scalars["u1"]["fields"]

    # Aggregates should be empty
    assert result.aggregates == []
