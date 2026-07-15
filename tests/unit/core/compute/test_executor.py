from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from diffract.core.compute.exceptions import KernelExecutionError
from diffract.core.compute.execution.aggregation import AggregationContext
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.compute.execution.executor import KernelExecutor
from diffract.core.compute.execution.utils import (
    execute_kernel,
    marshal_kernel,
    unmarshal_kernel,
)
from diffract.core.compute.registry import KernelRegistry
from diffract.core.data.nn.aggregates.repository import AggregateRepository
from diffract.core.data.nn.params.interface import IParameterView
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType

# ---------- utilities to build real-ish parameters without storage/cache ----------


class InMemoryStorage:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "InMemoryStorage":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def set_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = "default"
    ) -> None:
        self._data.setdefault(obj_uid, {})[field_name] = value

    def get_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> Any:
        try:
            return self._data[obj_uid][field_name]
        except KeyError as e:
            raise KeyError(f"Missing field {field_name} for {obj_uid}") from e

    def has_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> bool:
        return obj_uid in self._data and field_name in self._data[obj_uid]

    def erase_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> None:
        if obj_uid in self._data and field_name in self._data[obj_uid]:
            del self._data[obj_uid][field_name]

    def erase_obj(self, obj_uid: str, *, table: str = "default") -> None:
        self._data.pop(obj_uid, None)

    def list_fields(
        self, obj_uid: str | None = None, *, table: str = "default"
    ) -> list[str]:
        if obj_uid is None:
            all_fields: set[str] = set()
            for fields in self._data.values():
                all_fields.update(fields.keys())
            return list(all_fields)
        return list(self._data.get(obj_uid, {}).keys())

    def list_objs(self, *, table: str = "default") -> list[str]:
        return list(self._data.keys())

    def list_objs_has_field(
        self, field_name: str, *, table: str = "default"
    ) -> list[str]:
        return [uid for uid, fields in self._data.items() if field_name in fields]

    def get_field_metadata(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> dict[str, Any] | None:
        return None


class InMemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "InMemoryCache":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def set_field(self, obj_uid: str, field_name: str, value: Any) -> None:
        self._data.setdefault(obj_uid, {})[field_name] = value

    def get_field(self, obj_uid: str, field_name: str) -> Any:
        return self._data[obj_uid][field_name]

    def has_field(self, obj_uid: str, field_name: str) -> bool:
        return obj_uid in self._data and field_name in self._data[obj_uid]

    def erase_field(self, obj_uid: str, field_name: str) -> None:
        if obj_uid in self._data and field_name in self._data[obj_uid]:
            del self._data[obj_uid][field_name]

    def get_available_bytes(self) -> int:
        return 256 * 1024 * 1024  # 256MB for tests

    def list_uids(self, *, table: str = "default") -> list[str]:
        return list(self._data.keys())

    def upsert(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = "default"
    ) -> None:
        self._data.setdefault(obj_uid, {})[field_name] = value


class InMemoryMetadataIndex:
    """Simple in-memory metadata index for test doubles."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, dict[str, Any]]] = {}
        self._schemas: dict[str, dict[str, type]] = {}

    def __enter__(self) -> "InMemoryMetadataIndex":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def define_table(
        self,
        table: str,
        columns: dict[str, type],
        indexes: list[str] | None = None,
    ) -> None:
        if table not in self._tables:
            self._tables[table] = {}
        self._schemas[table] = columns

    def insert(self, table: str, uid: str, **fields: Any) -> None:
        if table not in self._tables:
            self._tables[table] = {}
        self._tables[table][uid] = {"uid": uid, **fields}

    def update(self, table: str, uid: str, **fields: Any) -> None:
        if table in self._tables and uid in self._tables[table]:
            self._tables[table][uid].update(fields)

    def upsert(self, table: str, uid: str, **fields: Any) -> None:
        if table not in self._tables:
            self._tables[table] = {}
        if uid in self._tables[table]:
            self._tables[table][uid].update(fields)
        else:
            self._tables[table][uid] = {"uid": uid, **fields}

    def get(self, table: str, uid: str) -> dict[str, Any] | None:
        if table not in self._tables:
            return None
        return self._tables[table].get(uid)

    def query(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        where_in: dict[str, list[Any]] | None = None,
        where_like: dict[str, str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        if table not in self._tables:
            return []

        results = []
        for uid, record in self._tables[table].items():
            match = True

            if where:
                for col, val in where.items():
                    if record.get(col) != val:
                        match = False
                        break

            if match and where_in:
                for col, vals in where_in.items():
                    if col == "uid":
                        if uid not in vals:
                            match = False
                            break
                    elif record.get(col) not in vals:
                        match = False
                        break

            if match:
                results.append(uid)

        if limit is not None:
            results = results[:limit]

        return results

    def delete(self, table: str, uid: str) -> None:
        if table in self._tables:
            self._tables[table].pop(uid, None)

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        if table not in self._tables:
            return 0
        if where is None:
            return len(self._tables[table])
        return len(self.query(table, where=where))

    def distinct(self, table: str, column: str) -> list[Any]:
        if table not in self._tables:
            return []
        values = set()
        for record in self._tables[table].values():
            if column in record:
                values.add(record[column])
        return list(values)

    def list_uids(self, table: str) -> list[str]:
        if table not in self._tables:
            return []
        return list(self._tables[table].keys())

    def clear(self, table: str | None = None) -> None:
        if table is None:
            self._tables.clear()
        elif table in self._tables:
            self._tables[table].clear()

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_repository() -> ParameterRepository:
    metadata_index = InMemoryMetadataIndex()
    return ParameterRepository.initialize(
        InMemoryStorage(), metadata_index, InMemoryCache()
    )


def make_aggregate_repository() -> AggregateRepository:
    metadata_index = InMemoryMetadataIndex()
    return AggregateRepository.initialize(
        InMemoryStorage(), metadata_index, InMemoryCache()
    )


def make_params(
    specs: list[tuple[str, str, dict[str, Any] | None]],
) -> tuple[ParameterRepository, list[ParameterDataProxy]]:
    """Create parameters in a single repository (so prefetch works)."""
    repo = make_repository()
    storage = repo.storage_manager

    proxies: list[ParameterDataProxy] = []
    for name, model_id, initial_fields in specs:
        meta = ParameterMetadata(
            name=name, ptype=ParameterType.DENSE, model_id=model_id
        )
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repo)
        if initial_fields:
            for k, v in initial_fields.items():
                storage.set_field(meta.uid, k, v)
        proxies.append(proxy)

    return repo, proxies


def make_view(
    specs: list[tuple[str, str, dict[str, Any] | None]],
) -> tuple[IParameterView, list[ParameterDataProxy]]:
    repo, proxies = make_params(specs)
    return repo.create_view(), proxies


# ----------------- fixtures -----------------


@pytest.fixture
def registry():
    # Avoid importing default kernels
    with patch("diffract.core.compute.decorator.register_default_kernels"):
        yield KernelRegistry()


@pytest.fixture
def aggregate_repo():
    return make_aggregate_repository()


@pytest.fixture
def executor(registry, aggregate_repo):
    return KernelExecutor(registry=registry, aggregate_repository=aggregate_repo)


# ----------------- AggregationContext tests -----------------


def test_aggregation_context_suffix_and_field():
    ctx = AggregationContext(models=("m2", "m1"), parameters=("p2", "p1"))
    assert ctx.to_field_suffix() == "models[m1,m2]@params[p1,p2]"
    assert ctx.create_field_name("k") == "k@models[m1,m2]@params[p1,p2]"


# ----------------- parameter-level execution -----------------


def test_parameter_level_execute_simple(executor, registry):
    # kernel: double(input) -> out
    def impl(x: int) -> int:
        return x * 2

    registry.register_kernel(
        name="double",
        require_fields=("input",),
        produce_fields=("out",),
        implementation=impl,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(impl)[1],
    )

    params, proxies = make_view(
        [
            ("w1", "m1", {"input": 3}),
            ("w2", "m1", {"input": 5}),
        ]
    )
    p1, p2 = proxies

    executor.execute("double", params)

    assert p1.get_field("out") == 6
    assert p2.get_field("out") == 10


def test_parameter_level_skips_already_computed(executor, registry):
    def impl(x: int) -> int:
        return x * 2

    registry.register_kernel(
        name="double",
        require_fields=("input",),
        produce_fields=("double",),
        implementation=impl,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(impl)[1],
    )

    params, proxies = make_view(
        [("w", "m", {"input": 2, "double": 123})]
    )  # already computed
    (p,) = proxies
    executor.execute("double", params)
    # value should remain untouched
    assert p.get_field("double") == 123


# ----------------- dependencies resolution path -----------------


def test_execute_resolves_dependencies(executor, registry):
    # dep: base(x) -> y ; final: plus10(y) -> z
    def base(x: int) -> int:
        return x + 1

    def final(y: int) -> int:
        return y + 10

    registry.register_kernel(
        name="base",
        require_fields=("in",),
        produce_fields=("y",),
        implementation=base,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(base)[1],
    )
    registry.register_kernel(
        name="final",
        require_fields=("y",),
        produce_fields=("z",),
        implementation=final,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(final)[1],
    )

    # Ensure resolve_dependencies returns chain (name-based)
    deps = registry.resolve_dependencies("final")
    assert "base" in deps
    assert "final" in deps

    params, proxies = make_view([("w", "m", {"in": 1})])
    (p,) = proxies

    executor.execute("final", params)
    assert p.get_field("z") == (1 + 1) + 10


# ----------------- aggregation: IN_MODEL -----------------


def test_in_model_aggregation(executor, registry, aggregate_repo):
    # per-parameter kernel: value -> v
    def value(x: int) -> int:
        return x

    # in-model aggregate: log_mean(vs) -> agg
    def log_mean(vs: tuple[int, ...]) -> float:
        arr = np.array(vs)
        return float(np.mean(np.log(arr + 1e-9)))

    registry.register_kernel(
        name="value",
        require_fields=("x",),
        produce_fields=("v",),
        implementation=value,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(value)[1],
    )

    registry.register_kernel(
        name="agg_log_mean",
        require_fields=("v",),
        produce_fields=("agg",),
        implementation=log_mean,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(log_mean)[1],
    )

    params, _ = make_view(
        [
            ("w1", "m1", {"x": 1}),
            ("w2", "m1", {"x": 3}),
            ("w3", "m2", {"x": 7}),
        ]
    )

    executor.execute("value", params)
    executor.execute("agg_log_mean", params)

    # Results stored in AggregateRepository
    agg1 = aggregate_repo.get_or_create(
        field_name="agg",
        context_models=("m1",),
        context_params=("w1", "w2"),
    )
    assert agg1.get_field("value") == pytest.approx(
        float(np.mean(np.log(np.array([1, 3]))))
    )

    agg2 = aggregate_repo.get_or_create(
        field_name="agg",
        context_models=("m2",),
        context_params=("w3",),
    )
    assert agg2.get_field("value") == pytest.approx(
        float(np.mean(np.log(np.array([7]))))
    )


# ----------------- aggregation: CROSS_MODEL -----------------


def test_cross_model_aggregation(executor, registry, aggregate_repo):
    # per-parameter kernel: project -> score
    def score(x: int) -> int:
        return x * 2

    # cross-model aggregate for same parameter name: sum(scores) -> total
    def sum_scores(vs: tuple[int, ...]) -> int:
        return int(np.sum(np.array(vs)).item())

    registry.register_kernel(
        name="score",
        require_fields=("x",),
        produce_fields=("score",),
        implementation=score,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(score)[1],
    )

    registry.register_kernel(
        name="total_score",
        require_fields=("score",),
        produce_fields=("total",),
        implementation=sum_scores,
        apply_level=KernelApplyLevel.CROSS_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(sum_scores)[1],
    )

    params, _ = make_view(
        [
            ("p", "m1", {"x": 2}),
            ("p", "m2", {"x": 4}),
            ("q", "m3", {"x": 8}),  # different parameter name -> excluded
        ]
    )

    executor.execute("score", params)
    executor.execute("total_score", params)

    # Results stored in AggregateRepository
    agg_p = aggregate_repo.get_or_create(
        field_name="total",
        context_models=("m1", "m2"),
        context_params=("p",),
    )
    assert agg_p.get_field("value") == 2 * 2 + 4 * 2

    agg_q = aggregate_repo.get_or_create(
        field_name="total",
        context_models=("m3",),
        context_params=("q",),
    )
    assert agg_q.get_field("value") == 8 * 2


# ----------------- contextual fields depending on contextual fields -----------------


def test_contextual_field_depends_on_parameter_field(
    executor, registry, aggregate_repo
):
    """Test IN_MODEL aggregation produces contextual field from parameter fields."""

    def param_value(x: int) -> int:
        return x * 2

    def agg_sum(vs: tuple[int, ...]) -> int:
        return sum(vs)

    registry.register_kernel(
        name="param_value",
        require_fields=("x",),
        produce_fields=("v",),
        implementation=param_value,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(param_value)[1],
    )

    registry.register_kernel(
        name="agg_sum",
        require_fields=("v",),
        produce_fields=("total",),
        implementation=agg_sum,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(agg_sum)[1],
    )

    params, _ = make_view(
        [
            ("w1", "m1", {"x": 1}),
            ("w2", "m1", {"x": 2}),
            ("w3", "m1", {"x": 3}),
        ]
    )

    executor.execute("param_value", params)
    executor.execute("agg_sum", params)

    # Verify aggregate is created with correct context
    agg = aggregate_repo.get_or_create(
        field_name="total",
        context_models=("m1",),
        context_params=("w1", "w2", "w3"),
    )
    # (1*2 + 2*2 + 3*2) = 12
    assert agg.get_field("value") == 12


def test_contextual_field_depends_on_contextual_field(
    executor, registry, aggregate_repo
):
    """Test aggregation kernel that depends on another aggregation kernel's output.

    This tests the case where:
    1. First kernel: PARAMETER level computes per-param values
    2. Second kernel: IN_MODEL aggregates to produce contextual field A
    3. Third kernel: IN_MODEL uses contextual field A to produce contextual field B
    """

    def param_score(x: int) -> int:
        return x

    def agg_mean(vs: tuple[int, ...]) -> float:
        return sum(vs) / len(vs)

    def normalize_by_mean(vs: tuple[int, ...], mean_val: float) -> list[float]:
        """Normalize values by the mean (requires contextual mean_val)."""
        return [v / mean_val for v in vs]

    registry.register_kernel(
        name="score",
        require_fields=("x",),
        produce_fields=("score",),
        implementation=param_score,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(param_score)[1],
    )

    registry.register_kernel(
        name="mean_score",
        require_fields=("score",),
        produce_fields=("mean",),
        implementation=agg_mean,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(agg_mean)[1],
    )

    registry.register_kernel(
        name="normalize",
        require_fields=("score", "mean"),
        produce_fields=("normalized",),
        implementation=normalize_by_mean,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(normalize_by_mean)[1],
    )

    params, _ = make_view(
        [
            ("w1", "m1", {"x": 2}),
            ("w2", "m1", {"x": 4}),
            ("w3", "m1", {"x": 6}),
        ]
    )

    # Execute the full chain
    executor.execute("score", params)
    executor.execute("mean_score", params)
    executor.execute("normalize", params)

    # Verify mean was computed correctly: (2+4+6)/3 = 4.0
    agg_mean = aggregate_repo.get_or_create(
        field_name="mean",
        context_models=("m1",),
        context_params=("w1", "w2", "w3"),
    )
    assert agg_mean.get_field("value") == pytest.approx(4.0)

    # Verify normalized values: [2/4, 4/4, 6/4] = [0.5, 1.0, 1.5]
    agg_norm = aggregate_repo.get_or_create(
        field_name="normalized",
        context_models=("m1",),
        context_params=("w1", "w2", "w3"),
    )
    assert agg_norm.get_field("value") == pytest.approx([0.5, 1.0, 1.5])


def test_chained_in_model_aggregations(executor, registry, aggregate_repo):
    """Test chained IN_MODEL aggregations where second depends on first.

    Scenario:
    1. First IN_MODEL kernel computes sum
    2. Second IN_MODEL kernel computes ratio using the sum (contextual field)
    """

    def param_value(x: int) -> int:
        return x

    def model_sum(vs: tuple[int, ...]) -> int:
        return sum(vs)

    def ratio_to_sum(vs: tuple[int, ...], total: int) -> list[float]:
        """Compute ratio of each value to the sum."""
        return [v / total for v in vs]

    registry.register_kernel(
        name="value",
        require_fields=("x",),
        produce_fields=("v",),
        implementation=param_value,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(param_value)[1],
    )

    registry.register_kernel(
        name="model_sum",
        require_fields=("v",),
        produce_fields=("total",),
        implementation=model_sum,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(model_sum)[1],
    )

    # Second IN_MODEL depends on first IN_MODEL's contextual output
    registry.register_kernel(
        name="compute_ratio",
        require_fields=("v", "total"),
        produce_fields=("ratio",),
        implementation=ratio_to_sum,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(ratio_to_sum)[1],
    )

    params, _ = make_view(
        [
            ("w1", "m1", {"x": 1}),
            ("w2", "m1", {"x": 2}),
            ("w3", "m1", {"x": 7}),  # total = 10
        ]
    )

    executor.execute("value", params)
    executor.execute("model_sum", params)
    executor.execute("compute_ratio", params)

    # Verify sum
    agg_sum = aggregate_repo.get_or_create(
        field_name="total",
        context_models=("m1",),
        context_params=("w1", "w2", "w3"),
    )
    assert agg_sum.get_field("value") == 10

    # Verify ratios: [1/10, 2/10, 7/10]
    agg_ratio = aggregate_repo.get_or_create(
        field_name="ratio",
        context_models=("m1",),
        context_params=("w1", "w2", "w3"),
    )
    assert agg_ratio.get_field("value") == pytest.approx([0.1, 0.2, 0.7])


def test_multiple_models_in_model_aggregation(executor, registry, aggregate_repo):
    """Test IN_MODEL aggregation with multiple models produces separate aggregates."""

    def param_value(x: int) -> int:
        return x * 2

    def model_max(vs: tuple[int, ...]) -> int:
        return max(vs)

    registry.register_kernel(
        name="double",
        require_fields=("x",),
        produce_fields=("v",),
        implementation=param_value,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(param_value)[1],
    )

    registry.register_kernel(
        name="model_max",
        require_fields=("v",),
        produce_fields=("max_val",),
        implementation=model_max,
        apply_level=KernelApplyLevel.IN_MODEL,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(model_max)[1],
    )

    params, _ = make_view(
        [
            ("w1", "m1", {"x": 5}),
            ("w2", "m1", {"x": 3}),
            ("w3", "m2", {"x": 10}),
            ("w4", "m2", {"x": 2}),
        ]
    )

    executor.execute("double", params)
    executor.execute("model_max", params)

    # Model m1: max(5*2, 3*2) = max(10, 6) = 10
    agg_m1 = aggregate_repo.get_or_create(
        field_name="max_val",
        context_models=("m1",),
        context_params=("w1", "w2"),
    )
    assert agg_m1.get_field("value") == 10

    # Model m2: max(10*2, 2*2) = max(20, 4) = 20
    agg_m2 = aggregate_repo.get_or_create(
        field_name="max_val",
        context_models=("m2",),
        context_params=("w3", "w4"),
    )
    assert agg_m2.get_field("value") == 20


# ----------------- prefetch, chunking, and batching -----------------


def test_prefetch_success_goes_batch(executor, registry):
    def impl(a: int) -> int:
        return a + 1

    registry.register_kernel(
        name="k",
        require_fields=("a",),
        produce_fields=("b",),
        implementation=impl,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=registry._split_signature(impl)[1],
    )

    view, _ = make_view([("w", "m", {"a": 1})])

    # Spy on batch path via the parameter runner
    from diffract.core.compute.execution.parameter_runner import ParameterKernelRunner

    with patch.object(ParameterKernelRunner, "_execute_batch") as spy:
        executor._parameter_runner.run("k", view)
        spy.assert_called_once()


# ----------------- restrictions & error handling ----------------


def test_execute_kernel_success_and_error():
    def ok(x):
        return x + 1

    assert execute_kernel("k", ok, (1,)) == 2

    def boom():
        raise ValueError("bad")

    with pytest.raises(KernelExecutionError):
        execute_kernel("k", boom, ())


# ----------------- multiprocessing marshalling -----------------


def test_marshal_unmarshal_closure_function_roundtrip():
    factor = 3

    def mul(x):
        return x * factor

    blob = marshal_kernel(mul)
    fn = unmarshal_kernel(blob)
    assert fn(4) == 12
