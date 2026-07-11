"""Stress tests for Session-level operations."""

from __future__ import annotations

import numpy as np
import pytest

from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.slow]


def test_stress_session_add_many_models(session_with_redis_sqlite) -> None:
    """Stress test: add many models to a session."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_sqlite

    num_models = 50
    params_per_model = 20

    with session:
        for model_idx in range(num_models):
            state_dict = {
                f"layer_{i}.weight": torch.randn(10, 10)
                for i in range(params_per_model)
            }
            session.models.add(state_dict, model_id=f"model_{model_idx}")

        assert len(session.models.list()) == num_models
        assert len(session.models.parameters.list()) == num_models * params_per_model


def test_stress_session_compute_chained_kernels(session_with_redis_sqlite) -> None:
    """Stress test: compute chained kernels (dependencies)."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_sqlite

    @session.compute.kernel(require_fields=("weights",), produce_fields=("w_sum",))
    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    @session.compute.kernel(require_fields=("w_sum",), produce_fields=("w_mean",))
    def w_mean(w_sum: float, *, count: int = 1) -> float:
        return w_sum / count

    @session.compute.kernel(
        require_fields=("w_mean", "w_sum"), produce_fields=("w_variance",)
    )
    def w_variance(w_mean: float, w_sum: float, *, count: int = 1) -> float:
        return (w_sum - w_mean * count) / count

    num_params = 100
    state_dict = {f"layer_{i}.weight": torch.randn(50, 50) for i in range(num_params)}

    with session:
        session.models.add(state_dict, model_id="chain_model")

        # Compute chain (w_sum -> w_mean -> w_variance)
        session.compute.apply("w_variance")

        variance = session.results.export_metrics("w_variance", export_format="dict")
        sums = session.results.export_metrics("w_sum", export_format="dict")
        means = session.results.export_metrics("w_mean", export_format="dict")

    assert len(variance) == num_params
    assert len(sums) == num_params
    assert len(means) == num_params


def test_stress_session_merge_large_sessions(temp_dir) -> None:
    """Stress test: merge large sessions."""
    from diffract.containers import WiringConfiguration, create_main_container
    from diffract.session import Session

    # Create two large sessions
    config_path_a = temp_dir / "merge_a.ini"
    config_path_b = temp_dir / "merge_b.ini"

    for idx, config_path in enumerate([config_path_a, config_path_b]):
        config_content = f"""
[storage]
backend = sqlite

[storage.sqlite]
path = {temp_dir / f"merge_storage_{idx}.db"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / f"merge_metadata_{idx}.db"}

[cache]
backend = simple

[cache.simple]
max_memory_mb = 256

[compute.executor]
max_workers = 4

[nn.extractor]
skip_not_implemented_types = true
"""
        config_path.write_text(config_content.strip() + "\n")

    container_a = create_main_container(config_path_a)
    WiringConfiguration.wire(container_a)
    session_a = Session(container=container_a)

    container_b = create_main_container(config_path_b)
    WiringConfiguration.wire(container_b)
    session_b = Session(container=container_b)

    repo_b = container_b.nn.parameter_repository()

    # Populate session_b with many parameters
    num_params = 500
    proxies = []
    for i in range(num_params):
        meta = ParameterMetadata(
            uid=f"merge_param_{i}",
            name=f"layer_{i}.weight",
            ptype=ParameterType.DENSE,
            model_id=f"model_{i % 10}",
        )
        weights = np.random.randn(20, 20).astype(np.float32)
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repo_b)
        proxy.set_field("weights", weights)
        proxy.set_field("computed_field", float(i))
        proxies.append(proxy)

    WiringConfiguration.wire(container_a)
    session_a.utils.merge_other_session(session_b, verify=True)

    merged_count = sum(1 for _ in session_a.models.parameters.list(verbose=True))
    assert merged_count == num_params

    result = session_a.results.export_metrics("computed_field", export_format="dict")
    assert len(result) >= num_params - 5, (
        f"Expected ~{num_params} results, got {len(result)}"
    )


def test_stress_session_prefetch_operations(temp_dir) -> None:
    """Stress test: mass prefetch into the Redis cache through a view."""
    import os

    from diffract.containers import WiringConfiguration, create_main_container

    redis_host = os.getenv("TEST_REDIS_HOST", "localhost")
    redis_port = int(os.getenv("TEST_REDIS_PORT", "6379"))
    redis_db = int(os.getenv("TEST_REDIS_DB", "15"))

    config_path = temp_dir / "prefetch.ini"
    config_path.write_text(
        f"""
[storage]
backend = sqlite

[storage.sqlite]
path = {temp_dir / "prefetch_storage.db"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / "prefetch_metadata.db"}

[cache]
backend = redis

[cache.redis]
host = {redis_host}
port = {redis_port}
db = {redis_db}
max_memory_mb = 128
key_prefix = diffract:test:prefetch:

[nn.extractor]
skip_not_implemented_types = true
""".strip()
        + "\n"
    )

    try:
        container = create_main_container(config_path)
        WiringConfiguration.wire(container)
        repo = container.nn.parameter_repository()
        cache = container.cache.cache_manager()
    except Exception as e:  # noqa: BLE001
        if "redis" in str(e).lower():
            pytest.skip(f"Redis not available: {e}")
        raise

    num_params = 200
    for i in range(num_params):
        meta = ParameterMetadata(
            uid=f"prefetch_{i}",
            name=f"layer_{i}.weight",
            ptype=ParameterType.DENSE,
            model_id="prefetch_model",
        )
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repo)
        proxy.set_field("weights", np.random.randn(30, 30).astype(np.float32))
        proxy.set_field("field_a", float(i))
        proxy.set_field("field_b", float(i * 2))

    view = repo.create_view()
    success = view.prefetch_fields(fields=["field_a", "field_b"])

    assert success

    for i in range(0, num_params, 10):
        uid = f"prefetch_{i}"
        assert cache.has_field(uid, "field_a")
        assert cache.has_field(uid, "field_b")


def test_stress_session_filter_and_compute(session_with_redis_sqlite) -> None:
    """Stress test: filter parameters and compute on subset."""
    torch = pytest.importorskip("torch")
    session = session_with_redis_sqlite

    @session.compute.kernel(require_fields=("weights",), produce_fields=("w_sum",))
    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    num_models = 20
    params_per_model = 10

    with session:
        for model_idx in range(num_models):
            state_dict = {
                f"layer_{i}.weight": torch.randn(25, 25)
                for i in range(params_per_model)
            }
            session.models.add(state_dict, model_id=f"filter_model_{model_idx}")

    # Compute on subset (first 5 models)
    selected_models = [f"filter_model_{i}" for i in range(5)]
    scoped = session.filter(model_ids=selected_models)
    with scoped:
        scoped.compute.apply("w_sum")
        result = scoped.results.export_metrics("w_sum", export_format="dict")

    assert len(result) == len(selected_models) * params_per_model
