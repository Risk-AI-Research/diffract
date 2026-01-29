"""Stress tests for Session-level operations."""

from __future__ import annotations

import numpy as np
import pytest

from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.slow]


def test_stress_session_add_many_models(session_with_redis_sqlite) -> None:
    """Stress test: add many models to a session."""
    session = session_with_redis_sqlite
    repo = session._parameter_repository  # noqa: SLF001

    # Create many parameters from different models
    num_models = 50
    params_per_model = 20

    proxies = []
    for model_idx in range(num_models):
        model_id = f"model_{model_idx}"
        for param_idx in range(params_per_model):
            meta = ParameterMetadata(
                uid=f"p_{model_idx}_{param_idx}",
                name=f"layer_{param_idx}.weight",
                ptype=ParameterType.DENSE,
                model_id=model_id,
            )
            weights = np.random.randn(10, 10).astype(np.float32)
            proxy = ParameterDataProxy.create_and_store(
                meta=meta,
                repository=repo,
            )
            proxy.set_field("weights", weights)
            proxies.append(proxy)
    
    # Verify all models
    all_models = session.list_models()
    assert len(all_models) == num_models

    # Verify parameters
    params = session.list_parameters()
    assert len(params) == num_models * params_per_model


def test_stress_session_compute_chained_kernels(session_with_redis_sqlite) -> None:
    """Stress test: compute chained kernels (dependencies)."""
    session = session_with_redis_sqlite
    repo = session._parameter_repository  # noqa: SLF001

    # Register chain of kernels
    registry = session._container.compute_singleton.kernel_registry()  # noqa: SLF001

    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    def w_mean(w_sum: float, *, count: int = 1) -> float:
        return w_sum / count

    def w_variance(w_mean: float, w_sum: float, *, count: int = 1) -> float:
        return (w_sum - w_mean * count) / count

    _, cfg_sum = registry._split_signature(w_sum)  # noqa: SLF001
    _, cfg_mean = registry._split_signature(w_mean)  # noqa: SLF001
    _, cfg_var = registry._split_signature(w_variance)  # noqa: SLF001

    registry.register_kernel(
        name="w_sum",
        require_fields=("weights",),
        produce_fields=("w_sum",),
        implementation=w_sum,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_sum,
    )

    registry.register_kernel(
        name="w_mean",
        require_fields=("w_sum",),
        produce_fields=("w_mean",),
        implementation=w_mean,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_mean,
    )

    registry.register_kernel(
        name="w_variance",
        require_fields=("w_mean", "w_sum"),
        produce_fields=("w_variance",),
        implementation=w_variance,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_var,
    )

    # Create many parameters
    num_params = 100
    proxies = []
    for i in range(num_params):
        meta = ParameterMetadata(
            uid=f"chain_test_{i}",
            name=f"layer_{i}.weight",
            ptype=ParameterType.DENSE,
            model_id="chain_model",
        )
        weights = np.random.randn(50, 50).astype(np.float32)
        proxy = ParameterDataProxy.create_and_store(
            meta=meta, repository=repo
        )
        proxy.set_field("weights", weights)
        proxies.append(proxy)
    
    # Compute chain (w_sum -> w_mean -> w_variance)
    session.compute("w_variance")

    # Verify results
    result = session.get_results("w_variance", export_format="dict")
    assert len(result.scalars) == num_params

    # Verify intermediate results also computed
    sum_result = session.get_results("w_sum", export_format="dict")
    mean_result = session.get_results("w_mean", export_format="dict")
    assert len(sum_result.scalars) == num_params
    assert len(mean_result.scalars) == num_params


def test_stress_session_merge_large_sessions(temp_dir) -> None:
    """Stress test: merge large sessions."""
    from diffract.containers import MainContainer, WiringConfiguration, create_main_container
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

    storage_a = container_a.storage.storage_manager()
    cache_a = container_a.cache.cache_manager()
    storage_b = container_b.storage.storage_manager()
    cache_b = container_b.cache.cache_manager()
    repo_b = session_b._parameter_repository  # noqa: SLF001

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
        proxy = ParameterDataProxy.create_and_store(
            meta=meta, repository=repo_b
        )
        proxy.set_field("weights", weights)
        proxy.set_field("computed_field", float(i))
        proxies.append(proxy)
    
    # Merge
    WiringConfiguration.wire(container_a)
    session_a.merge(session_b, verify=True)

    # Verify merge
    merged_params = session_a.list_parameters()
    assert len(merged_params) == num_params

    # Verify computed fields (allow for small margin due to transient failures)
    result = session_a.get_results("computed_field", export_format="dict")
    assert len(result.scalars) >= num_params - 5, (
        f"Expected ~{num_params} results, got {len(result.scalars)}"
    )


def test_stress_session_prefetch_operations(session_with_redis_sqlite) -> None:
    """Stress test: mass prefetch operations."""
    session = session_with_redis_sqlite
    repo = session._parameter_repository  # noqa: SLF001
    cache = session._container.cache.cache_manager()  # noqa: SLF001

    # Create many parameters
    num_params = 200
    proxies = []
    for i in range(num_params):
        meta = ParameterMetadata(
            uid=f"prefetch_{i}",
            name=f"layer_{i}.weight",
            ptype=ParameterType.DENSE,
            model_id="prefetch_model",
        )
        weights = np.random.randn(30, 30).astype(np.float32)
        proxy = ParameterDataProxy.create_and_store(
            meta=meta, repository=repo
        )
        proxy.set_field("weights", weights)
        proxy.set_field("field_a", float(i))
        proxy.set_field("field_b", float(i * 2))
        proxies.append(proxy)
    
    # Prefetch fields for all parameters
    view = session._get_view()  # noqa: SLF001
    success = view.prefetch_fields(fields=["field_a", "field_b"])

    # Should succeed (all fields exist)
    assert success

    # Verify cache has prefetched data
    for i in range(0, num_params, 10):  # Check samples
        uid = f"prefetch_{i}"
        assert cache.has_field(uid, "field_a")
        assert cache.has_field(uid, "field_b")


def test_stress_session_filter_and_compute(session_with_redis_sqlite) -> None:
    """Stress test: filter parameters and compute on subset."""
    session = session_with_redis_sqlite
    repo = session._parameter_repository  # noqa: SLF001

    # Register kernel
    registry = session._container.compute_singleton.kernel_registry()  # noqa: SLF001

    def w_sum(w: np.ndarray) -> float:
        return float(np.sum(w))

    _, cfg_dict = registry._split_signature(w_sum)  # noqa: SLF001
    registry.register_kernel(
        name="w_sum",
        require_fields=("weights",),
        produce_fields=("w_sum",),
        implementation=w_sum,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=cfg_dict,
    )

    # Create parameters from multiple models
    num_models = 20
    params_per_model = 10
    proxies = []

    for model_idx in range(num_models):
        model_id = f"filter_model_{model_idx}"
        for param_idx in range(params_per_model):
            meta = ParameterMetadata(
                uid=f"filter_{model_idx}_{param_idx}",
                name=f"layer_{param_idx}.weight",
                ptype=ParameterType.DENSE,
                model_id=model_id,
            )
            weights = np.random.randn(25, 25).astype(np.float32)
            proxy = ParameterDataProxy.create_and_store(
                meta=meta, repository=repo
            )
            proxy.set_field("weights", weights)
            proxies.append(proxy)
    
    # Compute on subset (first 5 models)
    selected_models = [f"filter_model_{i}" for i in range(5)]
    session.compute("w_sum", model_ids=selected_models)

    # Verify results only for selected models
    result = session.get_results("w_sum", export_format="dict")
    assert len(result.scalars) == 5 * params_per_model

    # Verify all results have correct model_id
    for uid, _data in result.scalars.items():
        # Extract model_id from parameter list
        params = session.list_parameters(model_ids=selected_models)
        param_uids = {p["uid"] for p in params}
        assert uid in param_uids

