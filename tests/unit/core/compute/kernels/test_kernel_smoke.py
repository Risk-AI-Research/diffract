"""Smoke test: every registered kernel must be executable end to end.

Guards against kernels that are registered but broken (wrong dependency
wiring, unreachable implementations). Values are not checked here; that
is the job of the per-kernel tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffract.containers import create_main_container
from diffract.core import WEIGHTS_FIELD
from diffract.core.compute.execution.enums import KernelApplyLevel
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.session import Session

pytestmark = pytest.mark.unit

# Snapshot of the built-in registry. A new kernel must be added here so it
# automatically joins this smoke suite; a renamed or removed kernel must be
# reflected here consciously.
EXPECTED_KERNELS = [
    "aspect_ratio",
    "avg_l_agreement",
    "avg_max_l_agreement",
    "avg_max_r_agreement",
    "avg_r_agreement",
    "effective_rank",
    "esd",
    "esd_max",
    "esd_min",
    "esd_rand",
    "esd_rand_max",
    "esd_rand_min",
    "expon_concentration",
    "expon_presence",
    "expon_scale",
    "exponential_fit",
    "frob_norm",
    "greater_dim",
    "hard_rank",
    "l2_norm",
    "l_agreement",
    "l_overlap",
    "log_norm",
    "log_prod_frob_norm",
    "log_prod_spectral_norm",
    "log_spectral_norm",
    "lower_dim",
    "marchenko_pastur_fit",
    "max_l_agreement",
    "max_r_agreement",
    "max_weights_rand_sval",
    "max_weights_sval",
    "min_weights_rand_sval",
    "min_weights_sval",
    "model_pl_alpha_norm",
    "model_tpl_alpha_norm",
    "mp_concentration",
    "mp_ks",
    "mp_num_spikes",
    "mp_presence",
    "mp_soft_rank",
    "mp_sval_max",
    "nuclear_norm",
    "param_norm",
    "pl_alpha_norm",
    "pl_alpha_weighted",
    "pl_concentration",
    "pl_presence",
    "power_law_fit",
    "r_agreement",
    "r_overlap",
    "stable_rank",
    "tpl_alpha_norm",
    "tpl_concentration",
    "tpl_presence",
    "tpl_scale",
    "truncated_power_law_fit",
    "tw_esd_bound",
    "tw_num_spikes",
    "w1_rand_distance",
    "weights_rand",
    "weights_rand_svd",
    "weights_std",
    "weights_svd",
]

# Kernels whose (transitive) dependencies include the truncated power law
# fit, which is by far the slowest computation in the registry.
SLOW_KERNELS = {
    "model_tpl_alpha_norm",
    "tpl_alpha_norm",
    "tpl_concentration",
    "tpl_presence",
    "tpl_scale",
    "truncated_power_law_fit",
}

# Registered only when the accelerated taichi implementation is importable.
OPTIONAL_TAICHI_KERNELS = ["expon_p_value", "pl_p_value", "tpl_p_value"]


def _spiked_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((rows, cols))
    u = rng.standard_normal((rows, 1))
    v = rng.standard_normal((1, cols))
    spike = (u / np.linalg.norm(u)) @ (v / np.linalg.norm(v))
    return (noise + 3.0 * np.sqrt(cols) * spike).astype(np.float32)


@pytest.fixture(scope="module")
def smoke_env(ram_config_factory):
    """Session with two models sharing parameter names and shapes.

    CROSS_MODEL kernels group by parameter name and require exactly two
    parameters; alignment kernels additionally require matching shapes,
    matching the cross-checkpoint use case (same architecture).
    """
    container = create_main_container(ram_config_factory("kernel_smoke"))
    repository = container.nn.parameter_repository()

    for uid, model_id, shape, seed in (
        ("p_m1", "m1", (64, 48), 0),
        ("p_m2", "m2", (64, 48), 1),
    ):
        meta = ParameterMetadata(
            uid=uid,
            name="layer.0.weight",
            ptype=ParameterType.DENSE,
            model_id=model_id,
        )
        proxy = ParameterDataProxy.create_and_store(meta=meta, repository=repository)
        proxy.set_field(WEIGHTS_FIELD, _spiked_matrix(*shape, seed=seed))

    registry = container.compute_singleton.kernel_registry()
    return Session(container=container), registry


def _taichi_fit_available() -> bool:
    from diffract.core.compute.kernels import heavy_tailed

    return heavy_tailed._DIFFRACT_FIT_AVAILABLE


def test_registry_matches_snapshot(smoke_env) -> None:
    """The built-in kernel manifest must match the snapshot exactly.

    Comparison uses the manifest (not the live registry) so that kernels
    registered by other tests into the shared registry cannot break it."""
    from diffract.core.compute.decorator import _DEFAULT_KERNEL_SPECS

    _, registry = smoke_env
    expected = list(EXPECTED_KERNELS)
    if _taichi_fit_available():
        expected = sorted(expected + OPTIONAL_TAICHI_KERNELS)
    builtin_names = sorted({spec["name"] for spec in _DEFAULT_KERNEL_SPECS})
    assert builtin_names == expected
    assert set(expected) <= set(registry.list_kernels())


@pytest.mark.parametrize(
    "kernel_name",
    [
        pytest.param(
            name,
            marks=[pytest.mark.slow]
            if name in SLOW_KERNELS or name in OPTIONAL_TAICHI_KERNELS
            else [],
        )
        for name in EXPECTED_KERNELS + OPTIONAL_TAICHI_KERNELS
    ],
)
def test_kernel_is_executable(smoke_env, kernel_name: str) -> None:
    session, registry = smoke_env
    if kernel_name in OPTIONAL_TAICHI_KERNELS and not _taichi_fit_available():
        pytest.skip("taichi extra not installed")
    field = registry.get_fields_kernel_produce(kernel_name)[0]

    session.compute.apply(field)

    level = registry.get_kernel_apply_level(kernel_name)
    if level is KernelApplyLevel.PARAMETER:
        listed = session.models.parameters.list(verbose=True)
        assert listed
        assert all(field in param["available_fields"] for param in listed)
    else:
        aggregates = session.results.export_aggregates(field, export_format="dict")
        assert aggregates
