"""Tests for Flax parameter extractor."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = pytest.mark.unit


def test_flax_params_extractor_dense_only(
    storage_cache_metadata: tuple[object, object, object],
) -> None:
    if not (import_utils.is_available("flax") and import_utils.is_available("jax")):
        pytest.skip("flax/jax not installed")

    flax = import_utils.require("flax")
    jax = import_utils.require("jax")
    jnp = import_utils.require("jax.numpy")

    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    class MLP(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, x):
            x = flax.linen.Dense(3, use_bias=False)(x)
            x = flax.linen.relu(x)
            return flax.linen.Dense(2, use_bias=False)(x)

    model = MLP()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 4), dtype=jnp.float32))

    from diffract.core.data.nn.extractors.flax import (
        FlaxParamsExtractor,
    )

    extractor = FlaxParamsExtractor(model=variables)
    extractor.extract_parameters(parameter_repository=repo)
    params = repo.create_view()

    got = list(params)
    assert len(got) == 2
    assert all(p.meta.ptype == ParameterType.DENSE for p in got)
    assert all(p.meta.name.endswith(".kernel") for p in got)

    for p in got:
        w = p.get_field("weights")
        assert isinstance(w, np.ndarray)
        assert w.ndim == 2


def test_flax_params_extractor_dense_only_without_flax(
    storage_cache_metadata: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers Flax extractor logic without the real flax/jax packages."""

    original_is_available = import_utils.is_available

    def fake_is_available(package: str) -> bool:
        if package in ("flax", "jax"):
            return True
        return original_is_available(package)

    monkeypatch.setattr(import_utils, "is_available", fake_is_available)

    flax_mod = types.ModuleType("flax")
    jax_mod = types.ModuleType("jax")

    def device_get(x: object) -> object:
        return x

    jax_mod.device_get = device_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flax", flax_mod)
    monkeypatch.setitem(sys.modules, "jax", jax_mod)

    import diffract.core.data.nn.extractors.flax as flax_extractor
    import diffract.core.data.nn.extractors.handlers as handlers_pkg
    from diffract.core.data.nn.extractors.handlers import (
        flax_handlers,
    )

    importlib.reload(flax_handlers)
    importlib.reload(handlers_pkg)
    importlib.reload(flax_extractor)

    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    params = {
        "Dense_0": {
            "kernel": np.ones((4, 3), dtype=np.float32),
            "bias": np.zeros((3,), dtype=np.float32),
        },
        "Other": {"scale": np.ones((3,), dtype=np.float32)},
    }

    extractor = flax_extractor.FlaxParamsExtractor(model=params)
    extractor.extract_parameters(parameter_repository=repo)
    extracted = list(repo.create_view())

    assert len(extracted) == 1
    p = extracted[0]
    assert p.meta.ptype == ParameterType.DENSE
    assert p.meta.name.endswith(".kernel")
    w = p.get_field("weights")
    assert isinstance(w, np.ndarray)
    assert w.shape == (4, 3)
