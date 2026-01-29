"""Tests for Torch model parameter extractors."""

from __future__ import annotations

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils

pytestmark = pytest.mark.unit

if not import_utils.is_available("torch"):
    pytest.skip("torch not installed", allow_module_level=True)

nn = import_utils.require("torch.nn")

from diffract.core.data.nn.extractors.base import (  # noqa: E402
    ExtractorOverrides,
)
from diffract.core.data.nn.extractors.torch import (  # noqa: E402
    TorchModuleExtractor,
    TorchStateDictExtractor,
)
from diffract.core.data.nn.params.repository import ParameterRepository  # noqa: E402
from diffract.core.data.nn.params.schema import ParameterType  # noqa: E402


def test_torch_module_extractor_linear_only(storage_cache_metadata: tuple[object, object, object]) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),  # supported DENSE
        nn.ReLU(),  # should be skipped
        nn.Linear(3, 2, bias=False),  # supported DENSE
    )

    extractor = TorchModuleExtractor(model=model)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert len(params) == 2
    for proxy in params:
        assert proxy.meta.ptype == ParameterType.DENSE
        w = proxy.get_field("weights")
        assert isinstance(w, np.ndarray)
        assert w.ndim == 2


def test_torch_state_dict_extractor_dense(storage_cache_metadata: tuple[object, object, object]) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    lin = nn.Linear(5, 4, bias=False)
    state = lin.state_dict()

    extractor = TorchStateDictExtractor(model=state)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert len(params) == 1
    p = params[0]
    assert p.meta.ptype == ParameterType.DENSE
    w = p.get_field("weights")
    assert isinstance(w, np.ndarray)
    assert w.shape == tuple(state["weight"].shape)


def test_overrides_change_name_and_type(storage_cache_metadata: tuple[object, object, object]) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    lin = nn.Linear(3, 3, bias=False)
    extractor = TorchModuleExtractor(
        model=lin,
        overrides=ExtractorOverrides(
            model_id="fixed-model",
            parameter_overrides={
                "": ExtractorOverrides.ParameterOverrides(
                    name="dense0_weight",
                    ptype="DENSE",  # string override path
                )
            },
        ),
    )

    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())
    assert params

    renamed = [p for p in params if p.meta.name == "dense0_weight"]
    assert len(renamed) == 1
    assert renamed[0].meta.model_id == "fixed-model"
    assert renamed[0].meta.ptype == ParameterType.DENSE


def test_skip_unsupported_layers(storage_cache_metadata: tuple[object, object, object]) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    class OnlyUnsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(x)

    model = OnlyUnsupported()
    extractor = TorchModuleExtractor(model=model, skip_not_implemented_types=True)

    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())
    assert params == []
