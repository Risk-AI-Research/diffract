"""Tests for NumPy dictionary parameter extractors."""

from __future__ import annotations

import numpy as np
import pytest

from diffract.core.data.nn.extractors.factory import create_extractor
from diffract.core.data.nn.extractors.handlers.numpy_handlers import NumpyDenseHandler
from diffract.core.data.nn.extractors.numpy import NumpyDictExtractor
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = pytest.mark.unit


def test_numpy_dict_extractor_dense(
    storage_cache_metadata: tuple[object, object, object],
) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    rng = np.random.default_rng(0)
    weights = {
        "encoder.weight": rng.random((6, 4)),
        "decoder.weight": rng.random((4, 6), dtype=np.float32),
    }

    extractor = NumpyDictExtractor(model=weights)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert len(params) == 2
    for proxy in params:
        assert proxy.meta.ptype == ParameterType.DENSE
        w = proxy.get_field("weights")
        assert isinstance(w, np.ndarray)
        assert w.shape == weights[proxy.meta.name].shape
        expected_dtype = str(weights[proxy.meta.name].dtype)
        assert proxy.meta.other_meta["numpy_dtype"] == expected_dtype


def test_numpy_dict_extractor_skips_unsupported_arrays(
    storage_cache_metadata: tuple[object, object, object],
) -> None:
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    weights = {
        "bias": np.zeros(8),
        "embedding": np.zeros((2, 3, 4)),
        "token_ids": np.zeros((4, 4), dtype=np.int64),
        "scale": np.array(1.0),
        "kernel": np.ones((4, 4)),
    }

    extractor = NumpyDictExtractor(model=weights)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert [p.meta.name for p in params] == ["kernel"]


def test_numpy_dense_handler_accepts_only_2d_float() -> None:
    handler = NumpyDenseHandler()

    assert handler.can_handle(np.ones((3, 3)), "w")
    assert handler.can_handle(np.ones((3, 3), dtype=np.float32), "w")
    assert not handler.can_handle(np.ones(3), "w")
    assert not handler.can_handle(np.ones((3, 3), dtype=np.int32), "w")
    assert not handler.can_handle(np.ones((1, 1)), "w")
    assert not handler.can_handle(np.ma.ones((3, 3)), "w")


def test_factory_dispatches_numpy_dict() -> None:
    extractor = create_extractor({"w": np.ones((2, 2))})
    assert isinstance(extractor, NumpyDictExtractor)


def test_factory_dispatches_empty_dict() -> None:
    extractor = create_extractor({})
    assert isinstance(extractor, NumpyDictExtractor)


def test_factory_rejects_non_string_keys() -> None:
    """An array dict with non-string keys raises the same actionable
    TypeError whether or not any framework is installed — it must never
    reach the framework branches or the no-frameworks ImportError guard."""
    with pytest.raises(TypeError, match="keys must be parameter-name strings"):
        create_extractor({1: np.ones((2, 2))})
