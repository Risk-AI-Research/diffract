"""Tests for TensorFlow model extractor."""

from __future__ import annotations

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.core.data.nn.extractors.tensorflow import TensorFlowModelExtractor
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = pytest.mark.unit


def test_tensorflow_model_extractor_dense_only(
    storage_cache_metadata: tuple[object, object, object],
) -> None:
    if not import_utils.is_available("tensorflow"):
        pytest.skip("tensorflow not installed")

    tf = import_utils.require("tensorflow")
    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(3, use_bias=False, name="dense0"),
            tf.keras.layers.ReLU(name="relu0"),
            tf.keras.layers.Dense(2, use_bias=False, name="dense1"),
        ]
    )

    extractor = TensorFlowModelExtractor(model=model)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert len(params) == 2
    names = sorted([p.meta.name for p in params])
    assert names == ["dense0.kernel", "dense1.kernel"]

    for p in params:
        assert p.meta.ptype == ParameterType.DENSE
        w = p.get_field("weights")
        assert isinstance(w, np.ndarray)
        assert w.ndim == 2

