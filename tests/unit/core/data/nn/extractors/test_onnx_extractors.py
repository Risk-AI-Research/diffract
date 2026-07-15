"""Tests for ONNX model extractor."""

from __future__ import annotations

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.core.data.nn.extractors.onnx import OnnxModelExtractor
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType

pytestmark = pytest.mark.unit


def test_onnx_model_extractor_gemm_dense_only(
    storage_cache_metadata: tuple[object, object, object],
) -> None:
    if not import_utils.is_available("onnx"):
        pytest.skip("onnx not installed")

    onnx = import_utils.require("onnx")
    helper = import_utils.require("onnx.helper")
    numpy_helper = import_utils.require("onnx.numpy_helper")

    storage, cache, metadata_index = storage_cache_metadata
    repo = ParameterRepository.initialize(storage, metadata_index, cache)

    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    w_init = numpy_helper.from_array(w, name="W")

    node = helper.make_node("Gemm", inputs=["X", "W"], outputs=["Y"], name="fc1")
    graph = helper.make_graph(
        nodes=[node],
        name="g",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
        initializer=[w_init],
    )
    model = helper.make_model(graph)

    extractor = OnnxModelExtractor(model=model)
    extractor.extract_parameters(parameter_repository=repo)
    params = list(repo.create_view())

    assert len(params) == 1
    p = params[0]
    assert p.meta.ptype == ParameterType.DENSE
    assert p.meta.name == "fc1.weight"
    got = p.get_field("weights")
    assert isinstance(got, np.ndarray)
    assert got.shape == (3, 4)
