"""Fixtures for viz tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_session():
    """Create a mock session that returns sample data."""
    session = MagicMock()

    # Sample data: 3 models, 2 parameters each
    sample_data = {
        "param1": {
            "metadata": {
                "name": "layer.0.weight",
                "model_id": "model_a",
                "layer_id": 0,
                "head_id": 0,
                "ptype": "weight",
            },
            "fields": {
                "stable_rank": 5.2,
                "frob_norm": 10.5,
                "greater_dim": 128,
                "weights_svals": np.array([1.0, 0.8, 0.5, 0.3, 0.1]),
            },
        },
        "param2": {
            "metadata": {
                "name": "layer.0.bias",
                "model_id": "model_a",
                "layer_id": 0,
                "head_id": 1,
                "ptype": "bias",
            },
            "fields": {
                "stable_rank": 3.1,
                "frob_norm": 5.2,
                "greater_dim": 64,
                "weights_svals": np.array([0.9, 0.7, 0.4, 0.2]),
            },
        },
        "param3": {
            "metadata": {
                "name": "layer.1.weight",
                "model_id": "model_b",
                "layer_id": 1,
                "head_id": 0,
                "ptype": "weight",
            },
            "fields": {
                "stable_rank": 8.0,
                "frob_norm": 15.3,
                "greater_dim": 256,
                "weights_svals": np.array([1.2, 0.9, 0.6, 0.4, 0.2, 0.1]),
            },
        },
    }

    def mock_get_results(*fields, **kwargs):
        return sample_data

    session.compute = MagicMock()
    session.get_results = mock_get_results

    return session


@pytest.fixture
def sample_results() -> dict[str, Any]:
    """Sample results dict for testing grouping/helpers."""
    return {
        "p1": {
            "metadata": {"name": "w1", "model_id": "m1", "layer_id": 0},
            "fields": {"metric": 1.0},
        },
        "p2": {
            "metadata": {"name": "w2", "model_id": "m1", "layer_id": 1},
            "fields": {"metric": 2.0},
        },
        "p3": {
            "metadata": {"name": "w3", "model_id": "m2", "layer_id": 0},
            "fields": {"metric": 3.0},
        },
    }
