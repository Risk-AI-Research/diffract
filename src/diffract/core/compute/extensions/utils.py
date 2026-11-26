"""Transfer NumPy inputs to CUDA and return solver outputs at their native dtype."""

from collections.abc import Callable
from typing import Any

import numpy as np

from diffract.core.utils.imports import require


def torch_cuda_wrapper(
    array: np.ndarray, function: Callable[..., Any], *, device: str = "cuda"
) -> Any:
    """Run a torch solver on the selected CUDA device."""
    torch = require("torch")
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        output = function(tensor)
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        return tuple(item.cpu().numpy() for item in output)
