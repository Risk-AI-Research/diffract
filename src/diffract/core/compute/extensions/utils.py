from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

import diffract.core.utils.imports as import_utils

if not import_utils.is_available("torch"):

    def torch_cuda_wrapper(*_args: Any, **_kwargs: Any) -> None:
        """Stub when torch is not available."""
        raise ImportError

else:
    torch = import_utils.require("torch")

    def torch_cuda_wrapper(
        array: NDArray[np.floating[Any]],
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:
        """Execute function on CUDA tensor and return to CPU."""
        torch.cuda.empty_cache()
        with torch.no_grad():
            tensor = torch.Tensor(array).to("cuda")
            output = function(tensor)
            if isinstance(output, torch.Tensor):
                output = output.to("cpu").float().numpy()
            else:
                output_container_type = type(output)

                output_ = [item.to("cpu").float().numpy() for item in output]
                output = output_container_type(output_)

        del tensor
        return output
