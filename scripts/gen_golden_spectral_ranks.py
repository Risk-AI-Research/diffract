"""Generate the FP64 norm and rank reference for a dense rectangular matrix."""

import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import linalg

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests/unit/session/_golden/spectral_ranks.json"
)


def main() -> None:
    """Write the input, independent references and library versions."""
    matrix = np.array([[3, -1, 4, 0], [2, 7, -3, 1], [-5, 1, 0, 2]])
    scale = int(np.abs(matrix).max())
    normalized = matrix.astype(np.float64) / scale
    _, singular, _ = linalg.svd(normalized, full_matrices=False, lapack_driver="gesvd")
    frobenius = math.hypot(*normalized.flat)
    probabilities = singular / math.fsum(singular)
    entropy = -math.fsum(float(p) * math.log(p) for p in probabilities)
    data = {
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matrix": matrix.tolist(),
        "frob_norm": scale * frobenius,
        "stable_rank": (frobenius / float(singular[0])) ** 2,
        "effective_rank": math.exp(entropy),
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
