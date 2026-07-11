"""Snapshot golden RMT values from scikit-rmt.

Regenerates tests/unit/core/compute/extensions/_golden/rmt_values.json, the
parity reference for the vendored Marchenko-Pastur CDF and Tracy-Widom ppf
in diffract.core.compute.extensions.rmt. scikit-rmt is not a dependency of the
package; run with an environment that has it installed:

    uv run --with "scikit-rmt>=1.1.0" --with "numpy<2" python scripts/gen_golden_rmt.py
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from skrmt.ensemble import MarchenkoPasturDistribution
from skrmt.ensemble.spectral_law import TracyWidomDistribution

OUTPUT = (
    Path(__file__).parent.parent
    / "tests/unit/core/compute/extensions/_golden/rmt_values.json"
)

MP_RATIOS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
MP_SIGMAS = [0.5, 1.0, 2.0]
MP_POINTS_PER_CASE = 41

TW_QUANTILES = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
]


def main() -> None:
    """Snapshot scikit-rmt reference values into the golden JSON."""
    mp_cases = []
    for ratio in MP_RATIOS:
        for sigma in MP_SIGMAS:
            dist = MarchenkoPasturDistribution(ratio, 1, sigma)
            lo = dist.lambda_minus
            hi = dist.lambda_plus
            span = hi - lo
            x = np.linspace(lo - 0.1 * span, hi + 0.1 * span, MP_POINTS_PER_CASE)
            x = x[x > 0]
            mp_cases.append(
                {
                    "ratio": ratio,
                    "sigma": sigma,
                    "lambda_minus": lo,
                    "lambda_plus": hi,
                    "x": x.tolist(),
                    "cdf": np.asarray(dist.cdf(x), dtype=float).tolist(),
                }
            )

    twd = TracyWidomDistribution()
    tw_case = {
        "beta": 1,
        "q": TW_QUANTILES,
        "ppf": [float(twd.ppf(q)) for q in TW_QUANTILES],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "generated_by": "scripts/gen_golden_rmt.py",
                "scikit_rmt_version": version("scikit-rmt"),
                "numpy_version": np.__version__,
                "marchenko_pastur_cdf": mp_cases,
                "tracy_widom_ppf": tw_case,
            }
        )
    )
    print(
        f"wrote {OUTPUT} ({len(mp_cases)} MP cases, {len(TW_QUANTILES)} TW quantiles)"
    )


if __name__ == "__main__":
    main()
