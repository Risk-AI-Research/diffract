"""Smoke-test a bare diffract-core install.

Run inside a clean environment where only the built wheel is installed
(no extras). Exercises the framework-free path end to end: NumPy dict
extraction, spectral and heavy-tailed kernels, export, and the actionable
error for the missing viz extra.
"""

from __future__ import annotations

import numpy as np

from diffract import Session


def main() -> None:
    """Run the bare-core scenario and fail loudly on any regression."""
    session = Session(profile="ram")
    with session:
        session.models.add(
            {"encoder.weight": np.random.default_rng(0).random((256, 128))},
            model_id="smoke",
        )
        session.compute.apply("frob_norm", "stable_rank", "pl_alpha", "mp_ks")
        rows = session.results.export_metrics(
            "frob_norm", "pl_alpha", "mp_ks", export_format="list"
        )
        if len(rows) != 1:
            raise AssertionError(f"expected 1 exported row, got {len(rows)}")

        try:
            session.viz.box(y="stable_rank", x="model_id")
            raise AssertionError("viz must raise without the viz extra")
        except ImportError as error:
            if "diffract-core[viz]" not in str(error):
                raise AssertionError(f"unhelpful viz hint: {error}") from error

    print("bare-core smoke OK")


if __name__ == "__main__":
    main()
