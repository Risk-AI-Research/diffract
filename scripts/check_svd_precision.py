"""Rehearse weight SVD precision on an explicitly selected CPU or NVIDIA GPU."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg

from diffract import Session

if TYPE_CHECKING:
    from collections.abc import Iterator


def matrices() -> dict[str, np.ndarray]:
    """Construct the fixed orthogonal-factor and Gaussian precision panel."""
    rng = np.random.default_rng(148)
    left, right = (linalg.qr(rng.standard_normal((128, 128)))[0] for _ in range(2))
    inputs = {
        f"cond{e}.weight": (left * np.geomspace(1, 10.0**-e, 128)) @ right.T
        for e in (2, 4, 6, 8, 10)
    }
    for i in range(8):
        design = np.random.default_rng(2001 + i).standard_normal((128, 384))
        inputs[f"draw{i}.weight"] = design[:, :128].copy()
    return inputs


def inverse_mass(weights_svals: np.ndarray) -> float:
    """Compute the inverse spectral moment in double precision."""
    return float(np.sum(np.asarray(weights_svals, dtype=np.float64) ** -2))


def execution(session: Session, run: str, uid: str, device: str) -> dict[str, Any]:
    """Check the solver and arithmetic used by one matrix decomposition."""
    (facts,) = session.results.describe_run(run, parameter_uid=uid)["arithmetic"]
    backend, solver, driver = (
        ("torch", "cuSOLVER", "gesvd")
        if device == "cuda"
        else ("scipy", "LAPACK", "gesdd")
    )
    expected = dict(
        origin="observed",
        backend=backend,
        solver=solver,
        driver=driver,
        input_dtype="float64",
        working_dtype="float64",
        output_dtype="float64",
    )
    if facts["device"].split(":")[0] != device or any(
        facts[key] != value for key, value in expected.items()
    ):
        raise AssertionError(f"Unexpected execution for {uid}: {facts}")
    return facts


def panel(device: str) -> Iterator[dict]:
    """Yield measured errors and tolerances for original and permuted matrices."""
    if device == "cuda":
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("The CUDA rehearsal requires an available NVIDIA GPU.")
    inputs = matrices()
    fields = (
        "weights_svals",
        "inverse_mass",
        "frob_norm",
        "weights_rand",
        "weights_rand_svals",
        "weights_rand_lsvs",
        "weights_rand_rsvs",
    )
    with Session(profile="ram") as session:
        session.compute.kernel()(inverse_mass)
        for kernel in ("weights_svd", "weights_rand_svd", "weights_rand_vectors"):
            session.compute.configure_kernel(
                kernel, allow_cuda=device == "cuda", precision="preserve"
            )
        session.models.add(inputs, model_id="design")
        session.compute.apply(*fields)
        rows = session.results.export_metrics(*fields, export_format="list")
        if {r["parameter_name"] for r in rows} != set(inputs) or len(rows) != len(
            inputs
        ):
            raise AssertionError("The precision panel requires one result per input.")
        identities = {
            (r["param_name"], r["field"]): (r["run_id"], r["parameter_uid"])
            for r in session.results.export(
                "weights_svals",
                "weights_rand_svals",
                "weights_rand_lsvs",
                shape="long",
                sources="metrics",
                export_format="list",
            )
        }
        for row in rows:
            name = row["parameter_name"]
            for randomized in (False, True):
                field = "weights_rand_svals" if randomized else "weights_svals"
                matrix = row["weights_rand"] if randomized else inputs[name]
                s = row[field]
                ref = linalg.svdvals(matrix.astype(np.float64, copy=False))
                # An independent QR driver checks the reference on the actual input.
                _, qr, _ = linalg.svd(
                    matrix, full_matrices=False, lapack_driver="gesvd"
                )
                eps = np.finfo(np.float64).eps
                kappa = ref[0] / ref[-1]
                tail_tol, inverse_tol, norm_tol = (
                    16 * eps * kappa,
                    64 * eps * kappa,
                    64 * eps,
                )
                np.testing.assert_allclose(qr, ref, rtol=tail_tol, atol=0)
                run, uid = identities[name, field]
                facts = execution(session, run, uid, device)
                if s.dtype != np.float64:
                    raise AssertionError(f"{name} {field}: unexpected dtype {s.dtype}")
                if randomized:
                    u, v = row["weights_rand_lsvs"], row["weights_rand_rsvs"]
                    np.testing.assert_allclose(
                        (u * s) @ v.T, matrix, atol=256 * eps * ref[0], rtol=256 * eps
                    )
                    vec_run, _ = identities[name, "weights_rand_lsvs"]
                    execution(session, vec_run, uid, device)
                    if u.dtype != np.float64 or v.dtype != np.float64:
                        raise AssertionError(f"{name}: unexpected vector dtypes")
                moment = inverse_mass(s) if randomized else row["inverse_mass"]
                norm = linalg.norm(s) if randomized else row["frob_norm"]
                errors = (
                    abs(s[0] / ref[-1] - 1),
                    abs(
                        moment / math.fsum(1.0 / (float(v) * float(v)) for v in ref) - 1
                    ),
                    abs(norm / linalg.norm(matrix, "fro") - 1),
                )
                yield {
                    "input": name,
                    "field": field,
                    "device": facts["device"],
                    "solver": facts["solver"],
                    "driver": facts["driver"],
                    "input_dtype": facts["input_dtype"],
                    "working_dtype": facts["working_dtype"],
                    "output_dtype": str(s.dtype),
                    "ref_min": ref[-1],
                    "out_min": s[0],
                    "rel_tail_error": errors[0],
                    "rel_inverse_error": errors[1],
                    "rel_frob_error": errors[2],
                    "tail_tol": tail_tol,
                    "inverse_tol": inverse_tol,
                    "frob_tol": norm_tol,
                    "pass": all(
                        err <= tol
                        for err, tol in zip(
                            errors, (tail_tol, inverse_tol, norm_tol), strict=True
                        )
                    ),
                }


def main() -> None:
    """Write the precision report and enforce its per-input tolerances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    rows = list(panel(args.device))
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [(r["input"], r["field"]) for r in rows if not r["pass"]]
    if failed:
        sys.exit(f"Outside precision tolerances: {failed}")


if __name__ == "__main__":
    main()
