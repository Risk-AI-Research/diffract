"""Qualify an installed wheel on the seven OLMo 3 7B Base block-zero matrices."""

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy import linalg
from scipy.special import entr

from diffract import Session

REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
SHARD = "model-00001-of-00003.safetensors"
SHA256 = "0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2"
RTOL = 5e-5
SHAPES = {
    "mlp.down_proj": (4096, 11008),
    "mlp.gate_proj": (11008, 4096),
    "mlp.up_proj": (11008, 4096),
    **{f"self_attn.{p}_proj": (4096, 4096) for p in ("q", "k", "v", "o")},
}
FIELDS = ("frob_norm", "l2_norm", "stable_rank", "effective_rank")


def measure(name: str, tensor: torch.Tensor) -> Iterator[dict]:
    """Compare the CUDA calculation with an independent CPU spectrum."""
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with Session(profile="ram") as session:
        report = session.models.add({name: tensor}, model_id="olmo3-7b-base")
        (item,) = report.items
        session.compute.configure_kernel(
            "weights_svd", allow_cuda=True, precision="preserve"
        )
        session.compute.apply(*FIELDS)
        (row,) = session.results.export_metrics(*FIELDS, export_format="list")
        (trace,) = session.results.export(
            "weights_svals", sources="metrics", shape="long", export_format="list"
        )
        (facts,) = session.results.describe_run(
            trace["run_id"], parameter_uid=trace["parameter_uid"]
        )["arithmetic"]
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        admitted = (
            report.counts["candidates"] == report.counts["extracted"] == 1
            and item.name == item.stored_name == row["parameter_name"] == name
            and tensor.dtype == torch.bfloat16
            and item.source_dtype == facts["checkpoint_dtype"] == "bfloat16"
            and item.storage_dtype
            == facts["input_dtype"]
            == facts["working_dtype"]
            == facts["output_dtype"]
            == "float32"
            and facts["device"].startswith("cuda:")
            and facts["backend"] == "torch"
            and facts["origin"] == "observed"
            and facts["solver"] == "cuSOLVER"
            and facts["driver"] == "gesvd"
            and peak > 0
        )
    start = time.perf_counter()
    array = tensor.double().numpy()
    sv = linalg.svdvals(array)
    ref = {
        "frob_norm": linalg.norm(array, "fro"),
        "l2_norm": sv[0],
        "stable_rank": np.sum((sv / sv[0]) ** 2),
        "effective_rank": np.exp(entr(sv / sv.sum()).sum()),
    }
    cpu_wall = time.perf_counter() - start
    for field in FIELDS:
        error = abs(float(row[field]) / ref[field] - 1.0)
        yield {
            "name": name,
            "rows": tensor.shape[0],
            "columns": tensor.shape[1],
            "metric": field,
            "observed": row[field],
            "cpu_fp64": ref[field],
            "relative_error": error,
            "rtol": RTOL,
            "source_dtype": item.source_dtype,
            "stored_dtype": item.storage_dtype,
            "working_dtype": facts["working_dtype"],
            "output_dtype": facts["output_dtype"],
            "device": facts["device"],
            "driver": facts["driver"],
            "svd_run_id": trace["run_id"],
            "cuda_seconds": wall,
            "reference_seconds": cpu_wall,
            "peak_allocated_bytes": peak,
            "pass": bool(admitted and np.isfinite(error) and error <= RTOL),
        }


def main() -> None:
    """Check the pinned checkpoint and write one result per weight and metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory with the pinned shard and index.json",
    )
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("An NVIDIA CUDA device is required.")
    index = json.loads((args.source / "index.json").read_text())["weight_map"]
    names = {f"model.layers.0.{part}.weight": shape for part, shape in SHAPES.items()}
    if any(index.get(n) != SHARD for n in names):
        raise ValueError("Unexpected block-zero shard mapping.")
    shard = args.source / SHARD
    with shard.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
            raise ValueError(
                f"Shard SHA-256 differs from the LFS object at {REVISION}."
            )
    failed = []
    with (
        safe_open(shard, framework="pt", device="cpu") as handle,
        args.csv.open("w", newline="") as out,
    ):
        writer = None
        for name, shape in names.items():
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            for row in measure(name, tensor):
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if not row["pass"]:
                    failed.append((name, row["metric"]))
            out.flush()
            print(name, flush=True)
    if failed:
        raise SystemExit(f"Outside qualification requirements: {failed}")


if __name__ == "__main__":
    main()
