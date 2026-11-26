"""Run the same public-session workload with either installation of Diffract."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

from diffract import Session
from diffract.viz.data.extraction import get_field_values

METRICS = ("weights", "weights_svals", "frob_norm")
AGGREGATES = ("param_norm", "l_overlap")


def array_record(value):
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def add_model(session, model, seed):
    rng = np.random.default_rng(seed)
    session.models.add(
        {
            f"layer.{i}.weight": rng.normal(size=(9, 6)).astype("float32")
            for i in range(2)
        },
        model_id=model,
    )


def snapshot(session):
    metrics = session.results.export_metrics(*METRICS, export_format="dict")
    rows = []
    for entry in metrics.values():
        meta = entry["metadata"]
        for field, value in entry["fields"].items():
            if field in METRICS:
                rows.append(
                    (
                        [meta["model_id"], meta.get("step", 0), meta["name"], field],
                        array_record(value),
                    )
                )
    agg = [
        {
            "key": [
                entry["field"],
                sorted(entry["context_models"]),
                sorted(entry["context_params"]),
            ],
            "roles": [entry.get("model_source"), entry.get("model_target")],
            "members": [
                [member["model_id"], member["step"], member["name"]]
                for member in entry.get("members", ())
            ],
            "steps": [entry.get("step_source"), entry.get("step_target")],
            "unknown": entry.get("direction_unknown", False),
            "value": array_record(entry["value"]),
        }
        for entry in session.results.export_aggregates(
            *AGGREGATES, export_format="list"
        )
    ]
    # The v0.3 visualization route accepts an explicit contextual address.
    entries = session.results.export("frob_norm", "param_norm", export_format="dict")
    viz = []
    for entry in entries.values():
        meta = entry["metadata"]
        model = meta["model_id"]
        if model in {"alpha", "beta"}:
            address = (
                f"param_norm@models[{model}]@params[layer.0.weight,layer.1.weight]"
            )
            (value,) = get_field_values({"entry": entry}, address)
            viz.append(([model, meta["name"]], array_record(value)))
    return {"metrics": rows, "aggregates": agg, "viz": viz}


def seed_edge(case):
    with Session(profile="local") as session:
        if case == "empty":
            return
        model = "legacy@,[]" if case == "reserved" else "net"
        rng = np.random.default_rng(87)
        names = [f"block.{i}.weight" for i in rng.permutation(8)]
        weights = {
            name: np.diag(np.arange(1, 6, dtype=np.float32) + i)
            for i, name in enumerate(names)
        }
        session.models.add(weights, model_id=model)
        for i, row in enumerate(session.models.parameters.list()):
            session.results.ingest_metrics({row["uid"]: {"observed": i + 0.125}})
        if case == "partial":
            session.compute.apply("weights_svals")
            session.results.erase("weights_svals")


def run(action, root):
    with Session(profile="local") as session:
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        if action == "seed":
            add_model(session, "alpha", 3)
            session.compute.apply("frob_norm")
            add_model(session, "beta", 19)
            for field in ("frob_norm", "param_norm", "l_overlap"):
                session.compute.apply(field)
        if action == "exercise":
            session.results.erase("weights_svals", "weights_lsvs", "weights_rsvs")
            erased = session.results.export_metrics(
                "weights_svals", export_format="dict"
            )
            assert all("weights_svals" not in row["fields"] for row in erased.values())
            session.compute.apply("weights_svals")
            with Session(profile="ram") as donor:
                add_model(donor, "donor", 31)
                donor.compute.apply("frob_norm")
                session.utils.merge_other_session(donor)
    with Session(profile="local") as reopened:
        if action == "exercise":
            reopened.filter(model_ids=["alpha", "donor"]).compute.apply("l_overlap")
        result = snapshot(reopened)
    (root / f"{action}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    import diffract

    if sys.argv[2] == "legacy" and diffract.__version__ != "0.3.0":
        raise RuntimeError(
            f"The reference workload needs Diffract 0.3.0, got {diffract.__version__}"
        )
    if sys.argv[1].startswith("edge-"):
        seed_edge(sys.argv[1].removeprefix("edge-"))
    else:
        run(sys.argv[1], Path.cwd())
