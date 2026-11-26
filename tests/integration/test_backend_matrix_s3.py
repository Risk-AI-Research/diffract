"""Integration: the S3-backed Zarr cell of the backend x arrangement matrix.

Requires an S3-compatible endpoint:
    DIFFRACT_TEST_S3_URL - fsspec-style store URL (e.g. s3://bucket/prefix)
    plus whatever credentials the endpoint expects (AWS_* variables).

Skip behavior: skips without DIFFRACT_TEST_S3_URL or without the zarr
extra; fails on zarr-python 2.x (the matrix supports zarr>=3 only).

The unit-tree matrix module names this cell so the grid stays whole:
tests/unit/core/storage/test_backend_profile_matrix.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.containers import create_main_container
from diffract.core.storage.maintenance import check_store, collect_store
from diffract.session import Session

pytestmark = [pytest.mark.integration, pytest.mark.network]

S3_SKIP_REASON = (
    "S3-backed Zarr cell: DIFFRACT_TEST_S3_URL is not set; the cell needs "
    "reachable S3-compatible credentials and is marker-gated (integration, "
    "network) by design"
)


def _seed_model() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "enc.weight": rng.standard_normal((8, 8)).astype("float32"),
        "head.weight": rng.standard_normal((64, 64)).astype("float32"),
    }


def test_s3_zarr_identity_roundtrip(tmp_path: Path) -> None:
    """Exercise identity, run traces and maintenance in an isolated S3 prefix."""
    url = os.environ.get("DIFFRACT_TEST_S3_URL")
    if not url:
        pytest.skip(S3_SKIP_REASON)
    if not import_utils.is_available("zarr"):
        pytest.skip('zarr not installed (pip install "diffract-core[zarr]")')
    import zarr

    if int(zarr.__version__.split(".")[0]) < 3:
        pytest.fail(
            f"zarr-python {zarr.__version__} is the 2.x line — "
            "the matrix supports zarr>=3 only"
        )

    url = f"{url.rstrip('/')}/phase2-{uuid4().hex}"
    index = tmp_path / "metadata.db"
    lines = [
        "[logging]",
        "version = 1",
        "disable_existing_loggers = false",
        "[logging.handlers.console]",
        "class = logging.StreamHandler",
        "level = WARNING",
        "stream = ext://sys.stdout",
        "[logging.root]",
        "level = WARNING",
        'handlers = ["console"]',
        "[storage]",
        'backend = "zarr"',
        "[storage.zarr]",
        f"store_url = {json.dumps(url)}",
        "[metadata]",
        'backend = "sqlite"',
        "[metadata.sqlite]",
        f"path = {json.dumps(str(index))}",
        "[cache]",
        'backend = "none"',
        "[nn.extractor]",
        "skip_not_implemented_types = true",
        "[export]",
        'default_export_format = "dict"',
        "",
    ]
    config = tmp_path / "s3_cell.ini"
    config.write_text("\n".join(lines))
    model = _seed_model()

    with Session(config_path=str(config)) as session:
        session.models.add(model, model_id="m1")

    container = create_main_container(str(config))
    with Session(container=container) as session:
        exported = session.results.export("weights", export_format="dict")
        stored = {
            entry["metadata"]["name"]: entry["fields"]["weights"]
            for entry in exported.values()
        }
        for name, weights in model.items():
            np.testing.assert_array_equal(stored[name], weights)
        store = container.storage.storage_manager()
        for uid in exported:
            generation = store.catalog.get("parameter", uid, "weights").generation
            assert generation is not None
            assert generation.output_name == "weights"
            assert store.catalog.index.find_row(
                "runs", columns=("kind", "status"), where={"run_id": generation.run_id}
            ) == {"kind": "ingest", "status": "COMPLETE"}
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        session.compute.apply("weights_svals")
        assert {row.status for row in session.results.status("weights_svals")} == {
            "fresh"
        }
        removed = session.results.erase("weights_svals")
        assert removed.fields == ("weights_lsvs", "weights_rsvs", "weights_svals")
        assert removed.affected_uids == len(model)

    assert not any(check_store(index, config=config).values())
    preview = collect_store(index, config=config, dry_run=True, grace_days=0)
    assert len(preview["tombstones"]) == len(preview["blobs"]) == 3 * len(model)
    assert collect_store(index, config=config, grace_days=0) == preview
    assert not any(check_store(index, config=config).values())
