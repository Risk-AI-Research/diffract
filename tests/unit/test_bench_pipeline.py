"""Unit tests for the pipeline benchmark harness (scripts/bench_pipeline.py).

The harness lives outside the importable package, so it is loaded by path.
Tests target real defects: the regression gate's decision (direction,
threshold, absolute floor, device guard) and an end-to-end run that proves
every timed stage measures real, positive work rather than a silent no-op.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_pipeline.py"


def _load_harness() -> Any:
    """Load the by-path harness module, registered so dataclasses resolve."""
    spec = importlib.util.spec_from_file_location("bench_pipeline", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_harness()


def _report(device: str, stages: dict[str, float]) -> dict[str, Any]:
    """Minimal report with one ram/large_few config carrying median seconds."""
    return {
        "protocol": {"device": device},
        "results": [
            {
                "profile": "ram",
                "shape": {"name": "large_few"},
                "stages": {name: {"median_s": secs} for name, secs in stages.items()},
            }
        ],
    }


class TestShape:
    """The synthetic-workload descriptor."""

    def test_total_bytes_is_count_times_area_times_itemsize(self) -> None:
        # 4 float32 matrices of 32x32 -> 4 * 32 * 32 * 4 bytes.
        assert bench.Shape("s", n_params=4, dim=32, dtype="float32").total_bytes() == (
            4 * 32 * 32 * 4
        )

    def test_describe_exposes_rows_cols_and_bytes(self) -> None:
        described = bench.Shape("s", n_params=2, dim=16).describe()
        assert described["rows"] == 16
        assert described["cols"] == 16
        assert described["total_bytes"] == 2 * 16 * 16 * 4


class TestSummarize:
    """Per-rep reduction."""

    def test_reports_min_median_mean_and_count(self) -> None:
        stats = bench._summarize([0.1, 0.3, 0.2])
        assert stats["min_s"] == pytest.approx(0.1)
        assert stats["median_s"] == pytest.approx(0.2)
        assert stats["mean_s"] == pytest.approx(0.2)
        assert stats["n"] == 3


class TestRegressionGate:
    """compare() decides which stage timings count as regressions."""

    def test_flags_stage_slower_than_threshold(self) -> None:
        base = _report("cpu", {"apply_parameter": 0.010})
        # +20% and +2 ms, comfortably over a 5% threshold and 0.2 ms floor.
        current = _report("cpu", {"apply_parameter": 0.012})
        (row,) = bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002)
        assert row["stage"] == "apply_parameter"
        assert row["regressed"] is True
        assert row["pct_change"] == pytest.approx(20.0)

    def test_does_not_flag_within_threshold(self) -> None:
        base = _report("cpu", {"apply_parameter": 0.010})
        current = _report("cpu", {"apply_parameter": 0.0104})  # +4% < 5%
        (row,) = bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002)
        assert row["regressed"] is False

    def test_does_not_flag_faster_stage(self) -> None:
        base = _report("cpu", {"apply_parameter": 0.010})
        current = _report("cpu", {"apply_parameter": 0.005})  # 2x faster
        (row,) = bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002)
        assert row["regressed"] is False
        assert row["pct_change"] < 0

    def test_absolute_floor_suppresses_sub_millisecond_noise(self) -> None:
        # Ratio doubles (+100%) but the absolute delta is only 0.05 ms, under
        # the 0.2 ms floor: measurement noise, not a regression.
        base = _report("cpu", {"export_warm": 0.00010})
        current = _report("cpu", {"export_warm": 0.00015})
        (row,) = bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002)
        assert row["pct_change"] == pytest.approx(50.0)
        assert row["regressed"] is False

    def test_device_mismatch_disables_the_gate(self) -> None:
        base = _report("cpu", {"apply_parameter": 0.010})
        current = _report("cuda", {"apply_parameter": 0.100})  # 10x slower
        (row,) = bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002)
        assert row["device_mismatch"] is True
        assert row["regressed"] is False

    def test_unmatched_config_is_skipped_without_error(self) -> None:
        base = _report("cpu", {"apply_parameter": 0.010})
        current = _report("cpu", {"apply_parameter": 0.020})
        current["results"][0]["shape"]["name"] = "small_many"  # no baseline match
        assert bench.compare(base, current, threshold=0.05, abs_floor_s=0.0002) == []


@pytest.mark.parametrize(
    ("forward", "both", "failed"),
    [(1.06, 2.08, "l_overlap_forward"), (1.03, 2.11, "l_overlap_both")],
)
def test_pair_gate_checks_each_direction_and_the_full_pair(forward, both, failed):
    base = _report("cpu", {})
    base["results"][0]["pair_stages"] = {"single": {"median_s": 1.0}}
    current = _report("cpu", {})
    current["results"][0]["pair_stages"] = {
        "forward": {"median_s": forward},
        "reverse": {"median_s": 1.04},
        "both": {"median_s": both},
    }

    rows = bench.compare_pairs(base, current, threshold=0.05)

    assert [row["stage"] for row in rows if row["regressed"]] == [failed]


class TestPeakRSS:
    """RSS sampling backend selection."""

    def test_reports_unavailable_when_no_backend_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A None sys.modules entry makes `import` raise, simulating a
        # platform with neither psutil nor the POSIX-only resource module.
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "resource", None)
        tracker = bench._PeakRSS()
        assert tracker.method == "unavailable"
        with tracker:
            pass
        assert tracker.peak_bytes == 0


class TestP95:
    """Nearest-rank 95th percentile; the realistic defect is an off-by-one."""

    def test_hundred_distinct_samples_pick_the_95th(self) -> None:
        samples = [float(i) for i in range(1, 101)]
        assert bench._p95(samples) == 95.0

    def test_single_sample_is_its_own_p95(self) -> None:
        assert bench._p95([0.25]) == 0.25

    def test_order_of_input_does_not_matter(self) -> None:
        samples = [float(i) for i in range(1, 101)]
        assert bench._p95(list(reversed(samples))) == bench._p95(samples)


class TestStoreSizes:
    """The on-disk split must find sidecars and exclude the harness config."""

    def test_splits_index_and_value_bytes_with_sidecars(self, tmp_path: Path) -> None:
        (tmp_path / "metadata.db").write_bytes(b"a" * 10)
        (tmp_path / "metadata.db-wal").write_bytes(b"b" * 5)
        (tmp_path / "storage.db").write_bytes(b"c" * 20)
        (tmp_path / "storage.db-shm").write_bytes(b"d" * 4)
        blob_dir = tmp_path / "sqlite_blobs"
        blob_dir.mkdir()
        (blob_dir / "blob.npy").write_bytes(b"e" * 100)
        (tmp_path / "bench_local.ini").write_bytes(b"f" * 999)

        sizes = bench._store_sizes(tmp_path)

        assert sizes["metadata_index_bytes"] == 15
        assert sizes["value_store_bytes"] == 124

    def test_missing_blob_directory_is_not_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "metadata.db").write_bytes(b"a" * 3)
        (tmp_path / "storage.db").write_bytes(b"b" * 7)

        sizes = bench._store_sizes(tmp_path)

        assert sizes == {"metadata_index_bytes": 3, "value_store_bytes": 7}


class TestEndToEnd:
    """A tiny real run must time every stage on actual pipeline work."""

    def test_ram_run_times_every_stage_with_positive_work(self) -> None:
        shape = bench.Shape(name="large_few", n_params=2, dim=8)
        report = bench.run_benchmark(
            profiles=("ram",), shapes=(shape,), device="cpu", reps=1
        )

        assert report["schema"] == bench.SCHEMA
        assert report["protocol"]["device"] == "cpu"
        assert report["protocol"]["warm"] is True
        assert report["environment"]["python"]
        assert report["environment"]["numpy"]

        resolver = report["resolver"]
        assert resolver["n"] == bench.RESOLVE_ITERS
        assert 0 < resolver["median_s"] <= resolver["p95_s"]

        (result,) = report["results"]
        assert result["profile"] == "ram"
        assert result["shape"]["n_params"] == 2
        assert result["store"] is None

        for stage in bench.STAGES:
            assert result["stages"][stage]["median_s"] >= 0
        # The parameter stage always runs a real SVD; it cannot be a zero no-op.
        assert result["stages"]["apply_parameter"]["median_s"] > 0
        # The no-op stages re-apply computed fields; a skip that recomputed the
        # SVD would take the same order as the apply stage itself.
        assert (
            result["stages"]["apply_parameter_noop"]["median_s"]
            < result["stages"]["apply_parameter"]["median_s"]
        )
        assert result["memory"]["peak_rss_mb"] > 0
