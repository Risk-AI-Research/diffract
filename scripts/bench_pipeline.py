"""Timing and memory benchmark for the core add -> apply -> export pipeline.

Times the three pipeline entry points -- ``models.add``, ``compute.apply``
(once per apply level: parameter, in-model, cross-model), and
``results.export`` -- across two storage profiles (``ram``, ``local``) and two
workload shapes (a few large matrices vs. many small matrices), on a pinned
device with warm timings, and records peak resident set size. It emits a small
JSON report; passed a prior report via ``--baseline`` it doubles as a
regression gate that fails when any stage gets slower than a threshold at an
equal profile/shape/device.

Beyond the pipeline stages the report records the substrate metrics that a
storage or identity change moves first: the no-op re-apply of an already
computed field at the parameter and cross-model levels (the skip-predicate
lookup path, whole-store wall time), the field-address resolution round-trip
(parse plus label render, microseconds, reported with p95), and -- for the
disk profile -- the on-disk split between the metadata index and the value
store (WAL sidecars included) measured after the session closes.

The many-small shape is not optional: pipeline overhead scales with the NUMBER
of parameters (a per-parameter Python loop), not their byte size, so a run on
large matrices alone would miss substrate regressions.

To keep runs comparable the harness pins:

* the storage profile (``ram`` vs. an isolated ``local`` SQLite store),
* the workload shape and parameter count,
* the SVD device (CUDA is disabled by default via the ``weights_svd`` kernel's
  ``allow_cuda`` config so the numbers are reproducible off-GPU),
* warm timing (process imports and a discarded warm-up round are excluded),
* progress bars and info logging (silenced inside every timed region).

Usage::

    python scripts/bench_pipeline.py --out results.json
    python scripts/bench_pipeline.py --baseline results.json --out new.json

Use `--pairs single` with a single-direction implementation and
`--pairs directed` with a two-direction implementation to measure `l_overlap`
with its inputs already stored. Run this same script with each checkout's
`src` on `PYTHONPATH` to record both reports. The pair comparison checks each
direction against the single-direction baseline and both together against
twice that baseline.

Peak RSS uses ``psutil`` (the ``bench`` extra); it falls back to
``resource.getrusage`` when ``psutil`` is not installed. Where neither is
available (Windows without the extra: the stdlib ``resource`` module is
POSIX-only) RSS is reported as zero with ``rss_method`` set to
``"unavailable"``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any, Self

import numpy as np

# --- Report identity ---------------------------------------------------------

SCHEMA = "diffract-bench-pipeline/2"

DEFAULT_REPS = 5
QUICK_REPS = 2

# --- Fields exercised at each apply level ------------------------------------
#
# frob_norm requires weights_svals, whose sole producer (weights_svd) emits the
# left/right singular vectors in the same call. Applying it therefore pays the
# full economy SVD once; the in-model reduce (param_norm) then reuses frob_norm,
# and the cross-model overlap (avg_l_agreement) reuses the stored left singular
# vectors -- so each stage times its own runner rather than recomputing the SVD.
PARAMETER_FIELD = "frob_norm"
IN_MODEL_FIELD = "param_norm"
CROSS_MODEL_FIELD = "avg_l_agreement"
EXPORT_FIELD = "frob_norm"
PAIR_FIELD = "l_overlap"

STAGES = (
    "add",
    "apply_parameter",
    "apply_in_model",
    "apply_cross_model",
    "export_cold",
    "export_warm",
    "apply_parameter_noop",
    "apply_cross_model_noop",
)

# The no-op stages run after the exports so export_cold stays a genuinely cold
# read; they re-apply already-computed fields, so they time the skip predicate
# (existing-result detection) over the whole store rather than kernel compute.

# One address-resolution round-trip = parse a contextual field address into its
# structured selector and render the canonical label back. The loop pools its
# samples across iterations for a stable p95.
RESOLVE_ITERS = 1000

# The SVD kernel routes to a CUDA path when torch reports a device; pinning its
# config keyword makes the timed path deterministic across machines.
_SVD_KERNEL = "weights_svd"

_MODEL_A = "bench_model_a"
_MODEL_B = "bench_model_b"


@dataclass(frozen=True)
class Shape:
    """A synthetic workload: ``n_params`` square ``dim`` x ``dim`` matrices."""

    name: str
    n_params: int
    dim: int
    dtype: str = "float32"

    def total_bytes(self) -> int:
        """Resident size of one model's weights in bytes."""
        return self.n_params * self.dim * self.dim * np.dtype(self.dtype).itemsize

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description of the shape."""
        d = asdict(self)
        d["rows"] = self.dim
        d["cols"] = self.dim
        d["total_bytes"] = self.total_bytes()
        return d


DEFAULT_SHAPES = (
    Shape(name="large_few", n_params=8, dim=512),
    Shape(name="small_many", n_params=256, dim=32),
)

QUICK_SHAPES = (
    Shape(name="large_few", n_params=3, dim=64),
    Shape(name="small_many", n_params=24, dim=16),
)


# --- Timing primitives -------------------------------------------------------


@contextlib.contextmanager
def _silenced() -> Any:
    """Redirect stdout and stderr to os.devnull for the timed region.

    Progress bars and info logging would otherwise write inside the measured
    window; an exception raised in the region still propagates and prints its
    traceback once the streams are restored.
    """
    with (
        Path(os.devnull).open("w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        yield


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Return the wall-clock seconds fn() took and its result."""
    with _silenced():
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
    return elapsed, result


def _summarize(samples: list[float]) -> dict[str, float | int]:
    """Reduce per-rep seconds to min/median/mean plus the sample count."""
    return {
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "n": len(samples),
    }


def _p95(samples: list[float]) -> float:
    """The 95th-percentile sample (nearest-rank, conservative for latency)."""
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


# --- Peak RSS ----------------------------------------------------------------


class _PeakRSS:
    """Sample resident set size and keep the running maximum.

    With psutil a daemon thread polls the process RSS on a short interval. When
    psutil is absent it falls back to ``resource.getrusage``, whose peak is
    process-wide and monotonic, so the per-config figure is the high-water mark
    up to that point rather than an isolated window. When the ``resource``
    module is also missing (Windows), sampling is disabled and the peak
    reads zero with ``method`` set to ``"unavailable"``.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self._interval = interval_s
        self._proc = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._peak = 0
        try:
            import psutil

            self._proc = psutil.Process()
            self.method = "psutil"
        except Exception:  # noqa: BLE001 - optional dependency, degrade to stdlib
            try:
                import resource  # noqa: F401 - availability probe (POSIX-only)

                self.method = "getrusage"
            except ImportError:
                self.method = "unavailable"

    @staticmethod
    def _getrusage_bytes() -> int:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        return maxrss * 1024 if sys.platform != "darwin" else maxrss

    def sample(self) -> int:
        """Current RSS in bytes; zero when no sampling backend is available."""
        if self._proc is not None:
            return int(self._proc.memory_info().rss)
        if self.method == "getrusage":
            return self._getrusage_bytes()
        return 0

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak = max(self._peak, self.sample())

    def __enter__(self) -> Self:
        self._peak = self.sample()
        if self._proc is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        self._peak = max(self._peak, self.sample())

    @property
    def peak_bytes(self) -> int:
        """Maximum RSS observed over the sampled window."""
        return self._peak


# --- Sessions ----------------------------------------------------------------


def _build_model(shape: Shape, seed: int) -> dict[str, np.ndarray]:
    """A dict-of-arrays model matching the shape, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(shape.dtype)
    return {
        f"layer.{i}.weight": rng.standard_normal((shape.dim, shape.dim)).astype(dtype)
        for i in range(shape.n_params)
    }


def _write_local_config(directory: Path) -> Path:
    """Write an isolated SQLite profile INI with absolute paths and quiet logs.

    Mirrors the packaged ``local`` profile (SQLite metadata + storage, bounded
    cache) but points every store at ``directory`` and drops the file log
    handler, so each rep starts from an empty store and logging never writes
    inside a timed region.
    """
    config = directory / "bench_local.ini"
    config.write_text(
        "\n".join(
            (
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
                'backend = "sqlite"',
                "[storage.sqlite]",
                f'path = "{directory / "storage.db"}"',
                "[metadata]",
                'backend = "sqlite"',
                "[metadata.sqlite]",
                f'path = "{directory / "metadata.db"}"',
                "[cache]",
                'backend = "simple"',
                "[cache.simple]",
                "max_memory_mb = 4096",
                "[parallel.thread_pool]",
                "max_workers = 8",
                "[nn.extractor]",
                "skip_not_implemented_types = true",
                "[export]",
                'default_export_format = "dict"',
                "",
            )
        )
    )
    return config


@contextlib.contextmanager
def _make_session(profile: str) -> Any:
    """Yield ``(session, store_dir)``, cleaning up any disk store after.

    ``ram`` uses the packaged in-memory profile and yields ``store_dir=None``;
    ``local`` gets a throwaway temp directory so every rep sees an empty
    persistent store, and yields that directory so the caller can measure the
    on-disk footprint before cleanup.
    """
    from diffract import Session

    if profile == "ram":
        yield Session(profile="ram"), None
        return

    if profile == "local":
        directory = Path(tempfile.mkdtemp(prefix="diffract-bench-"))
        try:
            yield Session(config_path=str(_write_local_config(directory))), directory
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
        return

    raise ValueError(f"Unknown profile: {profile!r}")


def _store_sizes(directory: Path) -> dict[str, int]:
    """On-disk bytes of the metadata index and the value store, WAL included.

    The value store is the storage database plus its external blob directory;
    the config INI written by the harness is excluded from both figures.
    """
    metadata_bytes = sum(f.stat().st_size for f in directory.glob("metadata.db*"))
    value_bytes = sum(f.stat().st_size for f in directory.glob("storage.db*"))
    blob_dir = directory / "sqlite_blobs"
    if blob_dir.exists():
        value_bytes += sum(f.stat().st_size for f in blob_dir.rglob("*") if f.is_file())
    return {
        "metadata_index_bytes": metadata_bytes,
        "value_store_bytes": value_bytes,
    }


def _run_rep(
    profile: str, shape: Shape, allow_cuda: bool
) -> tuple[dict[str, float], dict[str, int] | None]:
    """Run one full pipeline in a fresh session.

    Two same-shape models are added so the cross-model overlap has a pair to
    align. Only the first ``add`` is timed; the SVD prerequisite computed at the
    parameter stage is reused by the later stages. The no-op re-applies come
    after the exports so ``export_cold`` stays cold.

    Returns per-stage seconds plus the on-disk store split (None for ``ram``),
    measured after the session context closes so buffered writes are flushed.
    """
    model_a = _build_model(shape, seed=0)
    model_b = _build_model(shape, seed=1)
    timings: dict[str, float] = {}

    with _make_session(profile) as (session, store_dir):
        with session:
            session.compute.configure_kernel(_SVD_KERNEL, allow_cuda=allow_cuda)

            timings["add"], _ = _timed(
                lambda: session.models.add(model_a, model_id=_MODEL_A)
            )
            session.models.add(model_b, model_id=_MODEL_B)

            timings["apply_parameter"], _ = _timed(
                lambda: session.compute.apply(PARAMETER_FIELD)
            )
            timings["apply_in_model"], _ = _timed(
                lambda: session.compute.apply(IN_MODEL_FIELD)
            )
            timings["apply_cross_model"], _ = _timed(
                lambda: session.compute.apply(CROSS_MODEL_FIELD)
            )
            timings["export_cold"], _ = _timed(
                lambda: session.results.export(EXPORT_FIELD, export_format="dict")
            )
            timings["export_warm"], _ = _timed(
                lambda: session.results.export(EXPORT_FIELD, export_format="dict")
            )
            timings["apply_parameter_noop"], _ = _timed(
                lambda: session.compute.apply(PARAMETER_FIELD)
            )
            timings["apply_cross_model_noop"], _ = _timed(
                lambda: session.compute.apply(CROSS_MODEL_FIELD)
            )

        sizes = _store_sizes(store_dir) if store_dir is not None else None

    return timings, sizes


def _run_pair_rep(
    profile: str, shape: Shape, allow_cuda: bool, direction: str
) -> float:
    """Time the overlap stage and validate the number and orientation of its outputs."""
    model_a = _build_model(shape, seed=0)
    model_b = _build_model(shape, seed=1)
    with _silenced(), _make_session(profile) as (session, _), session:
        session.compute.configure_kernel(_SVD_KERNEL, allow_cuda=allow_cuda)
        session.models.add(model_a, model_id=_MODEL_A)
        session.models.add(model_b, model_id=_MODEL_B)
        session.compute.apply("weights_svals", "lower_dim")

        # The public apply expands both orientations. Seed the other result
        # through the same repository to measure one pending orientation.
        aggregate_repo = session._Session__agg_repo  # noqa: SLF001
        seeded_revisions = {}
        if direction in {"forward", "reverse"}:
            param_repo = session._Session__param_repo  # noqa: SLF001
            params = {
                (param.meta.model_id, param.meta.name): param
                for param in param_repo.create_view()
            }
            source, target = (
                (_MODEL_B, _MODEL_A) if direction == "forward" else (_MODEL_A, _MODEL_B)
            )
            with aggregate_repo:
                for name in model_a:
                    first = params[source, name]
                    second = params[target, name]
                    left = first.get_field("weights_lsvs")
                    right = second.get_field("weights_lsvs")
                    proxy = aggregate_repo.get_or_create(
                        PAIR_FIELD,
                        member_uids=(first.meta.uid, second.meta.uid),
                        directed=True,
                    )
                    proxy.set_field("value", np.abs(left.T @ right))
                    seeded_revisions[proxy.meta.uid] = (
                        aggregate_repo.storage_manager.live_pointer(
                            proxy.meta.uid, "value", aggregate_repo.TABLE
                        ).revision
                    )

        elapsed, summary = _timed(lambda: session.compute.apply(PAIR_FIELD))
        expected_count = shape.n_params * (1 if direction == "single" else 2)
        if (
            summary.parameter_fields
            or summary.cross_model_fields != (PAIR_FIELD,)
            or len(aggregate_repo) != expected_count
        ):
            raise RuntimeError("The pair probe did not execute the expected workload")
        for uid, revision in seeded_revisions.items():
            stored = aggregate_repo.storage_manager.live_pointer(
                uid, "value", aggregate_repo.TABLE
            )
            if stored.revision != revision:
                raise RuntimeError(
                    "The pair probe rewrote an already-computed direction"
                )

        if direction != "single":
            rows = session.results.export_aggregates(PAIR_FIELD, export_format="list")
            values = {
                (
                    row["model_source"],
                    row["model_target"],
                    row["members"][0]["name"],
                ): row["value"]
                for row in rows
            }
            if len(values) != expected_count:
                raise RuntimeError("The pair probe produced duplicate or missing roles")
            param_repo = session._Session__param_repo  # noqa: SLF001
            params = {
                (param.meta.model_id, param.meta.name): param
                for param in param_repo.create_view()
            }
            for name in model_a:
                left = params[_MODEL_A, name].get_field("weights_lsvs")
                right = params[_MODEL_B, name].get_field("weights_lsvs")
                expected = np.abs(left.T @ right)
                for pair, value in (
                    ((_MODEL_A, _MODEL_B, name), expected),
                    ((_MODEL_B, _MODEL_A, name), expected.T),
                ):
                    np.testing.assert_allclose(
                        values[pair], value, rtol=2e-6, atol=2e-7
                    )
    return elapsed


def _run_pair_config(
    profile: str, shape: Shape, allow_cuda: bool, reps: int, mode: str
) -> dict[str, dict[str, float | int]]:
    directions = ("single",) if mode == "single" else ("forward", "reverse", "both")
    results = {}
    for direction in directions:
        _run_pair_rep(profile, shape, allow_cuda, direction)
        samples = [
            _run_pair_rep(profile, shape, allow_cuda, direction) for _ in range(reps)
        ]
        results[direction] = _summarize(samples)
    return results


def _time_resolver(iters: int) -> list[float]:
    """Per-iteration seconds of the address-resolution round-trip.

    One iteration parses a contextual field address into its structured
    selector and renders the canonical label back -- the string-interpretation
    path every contextual export column and viz lookup pays.
    """
    from diffract.session.resolver import render_label, resolve

    address = f"{CROSS_MODEL_FIELD}@models[{_MODEL_A},{_MODEL_B}]"
    selector = resolve(address)
    if not selector.is_contextual or selector.models != (_MODEL_A, _MODEL_B):
        raise RuntimeError(
            f"Resolver self-check failed for {address!r}; the resolution "
            "loop would time a broken path."
        )

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        render_label(resolve(address))
        samples.append(time.perf_counter() - start)
    return samples


def _self_check() -> None:
    """Fail loudly if the pipeline wiring stops producing what the harness times.

    Guards the honesty of the benchmark: a stage that silently no-ops would
    otherwise report a fast-but-meaningless time, and a no-op stage that
    silently recomputes would report kernel time as lookup time.
    """
    from diffract import Session

    shape = Shape(name="check", n_params=2, dim=8)
    with _silenced(), Session(profile="ram") as session:
        session.compute.configure_kernel(_SVD_KERNEL, allow_cuda=False)
        session.models.add(_build_model(shape, seed=0), model_id=_MODEL_A)
        session.models.add(_build_model(shape, seed=1), model_id=_MODEL_B)
        session.compute.apply(PARAMETER_FIELD)
        session.compute.apply(IN_MODEL_FIELD)
        session.compute.apply(CROSS_MODEL_FIELD)
        metrics = session.results.export(EXPORT_FIELD, export_format="dict")
        aggs = session.results.export_aggregates(
            CROSS_MODEL_FIELD, export_format="dict"
        )
        noop = session.compute.apply(PARAMETER_FIELD)

    if not metrics:
        raise RuntimeError(
            f"Self-check produced no {EXPORT_FIELD!r} metrics; the parameter "
            "stage is not measuring real work."
        )
    if not aggs:
        raise RuntimeError(
            f"Self-check produced no {CROSS_MODEL_FIELD!r} aggregates; the "
            "cross-model stage is not measuring real work."
        )
    if not any(name == PARAMETER_FIELD for name, _ in noop.skipped):
        raise RuntimeError(
            f"Re-applying {PARAMETER_FIELD!r} wrote new fields; the no-op "
            "stages would time kernel compute instead of the skip predicate."
        )

    with _silenced():
        _run_config_store_check()


def _run_config_store_check() -> None:
    """Verify the on-disk store measurement sees both stores as non-empty.

    A rename of either database file would silently zero the size figures;
    this runs a tiny pipeline on the disk profile and demands non-zero bytes.
    """
    shape = Shape(name="check", n_params=2, dim=8)
    with _make_session("local") as (session, store_dir):
        with session:
            session.compute.configure_kernel(_SVD_KERNEL, allow_cuda=False)
            session.models.add(_build_model(shape, seed=0), model_id=_MODEL_A)
            session.compute.apply(PARAMETER_FIELD)
        sizes = _store_sizes(store_dir)

    if sizes["metadata_index_bytes"] <= 0 or sizes["value_store_bytes"] <= 0:
        raise RuntimeError(
            f"Store size self-check read {sizes}; the size probe is not "
            "finding the store files."
        )


# --- Orchestration -----------------------------------------------------------


def _resolve_device(device: str) -> tuple[str, bool]:
    """Map a device request to (recorded_device, allow_cuda).

    ``cpu`` disables the CUDA path; ``cuda`` requires it; ``auto`` records
    whichever the SVD kernel would actually use.
    """
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch is optional; no torch means no CUDA
        cuda_available = False

    if device == "cpu":
        return "cpu", False
    if device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "--device cuda requested but torch.cuda is not available."
            )
        return "cuda", True
    if device == "auto":
        return ("cuda", True) if cuda_available else ("cpu", False)
    raise ValueError(f"Unknown device: {device!r}")


def _run_config(
    profile: str, shape: Shape, allow_cuda: bool, reps: int
) -> dict[str, Any]:
    """Warm up once (discarded), then time ``reps`` reps and sample peak RSS."""
    _run_rep(profile, shape, allow_cuda)  # warm-up: backend first-touch, allocators

    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    store: dict[str, int] | None = None
    tracker = _PeakRSS()
    with tracker:
        baseline_rss = tracker.sample()
        for _ in range(reps):
            timings, sizes = _run_rep(profile, shape, allow_cuda)
            for stage, seconds in timings.items():
                samples[stage].append(seconds)
            store = sizes or store

    peak = tracker.peak_bytes
    return {
        "profile": profile,
        "shape": shape.describe(),
        "stages": {stage: _summarize(samples[stage]) for stage in STAGES},
        "store": store,
        "memory": {
            "rss_method": tracker.method,
            "baseline_rss_mb": round(baseline_rss / 1e6, 2),
            "peak_rss_mb": round(peak / 1e6, 2),
            "delta_rss_mb": round((peak - baseline_rss) / 1e6, 2),
        },
    }


def _environment() -> dict[str, Any]:
    """Machine and library versions needed to interpret the timings."""
    try:
        torch_version: str | None = version("torch")
    except PackageNotFoundError:
        torch_version = None
    try:
        psutil_version: str | None = version("psutil")
    except PackageNotFoundError:
        psutil_version = None
    try:
        diffract_version: str | None = version("diffract-core")
    except PackageNotFoundError:
        diffract_version = None

    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - psutil is optional
        physical = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": physical,
        "numpy": np.__version__,
        "torch": torch_version,
        "psutil": psutil_version,
        "diffract_core": diffract_version,
    }


def run_benchmark(
    *,
    profiles: tuple[str, ...],
    shapes: tuple[Shape, ...],
    device: str,
    reps: int,
    pairs: str | None = None,
) -> dict[str, Any]:
    """Run the full grid of profiles x shapes and return the JSON-ready report."""
    import_start = time.perf_counter()
    import diffract  # noqa: F401 - measured cold import

    import_s = time.perf_counter() - import_start

    _self_check()

    recorded_device, allow_cuda = _resolve_device(device)

    resolver_samples = _time_resolver(RESOLVE_ITERS)

    grid = tuple(product(profiles, shapes))
    results = [_run_config(profile, shape, allow_cuda, reps) for profile, shape in grid]
    if pairs is not None:
        for result, (profile, shape) in zip(results, grid, strict=True):
            result["pair_stages"] = _run_pair_config(
                profile, shape, allow_cuda, reps, pairs
            )

    return {
        "schema": SCHEMA,
        "generated_by": "scripts/bench_pipeline.py",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "protocol": {
            "device": recorded_device,
            "warm": True,
            "reps": reps,
            "tqdm": "disabled",
            "logging": "WARNING",
            "import_diffract_s": round(import_s, 4),
        },
        "resolver": {
            **_summarize(resolver_samples),
            "p95_s": _p95(resolver_samples),
        },
        "results": results,
    }


# --- Regression gate ---------------------------------------------------------


def _index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index a report's configs by (profile, shape name) for comparison."""
    return {(r["profile"], r["shape"]["name"]): r for r in report["results"]}


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    threshold: float,
    abs_floor_s: float,
    metric: str = "median_s",
) -> list[dict[str, Any]]:
    """Compare two reports stage by stage at equal profile/shape/device.

    A stage is a regression only when it is both slower by more than
    ``threshold`` (fractional) and by more than ``abs_floor_s`` (absolute) -- the
    floor keeps sub-millisecond stages from tripping on measurement noise.

    Returns one row per comparable stage; ``regressed`` marks the failures.
    """
    base_device = baseline.get("protocol", {}).get("device")
    cur_device = current.get("protocol", {}).get("device")
    device_mismatch = base_device != cur_device

    base_index = _index(baseline)
    rows: list[dict[str, Any]] = []
    for key, cur in _index(current).items():
        base = base_index.get(key)
        if base is None:
            continue
        for stage in STAGES:
            base_stage = base["stages"].get(stage)
            cur_stage = cur["stages"].get(stage)
            if not base_stage or not cur_stage:
                continue
            base_s = float(base_stage[metric])
            cur_s = float(cur_stage[metric])
            if base_s <= 0:
                continue
            ratio = cur_s / base_s
            delta = cur_s - base_s
            regressed = (
                not device_mismatch
                and (ratio - 1.0) > threshold
                and delta > abs_floor_s
            )
            rows.append(
                {
                    "profile": key[0],
                    "shape": key[1],
                    "stage": stage,
                    "baseline_s": base_s,
                    "current_s": cur_s,
                    "ratio": ratio,
                    "pct_change": (ratio - 1.0) * 100.0,
                    "regressed": regressed,
                    "device_mismatch": device_mismatch,
                }
            )
    return rows


def compare_pairs(
    baseline: dict[str, Any], current: dict[str, Any], *, threshold: float
) -> list[dict[str, Any]]:
    """Compare real single-direction and paired overlap stages at the same shape."""
    if baseline["protocol"]["device"] != current["protocol"]["device"]:
        raise ValueError("Pair timing reports must use the same SVD device")
    base_results = _index(baseline)
    rows = []
    for key, result in _index(current).items():
        base = base_results.get(key)
        single = base.get("pair_stages", {}).get("single") if base else None
        if base is None or base["shape"] != result["shape"] or single is None:
            raise ValueError(
                f"Missing same-shape single-direction pair baseline for {key}"
            )
        for direction in ("forward", "reverse", "both"):
            candidate = result.get("pair_stages", {}).get(direction)
            if candidate is None or not all(
                math.isfinite(sample) and sample > 0
                for sample in (candidate["median_s"], single["median_s"])
            ):
                raise ValueError(f"Missing valid pair timing for {key} {direction}")
            base_s = single["median_s"] * (2 if direction == "both" else 1)
            current_s = candidate["median_s"]
            change = current_s / base_s - 1
            rows.append(
                {
                    "profile": key[0],
                    "shape": key[1],
                    "stage": f"{PAIR_FIELD}_{direction}",
                    "baseline_s": base_s,
                    "current_s": current_s,
                    "ratio": current_s / base_s,
                    "pct_change": change * 100,
                    "regressed": change > threshold,
                    "device_mismatch": False,
                }
            )
    return rows


# --- CLI ---------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable table of the timings and memory."""
    proto = report["protocol"]
    env = report["environment"]
    print(
        f"device={proto['device']} reps={proto['reps']} "
        f"python={env['python']} numpy={env['numpy']} "
        f"import_diffract={proto['import_diffract_s']}s"
    )
    resolver = report.get("resolver")
    if resolver:
        print(
            f"resolve_roundtrip: median={resolver['median_s'] * 1e6:.2f} us  "
            f"p95={resolver['p95_s'] * 1e6:.2f} us  (n={resolver['n']})"
        )
    for result in report["results"]:
        shape = result["shape"]
        mem = result["memory"]
        print(
            f"\n[{result['profile']}/{shape['name']}] "
            f"{shape['n_params']}x{shape['rows']}x{shape['cols']} "
            f"({shape['total_bytes'] / 1e6:.1f} MB)  "
            f"peak_rss={mem['peak_rss_mb']} MB ({mem['rss_method']})"
        )
        store = result.get("store")
        if store:
            print(
                f"    metadata_index={store['metadata_index_bytes'] / 1e6:.2f} MB  "
                f"value_store={store['value_store_bytes'] / 1e6:.2f} MB"
            )
        for stage in STAGES:
            stats = result["stages"][stage]
            print(
                f"    {stage:<22} "
                f"min={stats['min_s'] * 1e3:9.3f} ms  "
                f"median={stats['median_s'] * 1e3:9.3f} ms"
            )
        for direction, stats in result.get("pair_stages", {}).items():
            print(
                f"    {(PAIR_FIELD + '_' + direction):<22} "
                f"min={stats['min_s'] * 1e3:9.3f} ms  "
                f"median={stats['median_s'] * 1e3:9.3f} ms"
            )


def _print_comparison(rows: list[dict[str, Any]], threshold: float) -> bool:
    """Print the baseline comparison; return True if any stage regressed."""
    if not rows:
        print("\nNo comparable (profile, shape) configs between the reports.")
        return False

    if rows[0]["device_mismatch"]:
        print("\nWARNING: device differs between reports; skipping the gate.")

    print(f"\nRegression gate (threshold {threshold * 100:.1f}% per stage):")
    regressed = False
    for row in rows:
        flag = "FAIL" if row["regressed"] else "ok"
        regressed = regressed or row["regressed"]
        print(
            f"    [{row['profile']}/{row['shape']}] {row['stage']:<22} "
            f"{row['pct_change']:+7.1f}%  "
            f"({row['baseline_s'] * 1e3:.3f} -> {row['current_s'] * 1e3:.3f} ms)  "
            f"{flag}"
        )
    return regressed


def _shape_arg(value: str) -> tuple[int, int]:
    """Parse a ``N:DIM`` shape override into (n_params, dim)."""
    try:
        n_str, dim_str = value.split(":")
        return int(n_str), int(dim_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected N:DIM (e.g. 8:512), got {value!r}"
        ) from exc


def _resolve_shapes(args: argparse.Namespace) -> tuple[Shape, ...]:
    """Build the shape pair from defaults, --quick, and explicit overrides."""
    base = QUICK_SHAPES if args.quick else DEFAULT_SHAPES
    large = base[0]
    small = base[1]
    if args.large_few is not None:
        large = Shape("large_few", args.large_few[0], args.large_few[1])
    if args.small_many is not None:
        small = Shape("small_many", args.small_many[0], args.small_many[1])
    return (large, small)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark, write the JSON report, and apply the gate if asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_pipeline_results.json"),
        help="where to write the JSON report (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="a prior report to compare against; fails on regressions",
    )
    parser.add_argument(
        "--profiles",
        default="ram,local",
        help="comma-separated storage profiles (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="SVD device to pin (default: %(default)s)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help="timed reps per config (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="per-stage regression threshold, fractional (default: %(default)s)",
    )
    parser.add_argument(
        "--abs-floor-ms",
        type=float,
        default=0.2,
        help="ignore regressions below this absolute delta (default: %(default)s)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny shapes and fewer reps for a fast smoke run",
    )
    parser.add_argument(
        "--large-few", type=_shape_arg, default=None, help="override N:DIM"
    )
    parser.add_argument(
        "--small-many", type=_shape_arg, default=None, help="override N:DIM"
    )
    parser.add_argument(
        "--pairs",
        choices=("single", "directed"),
        help="measure the isolated overlap stage for the corresponding implementation",
    )
    args = parser.parse_args(argv)

    if args.quick and args.reps == DEFAULT_REPS:
        args.reps = QUICK_REPS

    profiles = tuple(p.strip() for p in args.profiles.split(",") if p.strip())
    shapes = _resolve_shapes(args)

    report = run_benchmark(
        profiles=profiles,
        shapes=shapes,
        device=args.device,
        reps=args.reps,
        pairs=args.pairs,
    )

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    print(f"\nWrote {args.out}")

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        rows = compare(
            baseline,
            report,
            threshold=args.threshold,
            abs_floor_s=args.abs_floor_ms / 1e3,
        )
        if args.pairs == "directed":
            try:
                rows.extend(compare_pairs(baseline, report, threshold=args.threshold))
            except ValueError as err:
                parser.error(str(err))
        if _print_comparison(rows, args.threshold):
            print("\nRegression gate FAILED.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
