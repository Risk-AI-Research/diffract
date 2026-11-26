"""Storage-backend x profile-arrangement compatibility matrix.

The standing gate for storage-affecting work: every scenario runs on every
supported (arrangement, value backend) cell, where an arrangement is the
metadata-index/cache tier shape of a shipped profile and the value backend
is overridden per cell. Storage-touching changes plug new scenarios into
this module; the expected-grid literals below are the ground truth a
meta-test enforces, so a cell that silently stops running is itself a
failure — exemptions are enumerated with reasons, never discovered as
skips.

Cells synthesize a config INI per run (the DI container builds any backend
from a plain config dict); no new profiles are involved.

The S3-backed Zarr cell lives in tests/integration/test_backend_matrix_s3.py
(marker-gated there); it is named here so the whole picture stays in one
place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import diffract.core.utils.imports as import_utils
from diffract.containers import create_main_container
from diffract.session import Session

pytestmark = pytest.mark.unit

ARRANGEMENTS = ("ram", "local")
VALUE_BACKENDS = ("sqlite", "hdf5", "zarr", "ram", "hybrid")

# Small enough that the seeded model's large matrix crosses it, so hybrid
# cells exercise both the light store and the heavy store.
HYBRID_THRESHOLD = 4096


@dataclass(frozen=True)
class Cell:
    """One matrix cell: a profile-shaped arrangement x a value backend."""

    arrangement: str
    value_backend: str

    @property
    def id(self) -> str:
        return f"{self.arrangement}-{self.value_backend}"


ALL_CELLS = tuple(
    Cell(arrangement, backend)
    for arrangement in ARRANGEMENTS
    for backend in VALUE_BACKENDS
)

# Expected combinations are independent of the derived grid and its exemptions.
FULL_GRID = (
    "ram-sqlite",
    "ram-hdf5",
    "ram-zarr",
    "ram-ram",
    "ram-hybrid",
    "local-sqlite",
    "local-hdf5",
    "local-zarr",
    "local-ram",
    "local-hybrid",
)

REOPEN_CELLS = (
    "local-sqlite",
    "local-hdf5",
    "local-zarr",
    "local-hybrid",
)

REOPEN_EXEMPTIONS = {
    "ram-sqlite": "metadata index is :memory: — nothing persists by design",
    "ram-hdf5": "metadata index is :memory: — nothing persists by design",
    "ram-zarr": "metadata index is :memory: — nothing persists by design",
    "ram-ram": "metadata index is :memory: — nothing persists by design",
    "ram-hybrid": "metadata index is :memory: — nothing persists by design",
    "local-ram": "value store is process memory — reopen has nothing to read",
}


def cells(ids: tuple[str, ...]) -> list[Cell]:
    """The Cell objects for a tuple of expected-grid cell ids."""
    by_id = {cell.id: cell for cell in ALL_CELLS}
    return [by_id[cell_id] for cell_id in ids]


# Execution ledger for the zero-silent-skips meta-test: every cell test
# records its attempt before the extras gate and its execution after it.
_ATTEMPTED: dict[str, set[str]] = {}
_EXECUTED: dict[str, set[str]] = {}


def _track_attempt(scenario: str, cell: Cell) -> None:
    _ATTEMPTED.setdefault(scenario, set()).add(cell.id)


def _track_executed(scenario: str, cell: Cell) -> None:
    _EXECUTED.setdefault(scenario, set()).add(cell.id)


# ---------------- Cell construction -----------------------------------------


def _write_cell_config(directory: Path, cell: Cell) -> Path:
    """Synthesize the INI for a cell in an isolated directory."""
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
        f'backend = "{cell.value_backend}"',
    ]
    if cell.value_backend in ("sqlite", "hybrid"):
        lines += [
            "[storage.sqlite]",
            f"path = {json.dumps(str(directory / 'storage.db'))}",
        ]
    if cell.value_backend in ("hdf5", "hybrid"):
        lines += [
            "[storage.hdf5]",
            f"path = {json.dumps(str(directory / 'storage.h5'))}",
        ]
    if cell.value_backend == "zarr":
        lines += [
            "[storage.zarr]",
            f"store_url = {json.dumps(str(directory / 'zarr_store'))}",
        ]
    if cell.value_backend == "hybrid":
        lines += [
            "[storage.hybrid]",
            'light = "sqlite"',
            'heavy = "hdf5"',
            f"array_threshold = {HYBRID_THRESHOLD}",
        ]
    lines += ["[metadata]", 'backend = "sqlite"', "[metadata.sqlite]"]
    if cell.arrangement == "ram":
        lines += ['path = ":memory:"', "[cache]", 'backend = "none"']
    else:
        lines += [
            f"path = {json.dumps(str(directory / 'metadata.db'))}",
            "[cache]",
            'backend = "simple"',
            "[cache.simple]",
            "max_memory_mb = 64",
        ]
    lines += [
        "[nn.extractor]",
        "skip_not_implemented_types = true",
        "[export]",
        'default_export_format = "dict"',
        "",
    ]
    config = directory / "cell.ini"
    config.write_text("\n".join(lines))
    return config


def _require_cell_extras(cell: Cell) -> None:
    if cell.value_backend != "zarr":
        return
    if not import_utils.is_available("zarr"):
        pytest.skip(
            f"cell {cell.id}: zarr not installed (pip install "
            '"diffract-core[zarr]") — the executed-cell meta-test reports '
            "the coverage loss"
        )
    import zarr

    if int(zarr.__version__.split(".")[0]) < 3:
        pytest.fail(
            f"cell {cell.id}: zarr-python {zarr.__version__} is the 2.x "
            "line — the matrix supports zarr>=3 only (the dev extra pins "
            "zarr>=3.0.0)"
        )


def _seed_model() -> dict[str, np.ndarray]:
    """Two matrices straddling the hybrid threshold, deterministic."""
    rng = np.random.default_rng(7)
    return {
        "enc.weight": rng.standard_normal((8, 8)).astype("float32"),  # light
        "head.weight": rng.standard_normal((64, 64)).astype("float32"),  # heavy
    }


def _stored_weights(session: Session) -> dict[str, np.ndarray]:
    exported = session.results.export("weights", export_format="dict")
    return {
        entry["metadata"]["name"]: entry["fields"]["weights"]
        for entry in exported.values()
    }


# ---------------- Scenario: identity round-trip ------------------------------


def _assert_frob_generation(container, uids) -> None:
    store = container.storage.storage_manager()
    run_ids = set()
    for uid in uids:
        generation = store.catalog.get("parameter", uid, "frob_norm").generation
        assert generation is not None
        assert generation.output_name == "frob_norm"
        assert generation.built_at_seq == generation.changed_at_seq
        row = store.catalog.index.find_row(
            "runs",
            columns=("kind", "kernel_name", "kernel_version", "status", "seq"),
            where={"run_id": generation.run_id},
        )
        assert row == {
            "kind": "kernel",
            "kernel_name": "frob_norm",
            "kernel_version": "2",
            "status": "COMPLETE",
            "seq": generation.built_at_seq,
        }
        run_ids.add(generation.run_id)
    assert len(run_ids) == 1


@pytest.mark.parametrize("cell", cells(FULL_GRID), ids=lambda c: c.id)
def test_identity_roundtrip(cell: Cell, tmp_path: Path) -> None:
    """Write -> read back by (model_id, name) within one session, per cell."""
    _track_attempt("identity_roundtrip", cell)
    _track_attempt("generation", cell)
    _require_cell_extras(cell)
    _track_executed("identity_roundtrip", cell)
    config = _write_cell_config(tmp_path, cell)
    model = _seed_model()

    container = create_main_container(str(config))
    with Session(container=container) as session:
        session.models.add(model, model_id="m1")
        session.compute.apply("frob_norm")

        stored = _stored_weights(session)
        assert set(stored) == set(model)
        for name, weights in model.items():
            np.testing.assert_array_equal(stored[name], weights)

        metrics = session.results.export_metrics("frob_norm", export_format="dict")
        by_name = {
            entry["metadata"]["name"]: entry["fields"]["frob_norm"]
            for entry in metrics.values()
        }
        for name, weights in model.items():
            assert by_name[name] == pytest.approx(
                float(np.linalg.norm(weights)), rel=1e-6
            )
        _assert_frob_generation(container, metrics)
    _track_executed("generation", cell)


def _assert_checkpoint_values(session: Session) -> None:
    rows = session.results.export_metrics(
        "weights", "weights_svals", "frob_norm", "l2_norm", export_format="list"
    )
    assert [row["step"] for row in rows] == [0, 1]
    for row, scale in zip(rows, (1, 2), strict=True):
        np.testing.assert_array_equal(row["weights"], np.diag([3.0, 4.0]) * scale)
        np.testing.assert_allclose(row["weights_svals"], [3.0 * scale, 4.0 * scale])
        assert row["frob_norm"] == pytest.approx(5.0 * scale)
        assert row["l2_norm"] == pytest.approx(4.0 * scale)


@pytest.mark.parametrize("cell", cells(FULL_GRID), ids=lambda c: c.id)
def test_checkpoint_and_export_ownership(cell: Cell, tmp_path: Path) -> None:
    _track_attempt("checkpoint_ownership", cell)
    _require_cell_extras(cell)
    config = _write_cell_config(tmp_path, cell)
    with Session(config_path=config) as session:
        source = np.zeros((2, 4))
        weights = source[:, ::2]
        weights[:] = np.diag([3.0, 4.0])
        session.models.add({"w": weights}, model_id="m", step=0)
        source *= 2
        session.models.add({"w": weights}, model_id="m", step=1)
        source.fill(-9)
        session.compute.apply("frob_norm")
        first = session.filter(steps=[0])
        (exported,) = first.results.export_metrics(
            "weights", "weights_svals", export_format="list"
        )
        for field in ("weights", "weights_svals"):
            array = exported[field]
            if array.flags.writeable:
                array.fill(0)
        (status,) = first.results.status("frob_norm")
        assert status.status == "fresh"
        session.compute.apply("l2_norm")
        _assert_checkpoint_values(session)
    if cell.id in REOPEN_CELLS:
        with Session(config_path=config) as reopened:
            _assert_checkpoint_values(reopened)
    _track_executed("checkpoint_ownership", cell)


@pytest.mark.parametrize("cell", cells(FULL_GRID), ids=lambda c: c.id)
def test_step_trace_and_group_erasure(cell: Cell, tmp_path: Path) -> None:
    """Verify freshness, group tombstones, and recompute in each backend."""
    _track_attempt("step_trace_and_group_erasure", cell)
    _require_cell_extras(cell)
    config = _write_cell_config(tmp_path, cell)
    with Session(config_path=config) as session:
        session.models.add(_seed_model(), model_id="m1")

        @session.compute.kernel(
            require_fields=("weights",),
            produce_fields=("matrix_score", "matrix_sibling"),
        )
        def matrix_pair(
            weights: np.ndarray, *, scale: float = 1.0
        ) -> tuple[float, np.ndarray]:
            return float(weights.sum()) * scale, weights * scale

        scoped = session.filter(param_names=["head.weight"])
        (initial,) = scoped.results.status("matrix_score")
        assert initial.status == "never computed"
        session.compute.apply("matrix_score")
        (fresh,) = scoped.results.status("matrix_score")
        assert fresh.status == "fresh"

        session.compute.configure_kernel("matrix_pair", scale=2.0)
        (stale,) = scoped.results.status("matrix_score")
        assert (stale.status, stale.reason) == ("stale", "config key scale changed")
        scoped.compute.apply("matrix_score")
        (fresh,) = scoped.results.status("matrix_score")
        assert fresh.status == "fresh"

        summary = scoped.results.erase("matrix_score")
        assert summary.fields == ("matrix_score", "matrix_sibling")
        assert summary.affected_uids == 1
        (missing,) = scoped.results.status("matrix_score")
        assert missing.status == "never computed"
        untouched = session.filter(param_names=["enc.weight"])
        before = untouched.results.export_metrics(
            "matrix_sibling", export_format="dict"
        )
        assert len(before) == 1
        scoped.compute.apply("matrix_score")
        values = scoped.results.export_metrics("matrix_sibling", export_format="dict")
        (row,) = values.values()
        np.testing.assert_array_equal(
            row["fields"]["matrix_sibling"], _seed_model()["head.weight"] * 2
        )
    _track_executed("step_trace_and_group_erasure", cell)


@pytest.mark.parametrize("cell", cells(FULL_GRID), ids=lambda c: c.id)
def test_catalog_visibility_and_resurrection(cell: Cell, tmp_path: Path) -> None:
    """Live reads, tombstones and rewrites resolve through the stored pointer."""
    _track_attempt("catalog", cell)
    _require_cell_extras(cell)
    container = create_main_container(str(_write_cell_config(tmp_path, cell)))
    with Session(container=container) as session:
        session.models.add(_seed_model(), model_id="m1")
        repo = container.nn.parameter_repository()
        store = container.storage.storage_manager()
        first, second = repo.list_uids()
        field = "metric@scope[1]"
        store.set_field(first, field, 4, table="parameters")
        pointer = store.catalog.get("parameter", first, field)
        assert pointer.storage_key == f"{first}/metric%40scope%5B1%5D"
        assert pointer.format_version == 1
        assert store.get_field(first, field, table="parameters") == 4
        proxy = repo.get_proxy(first)
        assert proxy.get_field(field) == 4
        if repo.cache_manager is not None:
            assert repo.cache_manager.has_field(first, field)
        store.set_field(second, field, 6, table="parameters")
        assert store.catalog.tombstone("parameter", uid=first, field=field) == 1
        assert store.catalog.tombstone("parameter", uid=first, field=field) == 0
        with pytest.raises(KeyError):
            store.get_field(first, field, table="parameters")
        assert not proxy.has_field(field)
        with pytest.raises(KeyError):
            proxy.get_field(field)
        assert field not in store.list_fields(first, table="parameters")
        assert store.list_objs_has_field(field, table="parameters") == [second]
        assert store.get_field_metadata(first, field, table="parameters") is None
        physical = container.storage.physical_store()
        original = physical.light if cell.value_backend == "hybrid" else physical
        assert original.read_blob(pointer.storage_key) == 4
        replacement = np.arange(HYBRID_THRESHOLD, dtype="int32")
        store.set_field(first, field, replacement, table="parameters")
        live = store.catalog.get("parameter", first, field)
        assert live.size_bytes == replacement.nbytes
        if cell.value_backend == "hybrid":
            assert pointer.backend == "light/sqlite"
            assert live.backend == "heavy/hdf5"
        assert live.storage_key == pointer.storage_key
        np.testing.assert_array_equal(
            store.get_field(first, field, table="parameters"), replacement
        )
    _track_executed("catalog", cell)


# ---------------- Scenario: reopen and read by natural key -------------------


@pytest.mark.parametrize("cell", cells(REOPEN_CELLS), ids=lambda c: c.id)
def test_reopen_reads_by_natural_key(cell: Cell, tmp_path: Path) -> None:
    """Close the store, reopen it in a fresh session, read by (model, name)."""
    _track_attempt("reopen", cell)
    _track_attempt("generation_reopen", cell)
    _require_cell_extras(cell)
    _track_executed("reopen", cell)
    config = _write_cell_config(tmp_path, cell)
    model = _seed_model()

    with Session(config_path=str(config)) as session:
        session.models.add(model, model_id="m1")
        session.compute.apply("frob_norm")

    container = create_main_container(str(config))
    with Session(container=container) as reopened:
        stored = _stored_weights(reopened)
        assert set(stored) == set(model)
        for name, weights in model.items():
            np.testing.assert_array_equal(stored[name], weights)

        metrics = reopened.results.export_metrics("frob_norm", export_format="dict")
        assert {e["metadata"]["name"] for e in metrics.values()} == set(model)
        assert reopened.models.list() == ["m1"]
        _assert_frob_generation(container, metrics)
    _track_executed("generation_reopen", cell)


# ---------------- The expected-grid meta-tests -------------------------------


def test_expected_grid_matches_the_cell_axes() -> None:
    """The literal grid, the derived cells, and the exemptions must agree.

    This is the zero-silent-skips enforcement: shrinking an axis, widening
    an exemption, or renaming a cell breaks the equalities below instead of
    quietly collecting fewer tests.
    """
    derived = {cell.id for cell in ALL_CELLS}
    assert derived == set(FULL_GRID)
    assert len(FULL_GRID) == len(ARRANGEMENTS) * len(VALUE_BACKENDS)

    # Reopen coverage + enumerated exemptions partition the full grid.
    assert set(REOPEN_CELLS) | set(REOPEN_EXEMPTIONS) == set(FULL_GRID)
    assert not set(REOPEN_CELLS) & set(REOPEN_EXEMPTIONS)

    # Every exemption carries a visible reason.
    for cell_id, reason in REOPEN_EXEMPTIONS.items():
        assert cell_id in FULL_GRID
        assert len(reason) > 20, f"exemption for {cell_id} needs a real reason"


def test_zero_silent_skips_across_executed_cells() -> None:
    """Executed cells must equal the expected grid — a skip cannot hide.

    Meaningful when the whole module ran in one process (serial, no -k
    selection); under partial selection or xdist sharding it steps aside
    with a visible reason instead of guessing.
    """
    expected = {
        "identity_roundtrip": set(FULL_GRID),
        "checkpoint_ownership": set(FULL_GRID),
        "catalog": set(FULL_GRID),
        "generation": set(FULL_GRID),
        "step_trace_and_group_erasure": set(FULL_GRID),
        "reopen": set(REOPEN_CELLS),
        "generation_reopen": set(REOPEN_CELLS),
    }
    for scenario, expected_ids in expected.items():
        attempted = _ATTEMPTED.get(scenario, set())
        if attempted != expected_ids:
            pytest.skip(
                f"{scenario}: {len(attempted)}/{len(expected_ids)} cells "
                "attempted in this process — the coverage check needs the "
                "full module in one run"
            )
        silent = expected_ids - _EXECUTED.get(scenario, set())
        assert not silent, (
            f"{scenario}: cells attempted but never executed "
            f"{sorted(silent)} — a skipped cell is a red result here; "
            "extras absence is not an enumerated exemption"
        )


@pytest.mark.parametrize("erase_first", [False, True])
def test_open_reader_refreshes_after_another_process_writes(tmp_path, erase_first):
    import subprocess
    import sys

    cfg = _write_cell_config(tmp_path, Cell("local", "sqlite"))
    container = create_main_container(str(cfg))
    with Session(container=container) as session:
        session.models.add(_seed_model(), model_id="m1")
        repo = container.nn.parameter_repository()
        proxy = repo.get_proxy(repo.list_uids()[0])
        proxy.set_field("score", 4)
        assert proxy.get_field("score") == 4
        assert proxy.is_field_prefetched("score")
        child = """
import sys
from diffract import Session
from diffract.containers import create_main_container
container = create_main_container(sys.argv[1])
with Session(container=container):
    repo = container.nn.parameter_repository()
    proxy = repo.get_proxy(sys.argv[2])
    if sys.argv[3] == "True":
        proxy.erase_field("score")
    proxy.set_field("score", 9)
"""
        subprocess.run(  # noqa: S603 - fixed local regression fixture
            [sys.executable, "-c", child, str(cfg), proxy.meta.uid, str(erase_first)],
            check=True,
        )
        assert not proxy.is_field_prefetched("score")
        assert proxy.get_field("score") == 9
        assert proxy.is_field_prefetched("score")


def test_late_cache_publication_cannot_replace_newer_revision(tmp_path, monkeypatch):
    cfg = _write_cell_config(tmp_path, Cell("local", "sqlite"))
    first, second = (create_main_container(str(cfg)) for _ in range(2))
    with Session(container=first) as session, Session(container=second):
        session.models.add(_seed_model(), model_id="m1")
        repo = first.nn.parameter_repository()
        proxy = repo.get_proxy(repo.list_uids()[0])
        store = repo.storage_manager
        store.set_field(proxy.meta.uid, "score", 4, table="parameters")
        original = store.read_pointer

        def interrupted_read(pointer):
            value = original(pointer)
            second.storage.storage_manager().set_field(
                proxy.meta.uid, "score", 9, table="parameters"
            )
            return value

        with monkeypatch.context() as patch:
            patch.setattr(store, "read_pointer", interrupted_read)
            assert proxy.get_field("score") == 4
        assert proxy.get_field("score") == 9


@pytest.mark.parametrize("cell", cells(FULL_GRID), ids=lambda c: c.id)
def test_clear_removes_raw_tables_and_keeps_physical_catalog_values(tmp_path, cell):
    _require_cell_extras(cell)
    container = create_main_container(str(_write_cell_config(tmp_path, cell)))
    with Session(container=container):
        store = container.storage.storage_manager()
        owner = "a" * 32
        store.set_field(owner, "catalog", 4, table="parameters")
        pointer = store.catalog.get("parameter", owner, "catalog")
        # The raw table name can coincide with the physical catalog owner group.
        for table in ("custom", owner):
            store.set_field("raw-uid", "aux", 35, table=table)
        store.clear()
        assert not store.has_field(owner, "catalog", table="parameters")
        assert store._stores[pointer.backend].read_blob(pointer.storage_key) == 4
        for table in ("custom", owner):
            assert not store.has_field("raw-uid", "aux", table=table)


@pytest.mark.parametrize("inside_batch", [False, True])
def test_existing_reader_recovers_interrupted_same_size_overwrite(
    tmp_path, monkeypatch, inside_batch
):
    import subprocess
    import sys

    cfg = _write_cell_config(tmp_path, Cell("local", "sqlite"))
    container = create_main_container(str(cfg))
    with Session(container=container) as session:
        session.models.add(_seed_model(), model_id="m1")
        repo = container.nn.parameter_repository()
        proxy = repo.get_proxy(repo.list_uids()[0])
        proxy.set_field("sized", np.zeros(1, dtype=np.float32))
        proxy.set_field("score", 4)
        assert proxy.get_field("score") == 4
        child = """
import os
import sys

import numpy as np

from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.storage.catalog_manager import CatalogStorageManager
from diffract.core.storage.sqlite_manager import SQLiteStorageManager


class Interrupted(SQLiteStorageManager):
    def __exit__(self, *args):
        super().__exit__(*args)
        if not self._context_depth and args[0] is None:
            os._exit(89)


store = CatalogStorageManager(
    Interrupted(sys.argv[2]), SQLiteMetadataIndex(sys.argv[1])
)
with store:
    store.set_field(sys.argv[3], "score", 9, table="parameters")
    store.set_field(
        sys.argv[3], "sized", np.arange(20, dtype=np.float32), table="parameters"
    )
"""

        def lose_writer():
            result = subprocess.run(  # noqa: S603 - fixed local crash fixture
                [
                    sys.executable,
                    "-c",
                    child,
                    str(tmp_path / "metadata.db"),
                    str(tmp_path / "storage.db"),
                    proxy.meta.uid,
                ],
                check=False,
                timeout=30,
            )
            assert result.returncode == 89

        def assert_new_size():
            uid = proxy.meta.uid
            assert repo.storage_manager.read_sizes(
                {uid: ["sized"]}, table="parameters"
            ) == {(uid, "sized"): 80}

        if inside_batch:
            begin = repo.metadata_index._begin_write

            def enter_after_crash():
                monkeypatch.setattr(repo.metadata_index, "_begin_write", begin)
                lose_writer()
                begin()

            monkeypatch.setattr(repo.metadata_index, "_begin_write", enter_after_crash)

            def rollback_batch():
                with repo:
                    assert_new_size()
                    assert proxy.get_field("score") == 9
                    raise RuntimeError("rollback")

            with pytest.raises(RuntimeError, match="rollback"):
                rollback_batch()
        else:
            lose_writer()
            assert_new_size()
        assert proxy.get_field("score") == 9


def test_catalog_point_read_uses_its_own_uncommitted_revision(tmp_path):
    cfg = _write_cell_config(tmp_path, Cell("local", "sqlite"))
    container = create_main_container(str(cfg))
    with Session(container=container) as session:
        session.models.add(_seed_model(), model_id="m1")
        repo = container.nn.parameter_repository()
        proxy = repo.get_proxy(repo.list_uids()[0])
        uid = proxy.meta.uid
        proxy.set_field("score", 4)
        assert proxy.get_field("score") == 4
        with repo:
            proxy.set_field("score", 9)
            repo.cache_manager.erase_field(uid, "score")
            assert proxy.get_field("score") == 9
            repo.storage_manager.set_field(uid, "fresh", 17, table="parameters")
            assert proxy.get_field("fresh") == 17
        assert proxy.get_field("score") == 9
        assert proxy.get_field("fresh") == 17
        with Session(config_path=cfg) as reader:
            actual = reader.results.export_metrics(
                "score", "fresh", export_format="dict"
            )
            assert actual[uid]["fields"] == {"score": 9, "fresh": 17}
