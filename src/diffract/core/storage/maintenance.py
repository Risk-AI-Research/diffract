"""Check catalog integrity and collect unreferenced physical values."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex

from .legacy_values import legacy_keys

if TYPE_CHECKING:
    from collections.abc import Iterator

type Address = tuple[str, str]

_SHA256_HEX_LENGTH = 64
_POINTER_COLUMNS = (
    "obj_kind",
    "obj_uid",
    "field",
    "backend",
    "storage_key",
    "tombstoned_at",
    "format_version",
    "run_id",
    "built_at_seq",
    "changed_at_seq",
)


def _address(row: dict[str, Any]) -> Address:
    return row["backend"], row["storage_key"]


def _describe(address: Address) -> dict[str, str]:
    return dict(zip(("backend", "storage_key"), address, strict=True))


@dataclass
class _Snapshot:
    index: SQLiteMetadataIndex
    stores: dict[str, Any]
    keys: set[Address]
    missing_arrays: set[Address]
    rows: list[dict[str, Any]]

    def orphan_keys(self) -> set[Address]:
        return self.keys.difference(map(_address, self.rows))

    def live_missing(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows
            if row["tombstoned_at"] is None
            and (_address(row) not in self.keys or _address(row) in self.missing_arrays)
        ]


@contextmanager
def _open_snapshot(
    path: Path, config: Path | None, *, readonly: bool
) -> Iterator[_Snapshot]:
    from diffract.core.data.metadata.store_migration import resolve_store_layout

    index_path, specs = resolve_store_layout(path, config)
    with (
        closing(SQLiteMetadataIndex(str(index_path))) as index,
        index,
        ExitStack() as stack,
    ):
        stores = {
            spec.name: stack.enter_context(closing(spec.open(readonly=readonly)))
            for spec in specs
        }
        selector = {spec.name: spec for spec in specs}
        keys = {
            (name, key)
            for name, store in stores.items()
            for key in store.catalog_blob_keys()
        }
        keys.update(
            (name, selector[name].old_key(table, field, uid))
            for name, table, field, uid in legacy_keys(specs)
        )
        yield _Snapshot(
            index,
            stores,
            keys,
            {
                (spec.name, key)
                for spec in specs
                if spec.backend == "sqlite"
                for key in stores[spec.name].missing_catalog_arrays()
            },
            index.query_rows("field_values", columns=_POINTER_COLUMNS),
        )


def _run_config_valid(run: dict[str, Any]) -> bool:
    fingerprint = run["config_hash"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != _SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        return False
    try:
        return isinstance(json.loads(run["config_json"]), dict)
    except (TypeError, ValueError):
        return False


def check_store(path: Path, *, config: Path | None = None) -> dict[str, list]:
    """Report missing values, orphan blobs, and invalid live metadata."""
    with _open_snapshot(path, config, readonly=True) as store:
        runs = {r["run_id"]: r for r in store.index.query_rows("runs")}
        incomplete = []
        invalid_edges = []
        for row in store.rows:
            if row["tombstoned_at"] is not None:
                continue
            run = runs.get(row["run_id"])
            if run is None and row["run_id"] is None and row["format_version"] == 0:
                continue
            if run is None or run["status"] != "COMPLETE":
                incomplete.append(row)
            if (
                run is not None
                and row["built_at_seq"] is not None
                and (
                    row["built_at_seq"] != run["seq"]
                    or (
                        row["changed_at_seq"] is not None
                        and row["changed_at_seq"] > run["seq"]
                    )
                )
            ):
                invalid_edges.append(row)
        return {
            "missing_blobs": store.live_missing(),
            "orphan_blobs": [_describe(key) for key in sorted(store.orphan_keys())],
            "unknown_owners": store.index.catalog_owner_gaps(),
            "incomplete_runs": incomplete,
            "invalid_generations": invalid_edges,
            "invalid_run_configs": sorted(
                run["run_id"]
                for run in runs.values()
                if run["kind"] == "kernel" and not _run_config_valid(run)
            ),
        }


def _read_observations(path: Path) -> dict[Address, datetime]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (backend, key): datetime.fromisoformat(observed)
        for backend, key, observed in rows
    }


def _write_observations(path: Path, values: dict[Address, datetime]) -> None:
    if not values:
        path.unlink(missing_ok=True)
        return
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temp:
        temp_path = Path(temp.name)
        try:
            rows = [(*key, values[key].isoformat()) for key in sorted(values)]
            json.dump(rows, temp)
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def collect_store(
    path: Path,
    *,
    config: Path | None = None,
    dry_run: bool = False,
    grace_days: int = 14,
) -> dict[str, Any]:
    """Plan or apply removal of aged tombstones and observed orphan blobs."""
    if grace_days < 0:
        raise ValueError("GC grace period must be non-negative.")

    with _open_snapshot(path, config, readonly=dry_run) as store:
        if store.live_missing() or any(
            _address(row)[0] not in store.stores for row in store.rows
        ):
            raise ValueError(
                "The value store is missing catalog data. Verify its configuration "
                "with fsck before running gc."
            )
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=grace_days)
        sidecar = Path(store.index.database_path + ".gc.json")
        previous = _read_observations(sidecar)
        observations = {key: previous.get(key, now) for key in store.orphan_keys()}
        orphans = {key for key, seen in observations.items() if seen <= cutoff}
        tombstones = [
            row
            for row in store.rows
            if row["tombstoned_at"] is not None
            and datetime.fromisoformat(row["tombstoned_at"]) <= cutoff
        ]
        refs = Counter(map(_address, store.rows))
        refs.subtract(map(_address, tombstones))
        blobs = sorted(
            orphans
            | {
                key
                for key in map(_address, tombstones)
                if key in store.keys and not refs[key]
            }
        )
        if not dry_run:
            # The metadata writer lock excludes any publisher until deletion finishes.
            for row in tombstones:
                store.index.delete_rows(
                    "field_values",
                    where={
                        col: row[col] for col in ("obj_uid", "field", "tombstoned_at")
                    },
                )
            for backend, key in blobs:
                store.stores[backend].delete_blob(key)
            _write_observations(
                sidecar,
                {key: seen for key, seen in observations.items() if key not in orphans},
            )
        return {
            "tombstones": tombstones,
            "blobs": [_describe(key) for key in blobs],
        }
