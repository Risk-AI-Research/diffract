# Storage and Cache

Configure storage and cache backends for different workloads.

For profile selection, see [Configuration](../../reference/configuration.md).

**Note:** The `ini` blocks in the backend sections below are fragments — each shows only the
storage or cache portion of a config. A complete session config also requires `[metadata]`
(with `[metadata.sqlite]`), `[cache]`, and `[nn.extractor]` sections. See the
[complete minimal example](#complete-minimal-example) below, or start from the complete
configs shipped in `src/diffract/configs/` (`sqlite.ini`, `hybrid.ini`,
`fast_speed_without_disk.ini`).

## Stored fields and recovery

The metadata index records the backend and physical key of each field. New
fields use two path components, the owner UID and the encoded field name.
An erased field disappears from reads immediately. Computing or writing it
again makes it visible through the same logical address.

The value store writes a new field before the index publishes its pointer.
If a process exits between these operations, retrying the write replaces the
unreferenced value. An in-place overwrite can expose the replacement before
the index commits. When its indexed byte count or catalog format version changes, a small
recovery record beside the index lets the next session reconcile the
descriptor with the surviving value. A backup of an interrupted store must include this
`.pending` directory if present. RAM storage lasts for the life of its manager.

## Complete minimal example

SQLite storage with an in-process cache. Save as `diffract.ini` and pass it to
`Session(config_path="diffract.ini")`:

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[metadata]
backend = "sqlite"

[metadata.sqlite]
path = "data/metadata.db"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 512

[nn.extractor]
skip_not_implemented_types = true
```

## Storage backends

### RAM

In-memory storage. Fast, no persistence.

```ini
[storage]
backend = "ram"
```

**Use case:** Quick experiments, CI tests.

### SQLite

Single-file database. Good balance of speed and persistence.

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"
```

**Use case:** Local development, small to medium models.

### HDF5

Optimized for large numerical arrays. Serialized writes, concurrent reads.

```ini
[storage]
backend = "hdf5"

[storage.hdf5]
path = "data/arrays.h5"
```

**Use case:** Array-heavy workloads where you don't need metadata queries.

### Hybrid (SQLite + HDF5)

SQLite for metadata and small values, HDF5 for large arrays. Metadata queries stay fast
while large arrays don't bloat the SQLite file: values of `array_threshold` bytes or more
are routed to the `heavy` backend, everything else to `light`.

```ini
[storage]
backend = "hybrid"

[storage.hybrid]
light = "sqlite"
heavy = "hdf5"
array_threshold = 1048576

[storage.sqlite]
path = "data/metadata.db"

[storage.hdf5]
path = "data/arrays.h5"
```

**Use case:** Large models, production setups.

### Zarr (fsspec)

Zarr is a good fit for *cloud-native* array storage. It can store chunks as separate objects and
works with many backends via `fsspec`.

```ini
[storage]
backend = "zarr"

[storage.zarr]
store_url = ".diffract/zarr_store"
```

**Use case:** Remote object stores (S3/GCS/Azure), distributed filesystems, and array-heavy workloads.

### Hybrid (SQLite + Zarr)

SQLite for fast local metadata queries; Zarr for large arrays (often remote).

```ini
[storage]
backend = "hybrid"

[storage.hybrid]
light = "sqlite"
heavy = "zarr"
array_threshold = 1048576

[storage.sqlite]
path = ".diffract/metadata.db"

[storage.zarr]
store_url = "s3://my-bucket/diffract-data"
```

**Use case:** Local metadata queries with arrays in a remote object store, behind the same API.

## Cache backends

### None

No caching. Every read goes to storage.

```ini
[cache]
backend = "none"
```

**Use case:** Memory-constrained environments, write-heavy workloads.

### Simple (in-process LRU)

Process-local LRU cache. No external dependencies.

```ini
[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 256
ttl_seconds = 3600
```

**Use case:** Notebooks, single-process scripts.

### Redis

Shared cache across processes. Requires the `redis` extra.

```bash
# Install
uv sync --extra redis

# Start Redis (Docker)
docker run --rm -p 6379:6379 redis:7
```

```ini
[cache]
backend = "redis"

[cache.redis]
host = "localhost"
port = 6379
db = 0
max_memory_mb = 4096
ttl_seconds = 3600
key_prefix = "diffract:cache:"
```

**Use case:** Multi-process compute, shared cache across runs.

## Recommended setups

These blocks show the storage and cache sections; a complete config additionally needs the
`[metadata]` and `[nn.extractor]` sections from the
[complete minimal example](#complete-minimal-example).

### Local development

SQLite + simple cache. Works everywhere, no external services.

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 512
```

### Large models

Hybrid storage prevents SQLite bloat from large weight arrays.

```ini
[storage]
backend = "hybrid"

[storage.hybrid]
light = "sqlite"
heavy = "hdf5"
array_threshold = 1048576

[storage.sqlite]
path = "data/meta.db"

[storage.hdf5]
path = "data/arrays.h5"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 2048
```

### Multi-process / cluster

Hybrid + Redis for shared caching across workers.

```ini
[storage]
backend = "hybrid"

[storage.hybrid]
light = "sqlite"
heavy = "hdf5"
array_threshold = 1048576

[storage.sqlite]
path = "data/meta.db"

[storage.hdf5]
path = "data/arrays.h5"

[cache]
backend = "redis"

[cache.redis]
host = "redis-server"
port = 6379
max_memory_mb = 8192
key_prefix = "diffract:myproject:"
```

## Concurrency notes

- **SQLite on a local filesystem:** Multiple processes on one machine can
  write to a shared store when SQLite holds both metadata and values. The
  metadata index uses WAL with a bounded wait for the write lock and starts
  each write transaction with `BEGIN IMMEDIATE`. If two processes add the same
  model and checkpoint step, one succeeds and the other receives
  `ModelAlreadyExistsError`. For jobs on separate machines, write to separate
  stores and merge the sessions after the jobs finish.
- **HDF5:** Serialized writes, concurrent reads. Avoid multiple independent writers.
- **Redis:** Configured for RAM-only LRU eviction. Data is not persisted.

## Store consistency and maintenance

A write commits the value store before the metadata index publishes its
pointer. These stores commit independently. A process that stops between
the two commits can leave an orphaned payload. A published value whose run
lacks a completion record is eligible for recomputation. Back up a closed
store using the SQLite backup API or `VACUUM INTO` for the metadata index, and
include its value stores in the same backup.

Erasing a field through `results.erase()` hides the field and its kernel
siblings for the same execution group. Erasure leaves the physical values
for garbage collection. `results.erase(erase_all=True)` also erases ingested fields,
including weights, in the current scope.

Close sessions using the store before running maintenance. Pass the session's
INI or JSON file with `--config` when the value files use custom paths or
options.

```console
python -m diffract.store fsck /data/run/metadata_index.db --config /data/run/session.ini
python -m diffract.store gc /data/run/metadata_index.db --config /data/run/session.ini --dry-run
python -m diffract.store gc /data/run/metadata_index.db --config /data/run/session.ini
```

`fsck` reports missing and orphaned values, unresolved owners, incomplete
producing runs and inconsistent generation clocks. It exits with status 1
when it finds an issue. `gc` collects tombstones after 14 days by default.
For an orphan, the grace period starts when a collection run first records
its key in the index's `.gc.json` sidecar. Preview mode leaves values and
observations alone. `--grace-days` sets the age threshold.

## Store schema versions

The SQLite metadata index records its schema generation in the database
(`PRAGMA user_version`). Fresh stores are created at the current version;
a store written at an older schema version is refused at open time with
`IncompatibleStoreError` — nothing is migrated implicitly.

### Upgrading a store with legacy parameter IDs

Close sessions using the store. Preview the identity and value changes, then
run the migration with a new backup directory:

```bash
python -m diffract.migrate /data/run/metadata_index.db --dry-run
python -m diffract.migrate /data/run/metadata_index.db --backup-dir /backups/run
```

For custom paths, pass `--config settings.ini` (or a JSON configuration).
The standard layout uses `metadata_index.db` alongside `sqlite_storage.db`,
`hdf5_storage.h5`, or `zarr_storage.zarr`. A hybrid layout pairs SQLite with
HDF5 or Zarr. A supplied configuration uses the same paths as a session, and
its configured index must match the command argument. A local Zarr store
accepts a filesystem path or a file URI. Keep writers closed from preflight
through verification.

The migration assigns instance and parameter UIDs, builds aggregate
membership, and records the old-to-new mapping in `uid_map`. Migrated fields
have format version zero. A legacy comparison with two model endpoints has
unknown direction. The upgrade keeps existing SQLite rows, HDF5 datasets and
remote Zarr objects at their physical addresses. Local Zarr nodes move to their
new keys. Fields written after the upgrade use the canonical owner UID and
encoded field name.

The preview reports conflicting rows and their provenance. Choose `--keep oldest` or `--keep newest` to select a duplicate by its original SQLite row
order. Only the selected row contributes its metadata and fields. Use
`--rename-map names.json` for names outside the supported alphabet. For
example, `{"parameters":{"weight#0":"weight.0"}}` renames one parameter across
models. A `models` mapping renames model names. Steps default to zero. With
`--parse-step-suffix`, `net_step1000` becomes model `net`, step 1000.

The backup contains SQLite snapshots, external array files and copies of
closed local HDF5 or Zarr stores. For a remote store, make a complete external
backup of the index and all value stores, then pass `--i-have-a-backup`. A
failure during preflight or backup leaves the input at its original version.
Retry with a fresh backup directory. If the command or session reports a
half-migrated store, close all connections, move the affected store files and
SQLite sidecars aside, and restore the complete backup to its original paths.
A backup created by the command records those paths in `restore.json`. Re-run
the migration after restoration.

`diffract.upgrade_metadata_index(path)` accepts a current index or initializes
an empty index carrying the Diffract application ID. The path comes from
`[metadata.sqlite] path` in the configuration. Opening an index newer than the
library requires a compatible library release. In-memory sessions create a
fresh index.

### Which database the path names

A Diffract metadata index carries a SQLite application ID. For an unmarked
file with tables, Diffract checks the shape of its metadata tables before
checking the schema version. A session raises `UnrecognizedStoreError` for
a foreign database. Check the configured `[metadata.sqlite] path` when this
happens.

A session initializes an empty, unclaimed file. Calling
`diffract.upgrade_metadata_index(path)` on that file raises `StoreIdentityError`.
An empty file that already carries the Diffract application ID can be
initialized through either entry point.
