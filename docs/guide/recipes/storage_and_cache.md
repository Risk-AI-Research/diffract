# Storage and Cache

Configure storage and cache backends for different workloads.

For profile selection, see [Configuration](../../reference/configuration.md).

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

SQLite for metadata and small values, HDF5 for large arrays. Best of both worlds.

```ini
[storage]
backend = "hybrid"

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

[storage.sqlite]
path = ".diffract/metadata.db"

[storage.zarr]
store_url = "s3://my-bucket/diffract-data"
```

**Use case:** Local metadata + remote arrays (“cloud scale-out”) while keeping the same API.

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

- **SQLite:** Uses connection pooling for reads, serialized writes. Safe for typical local workloads.
- **HDF5:** Serialized writes, concurrent reads. Avoid multiple independent writers.
- **Redis:** Configured for RAM-only LRU eviction. Data is not persisted.
