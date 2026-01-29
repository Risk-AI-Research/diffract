# Configuration

Diffract uses a layered configuration system: **profiles** for quick setup, **config files** for full control.

## Profiles

Profiles are built-in configurations for common scenarios.

```python
from diffract import Session, list_profiles

print(list_profiles())  # ['ram', 'local', 'hybrid']

session = Session(profile="ram")
session = Session(profile="local")
session = Session(profile="hybrid")
```

### Available profiles

| Profile | Storage | Cache | Use case |
|---------|---------|-------|----------|
| `ram` | RAM | none | Quick experiments, no persistence |
| `local` | SQLite | Redis | Local development with persistence |
| `hybrid` | SQLite + HDF5 | Redis | Large models, optimized array storage |

**Notes:**

- **Redis is optional**: `local` and `hybrid` default to Redis cache. Use a config file with `cache.backend = "simple"` if you don't want Redis.
- **Relative paths** are resolved against your current working directory.

For backend details and recommended setups, see [Storage and Cache](../guide/recipes/storage_and_cache.md).

## Config files

Use config files when you need reproducibility or custom backends. Diffract supports INI, YAML, and JSON.

```python
session = Session(config_path="my_config.ini")
```

If both `profile` and `config_path` are provided and the file exists, the config file wins.

### INI syntax

INI files use dot-separated section names for nesting:

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 256
```

Values are coerced from strings:

- `none`/`null` → `None`
- `true`/`false` → booleans
- JSON literals → parsed as JSON
- numbers → `int`/`float` when possible

### Minimal example (SQLite + simple cache)

```ini
[storage]
backend = "sqlite"

[storage.sqlite]
path = "data/diffract.db"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 256

[compute.executor]
max_workers = 4
chunk_size = 32
```

### Redis cache

```ini
[cache]
backend = "redis"

[cache.redis]
host = "localhost"
port = 6379
db = 0
max_memory_mb = 4096
key_prefix = "diffract:cache:"
```

### Hybrid storage

```ini
[storage]
backend = "hybrid"

[storage.sqlite]
path = "data/metadata.db"

[storage.hdf5]
path = "data/arrays.h5"

[cache]
backend = "simple"

[cache.simple]
max_memory_mb = 512
```

## Example configs

The repository includes ready-to-use configs in `configs/`:

- `fast_speed_without_disk.ini` — RAM storage, no cache
- `sqlite.ini` — SQLite + Redis
- `hybrid.ini` — Hybrid + Redis

## Priority order

Configuration is resolved in this order:

1. **`config_path`** — Explicit config file (if exists)
2. **`profile`** — Built-in profile (defaults to `"ram"`)
3. **Internal defaults** — fallback

```python
# Config file wins over profile
session = Session(config_path="prod.ini", profile="local")
```
