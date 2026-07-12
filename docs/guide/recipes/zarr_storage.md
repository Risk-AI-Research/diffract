# Zarr storage (fsspec)

Diffract supports Zarr v3 as a storage backend for large arrays, with native support for
cloud object stores (S3, GCS, Azure, HDFS) via `fsspec`.

## Installation

The `zarr` extra bundles `s3fs`, so S3-compatible stores (AWS, MinIO, and other
S3-compatible endpoints) work without additional packages. GCS and Azure need their fsspec drivers installed
separately:

```bash
# Zarr support, including S3
uv sync --extra zarr

# For Google Cloud Storage
uv pip install gcsfs

# For Azure Blob Storage
uv pip install adlfs
```

## Storage URL schemes

`ZarrStorageManager` accepts `store_url` using standard fsspec URL schemes:

| Scheme | Example | Package |
|--------|---------|---------|
| Local | `/path/to/store` or `file:///path` | (built-in) |
| S3 | `s3://bucket/path` | `s3fs` |
| GCS | `gs://bucket/path` | `gcsfs` |
| Azure | `az://container/path` | `adlfs` |
| HDFS | `hdfs://namenode:8020/path` | `pyarrow` |

## Configuration options

### All parameters

```python
ZarrStorageManager(
    store_url: str,                    # Required: fsspec URL or local path
    storage_options: dict = None,      # fsspec options (credentials, endpoint, etc.)
    root: str = "root",                # Root group name within store
    readonly: bool = False,            # Read-only mode
    
    # Performance tuning
    compressor: str | None = "lz4",    # Compression: "lz4", "zstd", "zlib", None
    target_chunk_mb: float = 16.0,     # Target chunk size in MB (optimal: 8-32 for cloud)
    lazy_index_sync: bool = True,      # Defer index writes to close() for speed
    
    # Batching (inherited from BaseStorageManager)
    batch_size_limit_bytes: int = 50 * 1024 * 1024,
)
```

### INI configuration

The `ini` blocks on this page show only the storage-related sections. A complete session
config also requires `[metadata]`, `[cache]`, and `[nn.extractor]` sections — see the
complete minimal example in [Storage and Cache](storage_and_cache.md).

```ini
[storage]
backend = "zarr"

[storage.zarr]
store_url = "s3://my-bucket/diffract-data"
root = "root"
readonly = false
compressor = "lz4"
target_chunk_mb = 16.0
lazy_index_sync = true

# fsspec storage_options (for S3)
[storage.zarr.storage_options]
key = "${AWS_ACCESS_KEY_ID}"
secret = "${AWS_SECRET_ACCESS_KEY}"

[storage.zarr.storage_options.client_kwargs]
endpoint_url = "${S3_ENDPOINT_URL}"
region_name = "${AWS_DEFAULT_REGION}"
```

## Cloud provider examples

### AWS S3

```ini
[storage.zarr]
store_url = "s3://my-bucket/diffract/data"

[storage.zarr.storage_options]
# Uses default AWS credentials chain (env vars, ~/.aws/credentials, IAM role)
```

Environment variables:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

### Google Cloud Storage

```ini
[storage.zarr]
store_url = "gs://my-bucket/diffract/data"

[storage.zarr.storage_options]
token = "/path/to/service-account.json"
# Or use default credentials: token = "google_default"
```

### Azure Blob Storage

```ini
[storage.zarr]
store_url = "az://container/diffract/data"

[storage.zarr.storage_options]
account_name = "mystorageaccount"
account_key = "${AZURE_STORAGE_KEY}"
```

### HDFS

```ini
[storage.zarr]
store_url = "hdfs://namenode:8020/user/diffract/data"

[storage.zarr.storage_options]
host = "namenode"
port = 8020
user = "hadoop"
```

### MinIO (S3-compatible)

```ini
[storage.zarr]
store_url = "s3://my-bucket/diffract/data"

[storage.zarr.storage_options]
key = "minioadmin"
secret = "minioadmin"

[storage.zarr.storage_options.client_kwargs]
endpoint_url = "http://localhost:9000"
```

## Hybrid storage (recommended for production)

Combine fast local metadata with cloud array storage. As elsewhere on this page, the
block below shows the `[storage]` portion; the remaining required sections are covered in
[Storage and Cache](storage_and_cache.md).

```ini
[storage]
backend = "hybrid"

[storage.hybrid]
# Fast local access for metadata
light = "sqlite"
# Cloud storage for large arrays
heavy = "zarr"
# 128 MB threshold
array_threshold = 134217728

[storage.sqlite]
path = "data/metadata.db"

[storage.zarr]
store_url = "s3://bucket/arrays"
compressor = "lz4"
target_chunk_mb = 16.0

[storage.zarr.storage_options]
key = "${AWS_ACCESS_KEY_ID}"
secret = "${AWS_SECRET_ACCESS_KEY}"
```

**Benefits:**

- Metadata queries (list, filter) are fast (local SQLite)
- Large arrays stored in cloud (scalable, shareable)
- Automatic routing based on data size

## Performance tuning

### Compression

| Compressor | Speed | Ratio | Use Case |
|------------|-------|-------|----------|
| `lz4` | ~3 GB/s | ~2x | **Default**, best for speed |
| `zstd` | ~1 GB/s | ~3x | Better compression |
| `zlib` | ~0.3 GB/s | ~3x | Maximum compatibility |
| `None` | - | 1x | Already compressed data |

### Chunk size

```python
# For cloud storage (S3, GCS, Azure):
target_chunk_mb = 16.0  # Optimal: 8-32 MB per chunk

# For local storage:
target_chunk_mb = 4.0   # Smaller chunks OK
```

**Why 16 MB?**

- S3 multipart upload minimum: 5 MB
- HTTP overhead amortization: larger = fewer requests
- Memory usage: not too large for streaming

### Lazy index sync

```ini
[storage.zarr]
lazy_index_sync = true
```

- **true** (default): Faster — the index is written only on `close()` — but lost on crash
- **false**: Safer (immediate persistence), slower

## Python API

```python
from diffract.core.storage import ZarrStorageManager
import numpy as np

# Direct usage
storage = ZarrStorageManager(
    store_url="s3://bucket/data",
    storage_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {"endpoint_url": "https://..."}
    },
    compressor="lz4",
    target_chunk_mb=16.0,
)
storage.connect()

# Store data
storage.set_field("model_001", "weights", np.random.randn(1000, 500).astype(np.float32))
storage.set_field("model_001", "metadata", {"layers": 12, "params": "500K"})

# Retrieve
weights = storage.get_field("model_001", "weights")
meta = storage.get_field("model_001", "metadata")

# Batch operations (faster)
with storage:
    for i in range(100):
        storage.set_field(f"param_{i}", "weights", large_array)

storage.close()
```

## Session integration

```python
from diffract import Session

# Via config file
session = Session(config_path="configs/hybrid_s3.ini")

# Or programmatically
from diffract.containers import create_main_container
container = create_main_container("configs/hybrid_s3.ini")
session = Session(container=container)
```

## Testing

```bash
# Unit tests (local Zarr only)
make test-light

# Integration tests with S3 (requires .env)
pytest tests/integration/test_hybrid_zarr_s3.py -v

# Full test suite
make test
```

### Environment for S3 tests

Create `.env` in project root:

```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_ENDPOINT_URL=https://s3.example.com
S3_BUCKET=your-bucket
S3_PREFIX=diffract/
```

Tests skip automatically if credentials are missing.

## Troubleshooting

### "FsspecStore.__init__() got an unexpected keyword argument"

Ensure you're using Zarr v3:

```python
import zarr
print(zarr.__version__)  # Should be >= 3.0.0
```

### SSL errors on close

Harmless warnings from async S3 connection cleanup. Can be ignored.

### "botocore.exceptions.NoCredentialsError"

Set AWS credentials via environment variables or `~/.aws/credentials`.

### Slow performance

1. Enable compression: `compressor = "lz4"`
2. Increase chunk size: `target_chunk_mb = 32.0`
3. Use hybrid storage with local SQLite for metadata
