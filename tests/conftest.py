"""Global test configuration and fixtures."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import rootutils

from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex

_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

# Redis test configuration
REDIS_HOST = os.getenv("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("TEST_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("TEST_REDIS_DB", "15"))


class InMemoryStorage:
    """Simple in-memory key-value store for storage/cache manager test doubles."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, object]] = {}

    def __enter__(self) -> "InMemoryStorage":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def set_field(
        self, obj_uid: str, field_name: str, value: object, *, table: str = "default"
    ) -> None:
        self._store.setdefault(obj_uid, {})[field_name] = value

    def get_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> object:
        return self._store[obj_uid][field_name]

    def has_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> bool:
        return obj_uid in self._store and field_name in self._store[obj_uid]

    def erase_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> None:
        if obj_uid in self._store and field_name in self._store[obj_uid]:
            del self._store[obj_uid][field_name]

    def erase_obj(self, obj_uid: str, *, table: str = "default") -> None:
        self._store.pop(obj_uid, None)

    def list_fields(
        self, obj_uid: str | None = None, *, table: str = "default"
    ) -> list[str]:
        if obj_uid is None:
            all_fields: set[str] = set()
            for fields in self._store.values():
                all_fields.update(fields.keys())
            return list(all_fields)
        return list(self._store.get(obj_uid, {}).keys())

    def list_objs(self, *, table: str = "default") -> list[str]:
        return list(self._store.keys())

    def list_objs_has_field(
        self, field_name: str, *, table: str = "default"
    ) -> list[str]:
        return [uid for uid, fields in self._store.items() if field_name in fields]

    def get_field_metadata(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> dict[str, Any] | None:
        return None


class InMemoryCache(InMemoryStorage):
    def get_available_bytes(self) -> int:
        return 256 * 1024 * 1024  # 256MB for tests

    def list_uids(self, *, table: str = "default") -> list[str]:
        """List all UIDs (object keys) in the cache."""
        return list(self._store.keys())

    def upsert(
        self, obj_uid: str, field_name: str, value: object, *, table: str = "default"
    ) -> None:
        """Insert or update a field value."""
        self._store.setdefault(obj_uid, {})[field_name] = value


@pytest.fixture
def storage_cache() -> tuple[InMemoryStorage, InMemoryCache]:
    return InMemoryStorage(), InMemoryCache()


@pytest.fixture
def storage_cache_metadata() -> Generator[
    tuple[InMemoryStorage, InMemoryCache, SQLiteMetadataIndex]
]:
    index = SQLiteMetadataIndex(":memory:")
    try:
        yield InMemoryStorage(), InMemoryCache(), index
    finally:
        index.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Create a temporary configuration file."""
    config_file = temp_dir / "test_config.yaml"
    config_content = """
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    standard:
      format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  handlers:
    console:
      class: logging.StreamHandler
      level: INFO
      formatter: standard
      stream: ext://sys.stdout
  root:
    level: INFO
    handlers: [console]

storage:
  backend: hdf5
  hdf5:
    root: test_root
    compression: lzf
    readonly: false

metadata:
  backend: sqlite
  sqlite:
    path: ':memory:'

cache:
  backend: simple
  simple:
    max_memory_mb: 64
    ttl_seconds: 3600
    key_prefix: 'test:cache:'
  redis:
    host: localhost
    port: 6379
    db: 1
    max_memory_mb: 64
    ttl_seconds: 3600
    key_prefix: 'test:redis:cache:'

compute:
  max_workers: 2
  chunk_size: 16
  minimal_chunk_size: 1

nn:
  extractor:
    skip_not_implemented_types: true
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def temp_ini_config_file(temp_dir: Path) -> Path:
    """Create a temporary INI configuration file for container initialization."""
    config_file = temp_dir / "test_config.ini"
    config_content = f"""
[storage]
backend = hdf5

[storage.hdf5]
path = {temp_dir / "test_storage.h5"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = :memory:

[cache]
backend = simple

[cache.simple]
max_memory_mb = 64
ttl_seconds = 3600
key_prefix = "test:cache:"

[compute.executor]
max_workers = 2
chunk_size = 16
minimal_chunk_size = 1

[nn.extractor]
skip_not_implemented_types = true
"""
    config_file.write_text(config_content.strip() + "\n")
    return config_file


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "storage": {
            "backend": "hdf5",
            "hdf5": {
                "root": "test_root",
                "compression": "lzf",
                "readonly": False,
            },
        },
        "cache": {
            "backend": "simple",
            "simple": {
                "max_memory_mb": 64,
                "ttl_seconds": 3600,
                "key_prefix": "test:cache:",
            },
        },
        "compute": {
            "max_workers": 2,
            "chunk_size": 16,
            "minimal_chunk_size": 1,
        },
        "nn": {
            "extractor": {
                "skip_not_implemented_types": True,
            },
        },
    }


# Integration test fixtures for storage and cache managers


@pytest.fixture
def redis_cache_manager() -> Generator:
    """Fixture for Redis cache manager with cleanup."""
    from diffract.core.cache.redis_manager import RedisLRUCacheManager

    try:
        cache = RedisLRUCacheManager(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            max_memory_mb=128,
            ttl_seconds=None,
            key_prefix="diffract:test:integration:",
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Redis not available: {e}")

    cache.clear()
    try:
        yield cache
    finally:
        cache.clear()
        cache.close()


@pytest.fixture
def sqlite_storage_manager(temp_dir: Path) -> Generator:
    """Fixture for SQLite storage manager."""
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager

    db_path = str(temp_dir / "test_storage.db")
    storage = SQLiteStorageManager(path=db_path)
    storage.connect()
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def hdf5_storage_manager(temp_dir: Path) -> Generator:
    """Fixture for HDF5 storage manager."""
    from diffract.core.storage.hdf5_manager import HDF5StorageManager

    h5_path = str(temp_dir / "test_storage.h5")
    storage = HDF5StorageManager(path=h5_path, swmr=True, keep_file_open=False)
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def hybrid_storage_manager(temp_dir: Path) -> Generator:
    """Fixture for Hybrid storage manager (SQLite + HDF5)."""
    from diffract.core.storage.hdf5_manager import HDF5StorageManager
    from diffract.core.storage.hybrid_manager import HybridStorageManager
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager

    sqlite_path = str(temp_dir / "test_meta.db")
    hdf5_path = str(temp_dir / "test_arrays.h5")

    sqlite_mgr = SQLiteStorageManager(path=sqlite_path)
    sqlite_mgr.connect()

    hdf5_mgr = HDF5StorageManager(path=hdf5_path, swmr=True, keep_file_open=False)

    hybrid = HybridStorageManager(
        sqlite_mgr,
        hdf5_mgr,
        array_threshold=1024 * 1024,  # 1MB threshold for testing
    )

    try:
        yield hybrid
    finally:
        hybrid.close()


@pytest.fixture
def session_with_redis_sqlite(temp_dir: Path) -> Generator:
    """Fixture for Session with Redis cache and SQLite storage."""
    from diffract.containers import WiringConfiguration, create_main_container
    from diffract.session import Session

    key_prefix = "diffract:test:session:redis_sqlite:"

    # Create config file
    config_path = temp_dir / "config_redis_sqlite.ini"
    config_content = f"""
[storage]
backend = sqlite

[storage.sqlite]
path = {temp_dir / "session_storage.db"}

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / "session_metadata.db"}

[cache]
backend = redis

[cache.redis]
host = {REDIS_HOST}
port = {REDIS_PORT}
db = {REDIS_DB}
max_memory_mb = 128
ttl_seconds = 3600
key_prefix = {key_prefix}

[compute.executor]
max_workers = 4
chunk_size = 16

[nn.extractor]
skip_not_implemented_types = true
"""
    config_path.write_text(config_content.strip() + "\n")

    # Clear Redis keys with the prefix before test
    try:
        import redis

        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(
                cursor=cursor, match=f"{key_prefix}*", count=1000
            )
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
        redis_client.close()
    except Exception:  # noqa: BLE001
        pass  # Redis not available, will be caught later

    try:
        container = create_main_container(config_path)
        WiringConfiguration.wire(container)
        session = Session(container=container)
        yield session
    except Exception as e:
        if "Redis" in str(e) or "redis" in str(e).lower():
            pytest.skip(f"Redis not available: {e}")
        raise
    finally:
        # Clean up Redis keys after test
        try:
            import redis

            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor=cursor, match=f"{key_prefix}*", count=1000
                )
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            redis_client.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def session_with_redis_hdf5(temp_dir: Path) -> Generator:
    """Fixture for Session with Redis cache and HDF5 storage."""
    from diffract.containers import WiringConfiguration, create_main_container
    from diffract.session import Session

    key_prefix = "diffract:test:session:redis_hdf5:"

    config_path = temp_dir / "config_redis_hdf5.ini"
    config_content = f"""
[storage]
backend = hdf5

[storage.hdf5]
path = {temp_dir / "session_storage.h5"}
swmr = true
keep_file_open = false

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / "session_metadata.db"}

[cache]
backend = redis

[cache.redis]
host = {REDIS_HOST}
port = {REDIS_PORT}
db = {REDIS_DB}
max_memory_mb = 128
ttl_seconds = 3600
key_prefix = {key_prefix}

[compute.executor]
max_workers = 4
chunk_size = 16

[nn.extractor]
skip_not_implemented_types = true
"""
    config_path.write_text(config_content.strip() + "\n")

    # Clear Redis keys with the prefix before test
    try:
        import redis

        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(
                cursor=cursor, match=f"{key_prefix}*", count=1000
            )
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
        redis_client.close()
    except Exception:  # noqa: BLE001
        pass  # Redis not available, will be caught later

    try:
        container = create_main_container(config_path)
        WiringConfiguration.wire(container)
        session = Session(container=container)
        yield session
    except Exception as e:
        if "Redis" in str(e) or "redis" in str(e).lower():
            pytest.skip(f"Redis not available: {e}")
        raise
    finally:
        # Clean up Redis keys after test
        try:
            import redis

            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor=cursor, match=f"{key_prefix}*", count=1000
                )
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            redis_client.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def session_with_redis_hybrid(temp_dir: Path) -> Generator:
    """Fixture for Session with Redis cache and Hybrid storage."""
    from diffract.containers import WiringConfiguration, create_main_container
    from diffract.session import Session

    key_prefix = "diffract:test:session:redis_hybrid:"

    config_path = temp_dir / "config_redis_hybrid.ini"
    config_content = f"""
[storage]
backend = hybrid

[storage.hybrid]
light = sqlite
heavy = hdf5
array_threshold = 1048576

[storage.sqlite]
path = {temp_dir / "session_meta.db"}

[storage.hdf5]
path = {temp_dir / "session_arrays.h5"}
swmr = true
keep_file_open = false

[metadata]
backend = sqlite

[metadata.sqlite]
path = {temp_dir / "session_metadata.db"}

[cache]
backend = redis

[cache.redis]
host = {REDIS_HOST}
port = {REDIS_PORT}
db = {REDIS_DB}
max_memory_mb = 128
ttl_seconds = 3600
key_prefix = {key_prefix}

[compute.executor]
max_workers = 4
chunk_size = 16

[nn.extractor]
skip_not_implemented_types = true
"""
    config_path.write_text(config_content.strip() + "\n")

    # Clear Redis keys with the prefix before test
    try:
        import redis

        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(
                cursor=cursor, match=f"{key_prefix}*", count=1000
            )
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
        redis_client.close()
    except Exception:  # noqa: BLE001
        pass  # Redis not available, will be caught later

    try:
        container = create_main_container(config_path)
        WiringConfiguration.wire(container)
        session = Session(container=container)
        yield session
    except Exception as e:
        if "Redis" in str(e) or "redis" in str(e).lower():
            pytest.skip(f"Redis not available: {e}")
        raise
    finally:
        # Clean up Redis keys after test
        try:
            import redis

            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor=cursor, match=f"{key_prefix}*", count=1000
                )
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            redis_client.close()
        except Exception:  # noqa: BLE001
            pass
