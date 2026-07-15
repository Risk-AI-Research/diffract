"""Integration tests: Hybrid storage (SQLite light + Zarr S3 heavy).

Requires environment variables for an S3-compatible store:
    AWS_ACCESS_KEY_ID - Access key ID
    AWS_SECRET_ACCESS_KEY - Secret access key
    AWS_DEFAULT_REGION - Region (default: us-east-1)
    S3_ENDPOINT_URL - Endpoint URL (optional; defaults to AWS)
    S3_BUCKET - Bucket name (required)
    S3_PREFIX - Prefix within bucket (default: diffract/)

Skip behavior:
    - Tests skip if S3_BUCKET is not set
    - Tests skip if s3fs/zarr packages are not installed
    - Tests skip on connectivity errors

Reference: https://cloud.vk.com/docs/storage/s3/quick-start
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _get_s3_config() -> dict[str, str] | None:
    """Return S3 configuration from environment or None if not configured."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None
    return {
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
        "bucket": bucket,
        "prefix": os.getenv("S3_PREFIX", "diffract/"),
    }


def _check_s3fs_available() -> bool:
    """Check if s3fs and zarr are available."""
    try:
        import s3fs  # noqa: F401
        import zarr  # noqa: F401
    except ImportError:
        return False
    else:
        return True


S3_CONFIG = _get_s3_config()
S3FS_AVAILABLE = _check_s3fs_available()
SKIP_REASON_NO_CONFIG = "S3_BUCKET environment variable not set"
SKIP_REASON_NO_S3FS = "s3fs or zarr package not installed"


@pytest.fixture
def zarr_s3_storage_manager() -> Generator:
    """Fixture for Zarr storage manager using S3 backend."""
    if not S3_CONFIG:
        pytest.skip(SKIP_REASON_NO_CONFIG)
    if not S3FS_AVAILABLE:
        pytest.skip(SKIP_REASON_NO_S3FS)

    from diffract.core.storage.zarr_manager import ZarrStorageManager

    run_id = uuid.uuid4().hex[:12]
    store_url = f"s3://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}test_{run_id}"
    storage_options = {
        "key": S3_CONFIG["aws_access_key_id"],
        "secret": S3_CONFIG["aws_secret_access_key"],
        "client_kwargs": {
            "endpoint_url": S3_CONFIG["endpoint_url"],
            "region_name": S3_CONFIG["region"],
        },
    }

    try:
        storage = ZarrStorageManager(
            store_url=store_url,
            storage_options=storage_options,
            root="root",
            readonly=False,
        )
        storage.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"S3 connection failed: {exc}")

    try:
        yield storage
    finally:
        with contextlib.suppress(Exception):
            storage.clear()
        storage.close()

        with contextlib.suppress(Exception):
            import s3fs

            fs = s3fs.S3FileSystem(
                key=S3_CONFIG["aws_access_key_id"],
                secret=S3_CONFIG["aws_secret_access_key"],
                client_kwargs={
                    "endpoint_url": S3_CONFIG["endpoint_url"],
                    "region_name": S3_CONFIG["region"],
                },
            )
            if fs.exists(store_url):
                fs.rm(store_url, recursive=True)


@pytest.fixture
def hybrid_zarr_s3_storage_manager(temp_dir: Path) -> Generator:
    """Fixture for Hybrid storage manager (SQLite light + Zarr S3 heavy)."""
    if not S3_CONFIG:
        pytest.skip(SKIP_REASON_NO_CONFIG)
    if not S3FS_AVAILABLE:
        pytest.skip(SKIP_REASON_NO_S3FS)

    from diffract.core.storage.hybrid_manager import HybridStorageManager
    from diffract.core.storage.sqlite_manager import SQLiteStorageManager
    from diffract.core.storage.zarr_manager import ZarrStorageManager

    sqlite_path = str(temp_dir / "hybrid_meta.db")
    sqlite_mgr = SQLiteStorageManager(path=sqlite_path)
    sqlite_mgr.connect()

    run_id = uuid.uuid4().hex[:12]
    store_url = f"s3://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}hybrid_{run_id}"
    storage_options = {
        "key": S3_CONFIG["aws_access_key_id"],
        "secret": S3_CONFIG["aws_secret_access_key"],
        "client_kwargs": {
            "endpoint_url": S3_CONFIG["endpoint_url"],
            "region_name": S3_CONFIG["region"],
        },
    }

    try:
        zarr_mgr = ZarrStorageManager(
            store_url=store_url,
            storage_options=storage_options,
            root="root",
            readonly=False,
        )
        zarr_mgr.connect()
    except Exception as exc:  # noqa: BLE001
        sqlite_mgr.close()
        pytest.skip(f"S3 connection failed: {exc}")

    hybrid = HybridStorageManager(
        light_storage=sqlite_mgr,
        heavy_storage=zarr_mgr,
        array_threshold=1024 * 1024,  # 1MB threshold
    )

    try:
        yield hybrid
    finally:
        with contextlib.suppress(Exception):
            hybrid.clear()
        hybrid.close()

        with contextlib.suppress(Exception):
            import s3fs

            fs = s3fs.S3FileSystem(
                key=S3_CONFIG["aws_access_key_id"],
                secret=S3_CONFIG["aws_secret_access_key"],
                client_kwargs={
                    "endpoint_url": S3_CONFIG["endpoint_url"],
                    "region_name": S3_CONFIG["region"],
                },
            )
            if fs.exists(store_url):
                fs.rm(store_url, recursive=True)


class TestZarrS3Basic:
    """Basic Zarr S3 storage tests."""

    def test_json_roundtrip(self, zarr_s3_storage_manager) -> None:
        """Test JSON data roundtrip to S3."""
        storage = zarr_s3_storage_manager
        uid = "s3_json_test"
        data = {"model": "test", "config": {"layers": [1, 2, 3]}}

        storage.set_field(uid, "metadata", data)
        assert storage.has_field(uid, "metadata")
        assert storage.get_field(uid, "metadata") == data

    def test_ndarray_roundtrip(self, zarr_s3_storage_manager) -> None:
        """Test ndarray roundtrip to S3."""
        storage = zarr_s3_storage_manager
        uid = "s3_array_test"
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((100, 50)).astype(np.float32)

        storage.set_field(uid, "weights", arr)
        got = storage.get_field(uid, "weights")

        assert isinstance(got, np.ndarray)
        assert got.shape == arr.shape
        assert got.dtype == arr.dtype
        np.testing.assert_allclose(got, arr)

    def test_list_operations(self, zarr_s3_storage_manager) -> None:
        """Test list operations on S3."""
        storage = zarr_s3_storage_manager

        storage.set_field("a", "f1", 1)
        storage.set_field("b", "f1", 2)
        storage.set_field("b", "f2", 3)

        assert set(storage.list_objs()) >= {"a", "b"}
        assert set(storage.list_fields()) >= {"f1", "f2"}
        assert set(storage.list_fields("b")) >= {"f1", "f2"}
        assert set(storage.list_objs_has_field("f1")) >= {"a", "b"}

    def test_erase_operations(self, zarr_s3_storage_manager) -> None:
        """Test erase operations on S3."""
        storage = zarr_s3_storage_manager

        storage.set_field("x", "f", 1)
        assert storage.has_field("x", "f")

        storage.erase_field("x", "f")
        assert not storage.has_field("x", "f")

        storage.set_field("y", "a", 1)
        storage.set_field("y", "b", 2)
        storage.erase_obj("y")
        assert not storage.has_field("y", "a")
        assert not storage.has_field("y", "b")


class TestHybridZarrS3:
    """Hybrid storage (SQLite + Zarr S3) tests."""

    def test_routing_small_to_sqlite(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test small data routes to SQLite (light storage)."""
        storage = hybrid_zarr_s3_storage_manager
        small_data = {"name": "layer_0", "type": "dense"}

        storage.set_field("small", "meta", small_data)
        assert storage.has_field("small", "meta")
        assert storage.get_field("small", "meta") == small_data

    def test_routing_large_to_zarr_s3(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test large array routes to Zarr S3 (heavy storage)."""
        storage = hybrid_zarr_s3_storage_manager
        from .helpers import create_large_array

        large_array = create_large_array(2, dtype=np.float32)  # 2MB

        storage.set_field("large", "weights", large_array)
        assert storage.has_field("large", "weights")

        got = storage.get_field("large", "weights")
        assert isinstance(got, np.ndarray)
        assert got.shape == large_array.shape
        np.testing.assert_allclose(got, large_array)

    def test_mixed_data_object(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test object with both small metadata and large arrays."""
        storage = hybrid_zarr_s3_storage_manager
        from .helpers import create_large_array

        uid = "mixed_obj"
        meta = {"model": "test", "layer": 0}
        weights = create_large_array(2, dtype=np.float32)

        storage.set_field(uid, "metadata", meta)
        storage.set_field(uid, "weights", weights)

        assert storage.has_field(uid, "metadata")
        assert storage.has_field(uid, "weights")

        fields = storage.list_fields(uid)
        assert "metadata" in fields
        assert "weights" in fields

        assert storage.get_field(uid, "metadata") == meta
        got_weights = storage.get_field(uid, "weights")
        np.testing.assert_allclose(got_weights, weights)

    def test_batch_operations(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test batch context with hybrid storage."""
        storage = hybrid_zarr_s3_storage_manager
        from .helpers import create_large_array

        with storage:
            for i in range(5):
                storage.set_field(f"obj_{i}", "meta", {"idx": i})
                if i % 2 == 0:
                    arr = create_large_array(1, dtype=np.float32)
                    storage.set_field(f"obj_{i}", "array", arr)

        for i in range(5):
            assert storage.has_field(f"obj_{i}", "meta")
            assert storage.get_field(f"obj_{i}", "meta") == {"idx": i}
            if i % 2 == 0:
                assert storage.has_field(f"obj_{i}", "array")

    def test_erase_operations(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test erase operations across backends."""
        storage = hybrid_zarr_s3_storage_manager
        from .helpers import create_large_array

        uid = "erase_test"
        storage.set_field(uid, "meta", {"test": True})
        storage.set_field(uid, "weights", create_large_array(1, dtype=np.float32))

        storage.erase_field(uid, "meta")
        assert not storage.has_field(uid, "meta")
        assert storage.has_field(uid, "weights")

        storage.erase_obj(uid)
        assert not storage.has_field(uid, "weights")

    def test_clear_all(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test clear operation on hybrid storage."""
        storage = hybrid_zarr_s3_storage_manager
        from .helpers import create_large_array

        storage.set_field("c1", "meta", {"a": 1})
        storage.set_field("c2", "weights", create_large_array(1, dtype=np.float32))

        assert len(storage.list_objs()) >= 2

        storage.clear()
        assert storage.list_objs() == []
        assert storage.list_fields() == []

    def test_metadata_retrieval(self, hybrid_zarr_s3_storage_manager) -> None:
        """Test metadata retrieval for large arrays stored in Zarr S3."""
        storage = hybrid_zarr_s3_storage_manager
        arr = np.ones((100, 200), dtype=np.float64)

        storage.set_field("meta_test", "weights", arr)

        meta = storage.get_field_metadata("meta_test", "weights")
        assert meta is not None
        assert tuple(meta.get("shape", [])) == (100, 200)
        assert meta.get("dtype") == "float64"
