"""Storage management module providing persistent data storage backends.

This module offers a unified interface for persistent storage with support for
multiple backends including HDF5. It uses dependency injection for flexible
configuration and backend selection, providing durability guarantees for
cached computations.

Features:
    - HDF5-based persistent storage with compression
    - Hierarchical data organization by objects and fields
    - Support for NumPy arrays, JSON-serializable data, and arbitrary objects
    - Atomic write operations and data integrity
    - Configurable compression and chunking strategies

Example:
    >>> from diffract.core.storage import StorageContainer
    >>> container = StorageContainer()
    >>> storage_manager = container.storage_manager()
    >>> storage_manager.set_field("obj123", "result", numpy_array)
"""

from .containers import StorageContainer
from .hdf5_manager import HDF5StorageManager
from .interface import DEFAULT_TABLE, UID, IStorageManager
from .ram_manager import RAMStorageManager
from .zarr_manager import ZarrStorageManager

__all__ = [
    "DEFAULT_TABLE",
    "UID",
    "HDF5StorageManager",
    "IStorageManager",
    "StorageContainer",
    "ZarrStorageManager",
]
