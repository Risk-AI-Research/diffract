"""Generic data repository for managing storage, metadata index, and cache.

This module provides the base DataRepository class that can be specialized
for different data types (parameters, relations, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

from diffract.core.parallel import ParallelContext
from diffract.core.storage.interface import DEFAULT_TABLE

from .view import DataView

if TYPE_CHECKING:
    import types
    from collections.abc import Iterable, Iterator

    from diffract.core.cache.interface import ICacheManager
    from diffract.core.data.metadata.interface import IMetadataIndex
    from diffract.core.storage.interface import IStorageManager

TMetadata = TypeVar("TMetadata")
TProxy = TypeVar("TProxy")

logger = logging.getLogger(__name__)


class DataRepository[TMetadata, TProxy]:
    """Generic repository owning storage/cache/metadata and managing entity membership.

    Provides the infrastructure for persistent storage, metadata indexing, and
    caching of metadata-bearing entities. Subclasses specify the metadata type,
    proxy class, view class, storage table, and metadata schema.

    Type Parameters:
        TMetadata: The metadata type for entities in this repository.
        TProxy: The proxy type for entities in this repository.
    """

    # Override in subclasses
    METADATA_CLASS: type[TMetadata]
    PROXY_CLASS: type[TProxy]
    VIEW_CLASS: type[DataView[TMetadata, TProxy]]
    TABLE: str = DEFAULT_TABLE

    # Schema for MetadataIndex - override in subclasses
    METADATA_COLUMNS: ClassVar[dict[str, type]] = {}
    METADATA_INDEXES: ClassVar[list[str]] = []

    def __init__(
        self,
        storage_manager: IStorageManager,
        metadata_index: IMetadataIndex,
        cache_manager: ICacheManager | None = None,
    ) -> None:
        self._storage_manager = storage_manager
        self._metadata_index = metadata_index
        self._cache_manager = cache_manager
        self._proxy_cache: dict[str, TProxy] = {}

    @property
    def storage_manager(self) -> IStorageManager:
        """Return the storage manager owned by this repository."""
        return self._storage_manager

    @property
    def metadata_index(self) -> IMetadataIndex:
        """Return the metadata index owned by this repository."""
        return self._metadata_index

    @property
    def cache_manager(self) -> ICacheManager | None:
        """Return the cache manager owned by this repository."""
        return self._cache_manager

    def __enter__(self) -> Self:
        """Enter batch write context for repository storage."""
        self._storage_manager.__enter__()
        self._metadata_index.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit batch write context for repository storage."""
        self._metadata_index.__exit__(exc_type, exc_val, exc_tb)
        self._storage_manager.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self) -> int:
        """Return number of entities in the repository."""
        return self._metadata_index.count(self.TABLE)

    def __iter__(self) -> Iterator[TProxy]:
        """Iterate entity proxies in repository order."""
        for uid in self._metadata_index.list_uids(self.TABLE):
            yield self.get_proxy(uid)

    def __getitem__(self, index: int) -> TProxy:
        """Return entity proxy by positional index."""
        uids = self._metadata_index.list_uids(self.TABLE)
        return self.get_proxy(uids[index])

    def get_proxy(self, uid: str) -> TProxy:
        """Return entity proxy by uid with lazy loading."""
        if uid in self._proxy_cache:
            return self._proxy_cache[uid]

        meta_dict = self._metadata_index.get(self.TABLE, uid)
        if meta_dict is None:
            raise KeyError(f"Entity with uid '{uid}' not found in {self.TABLE}")

        meta = self.METADATA_CLASS.from_dict(meta_dict)
        proxy = self.PROXY_CLASS(meta=meta, _repository=self)
        self._proxy_cache[uid] = proxy
        return proxy

    def has_uid(self, uid: str) -> bool:
        """Check if entity with given uid exists."""
        return self._metadata_index.get(self.TABLE, uid) is not None

    def append(self, proxy: TProxy) -> None:
        """Add an entity proxy to repository (cache only, metadata already in index)."""
        self._proxy_cache[proxy.meta.uid] = proxy

    def extend(self, proxies: Iterable[TProxy]) -> None:
        """Add multiple entity proxies to repository cache."""
        for proxy in proxies:
            self.append(proxy)

    def remove_by_uid(self, uid: str) -> None:
        """Remove an entity from repository by uid."""
        self._proxy_cache.pop(uid, None)
        self._metadata_index.delete(self.TABLE, uid)
        self._storage_manager.erase_obj(uid, table=self.TABLE)

    def clear(self, erase: bool = False) -> None:
        """Clear repository membership and optionally erase storage data.

        Args:
            erase: If True, also erase underlying storage data and cache.
                   If False, only clear membership (metadata index and proxy cache).
        """
        if erase:
            if self._cache_manager is not None:
                self._cache_manager.clear()

            uids = self._metadata_index.list_uids(self.TABLE)
            with self._storage_manager:
                for uid in uids:
                    self._storage_manager.erase_obj(uid, table=self.TABLE)

        self._metadata_index.clear(self.TABLE)
        self._proxy_cache.clear()

    def query(
        self,
        where: dict[str, Any] | None = None,
        where_in: dict[str, list[Any]] | None = None,
        where_like: dict[str, str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Query UIDs from metadata index."""
        return self._metadata_index.query(
            self.TABLE,
            where=where,
            where_in=where_in,
            where_like=where_like,
            order_by=order_by,
            limit=limit,
        )

    def list_uids(self) -> list[str]:
        """List all entity UIDs in the repository."""
        return self._metadata_index.list_uids(self.TABLE)

    def prefetch_proxies(
        self,
        uids: list[str],
        parallel: ParallelContext | None = None,
    ) -> None:
        """Batch prefetch proxies into cache."""
        uids_to_load = [uid for uid in uids if uid not in self._proxy_cache]
        if not uids_to_load:
            return

        meta_batch = self._metadata_index.get_batch(self.TABLE, uids_to_load)

        for uid, meta_dict in zip(uids_to_load, meta_batch, strict=True):
            if meta_dict is not None:
                meta = self.METADATA_CLASS.from_dict(meta_dict)
                self._proxy_cache[uid] = self.PROXY_CLASS(meta=meta, _repository=self)

    @classmethod
    def define_schema(cls, metadata_index: IMetadataIndex) -> None:
        """Define metadata schema in the index. Override in subclasses."""
        metadata_index.define_table(
            cls.TABLE,
            columns=cls.METADATA_COLUMNS,
            indexes=cls.METADATA_INDEXES,
        )

    @classmethod
    def initialize(
        cls,
        storage_manager: IStorageManager,
        metadata_index: IMetadataIndex,
        cache_manager: ICacheManager | None = None,
        parallel: ParallelContext | None = None,
    ) -> Self:
        """Initialize repository with schema definition."""
        cls.define_schema(metadata_index)
        return cls(
            storage_manager=storage_manager,
            metadata_index=metadata_index,
            cache_manager=cache_manager,
        )

    def create_view(self) -> DataView[TMetadata, TProxy]:
        """Create a view over all entities in the repository."""
        return self.VIEW_CLASS(repository=self, uids=None)
