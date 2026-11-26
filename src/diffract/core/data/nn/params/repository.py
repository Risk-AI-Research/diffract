"""Parameter repository for managing neural network parameters.

This module provides the ParameterRepository class that extends the generic
DataRepository with parameter-specific functionality.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Self

from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.identity import DuplicateParameterError
from diffract.core.data.metadata.errors import UniqueConstraintError
from diffract.core.data.nn.instances.repository import InstanceRepository
from diffract.core.data.repository import DataRepository
from diffract.core.utils.hashing import new_entity_uid

from .interface import IParameterProxy, IParameterRepository
from .metadata import ParameterMetadata
from .proxy import ParameterDataProxy
from .view import ParameterView

if TYPE_CHECKING:
    from diffract.core.cache.interface import ICacheManager
    from diffract.core.data.metadata.interface import IMetadataIndex
    from diffract.core.parallel import ParallelContext
    from diffract.core.storage.interface import IStorageManager


class ParameterRepository(
    DataRepository[ParameterMetadata, IParameterProxy], IParameterRepository
):
    """Repository owning storage/cache/metadata and managing parameter membership.

    Extends the generic DataRepository with parameter-specific configuration
    for metadata class, proxy class, view class, storage table, and metadata schema.
    """

    supports_direct_kernel_reads = True
    METADATA_CLASS = ParameterMetadata
    PROXY_CLASS = ParameterDataProxy
    VIEW_CLASS = ParameterView
    TABLE = TABLE_PARAMETERS

    # Schema for MetadataIndex
    METADATA_COLUMNS: ClassVar[dict[str, type]] = {
        "name": str,
        "instance_uid": str,
        "step": int,
        "ptype": str,
    }
    METADATA_INDEXES: ClassVar[list[str]] = []

    def __init__(
        self,
        storage_manager: IStorageManager,
        metadata_index: IMetadataIndex,
        cache_manager: ICacheManager | None = None,
        *,
        instances: InstanceRepository | None = None,
    ) -> None:
        super().__init__(storage_manager, metadata_index, cache_manager)
        self.instances = instances or InstanceRepository.initialize(metadata_index)
        self._names: dict[str, str] = {}

    @classmethod
    def initialize(
        cls,
        storage_manager: IStorageManager,
        metadata_index: IMetadataIndex,
        cache_manager: ICacheManager | None = None,
        parallel: ParallelContext | None = None,
        *,
        instances: InstanceRepository | None = None,
    ) -> Self:
        """Bind the parameter repository and register its index columns."""
        cls.define_schema(metadata_index)
        return cls(storage_manager, metadata_index, cache_manager, instances=instances)

    def refresh_names(self) -> None:
        """Refresh model name projections on the next read."""
        self._names.clear()

    def _name(self, instance_uid: str) -> str:
        if instance_uid not in self._names:
            instance = self.instances.get(instance_uid)
            if instance is None:
                raise KeyError(instance_uid)
            self._names[instance_uid] = instance.name
        return self._names[instance_uid]

    def _decode_metadata(self, row: dict[str, Any]) -> ParameterMetadata:
        return ParameterMetadata.from_dict(
            {**row, "model_id": self._name(row["instance_uid"])}
        )

    def get_proxy(self, uid: str) -> IParameterProxy:
        """Read a parameter with its current model name."""
        proxy = self._proxy_cache.get(uid)
        if proxy is None:
            return super().get_proxy(uid)
        meta = proxy.meta
        if (
            meta.instance_uid is not None
            and self._names.get(meta.instance_uid) != meta.model_id
        ):
            name = self._name(meta.instance_uid)
            if meta.model_id != name:
                proxy.meta = replace(meta, model_id=name)
        return proxy

    def store_metadata(
        self,
        meta: ParameterMetadata,
        *,
        instance_uid: str | None = None,
        update_existing: bool = False,
    ) -> ParameterMetadata:
        """Persist the parameter with its store-local model identity."""
        with self.metadata_index as index:
            if instance_uid is None:
                instance_uid = self.instances.get_or_create(meta.model_id)
            write = index.upsert if update_existing else index.insert
            try:
                write(
                    self.TABLE,
                    uid=meta.uid,
                    instance_uid=instance_uid,
                    step=meta.step,
                    name=meta.name,
                    ptype=meta.ptype.name,
                    other_meta=meta.other_meta,
                )
            except UniqueConstraintError as exc:
                if exc.table == self.TABLE and exc.columns == (
                    "instance_uid",
                    "step",
                    "name",
                ):
                    raise DuplicateParameterError(
                        instance_uid, meta.step, meta.name
                    ) from exc
                raise
        return (
            meta
            if meta.instance_uid == instance_uid
            else replace(meta, instance_uid=instance_uid)
        )

    def merge_target(
        self, source: ParameterMetadata, *, overwrite: bool
    ) -> IParameterProxy:
        """Resolve the destination coordinate and retain its local UID on update."""
        with self.metadata_index as index:
            instance_uid = self.instances.get_or_create(source.model_id)
            matches = index.query(
                self.TABLE,
                where={
                    "instance_uid": instance_uid,
                    "step": source.step,
                    "name": source.name,
                },
                limit=1,
            )
            if matches:
                proxy = self.get_proxy(matches[0])
                if overwrite:
                    proxy.meta = self.store_metadata(
                        replace(source, uid=proxy.meta.uid),
                        instance_uid=instance_uid,
                        update_existing=True,
                    )
                return proxy
            if self.has_uid(source.uid):
                source = replace(source, uid=new_entity_uid())
            return ParameterDataProxy.create_and_store(
                source, self, instance_uid=instance_uid
            )

    def create_view(self) -> ParameterView:
        """Create a view over all parameters in this repository.

        Returns:
            ParameterView containing all parameters.
        """
        return ParameterView(repository=self, uids=None)
