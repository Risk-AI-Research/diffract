"""Storage-backed parameter data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.proxy import DataProxy

from .interface import IParameterProxy
from .metadata import ParameterMetadata

if TYPE_CHECKING:
    from .interface import IParameterRepository


@dataclass(kw_only=True, eq=False)
class ParameterDataProxy(
    DataProxy[ParameterMetadata, "IParameterRepository"], IParameterProxy
):
    """Read and write fields for a parameter through its owning repository."""

    meta: ParameterMetadata
    _repository: IParameterRepository = field(repr=False)

    def update_other_meta(self, updates: dict[str, Any]) -> None:
        """Merge metadata keys into this parameter's persisted record."""
        meta = replace(self.meta, other_meta=self.meta.other_meta | updates)
        self.meta = self._repository.store_metadata(
            meta, instance_uid=meta.instance_uid, update_existing=True
        )

    @classmethod
    def get_table(cls) -> str:
        """Return the storage table name for parameters."""
        return TABLE_PARAMETERS

    @classmethod
    def create_and_store(
        cls,
        meta: ParameterMetadata,
        repository: IParameterRepository,
        *,
        instance_uid: str | None = None,
    ) -> ParameterDataProxy:
        """Store metadata under the reserved instance, then return its proxy."""
        stored = repository.store_metadata(meta, instance_uid=instance_uid)
        proxy = cls(meta=stored, _repository=repository)
        repository.append(proxy)
        return proxy
