"""Parameter view for batch operations on filtered parameter collections.

This module provides the ParameterView class that extends the generic DataView
with parameter-specific filtering methods.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from diffract.core.constants import REGEX_PREFIX, TABLE_PARAMETERS
from diffract.core.data.metadata.runs import RunJournal
from diffract.core.data.utils import build_matcher
from diffract.core.data.view import DataView
from diffract.core.parallel import ParallelContext, map_maybe_parallel
from diffract.core.storage.catalog_manager import CatalogStorageManager

from .interface import IParameterProxy, IParameterView
from .metadata import ParameterMetadata
from .schema import FieldName, ModelID, ParameterType, ParameterUID

if TYPE_CHECKING:
    from diffract.core.storage.catalog import ValuePointer

    from .interface import IParameterRepository


logger = logging.getLogger(__name__)


class ParameterView(DataView[ParameterMetadata, IParameterProxy], IParameterView):
    """A numpy-like view over a subset of parameters owned by a repository.

    Extends the generic DataView with parameter-specific filtering methods
    such as filter_by_name, filter_by_ptype, and filter_by_model_id.
    Uses SQL-based filtering via MetadataIndex for efficiency.
    """

    _repository: IParameterRepository

    def __init__(
        self,
        *,
        repository: IParameterRepository,
        uids: list[ParameterUID] | None = None,
    ) -> None:
        super().__init__(repository=repository, uids=uids)
        self._starts_unscoped = uids is None

    def _sort_in_place(self) -> None:
        """Sort parameters by model, numeric step, name, and UID."""
        uids = self._ensure_uids()
        self._repository.prefetch_proxies(uids)

        def order(uid: str) -> tuple[str, int, str, str]:
            meta = self._repository.get_proxy(uid).meta
            return meta.model_id, meta.step, meta.name, uid

        uids.sort(key=order)
        self._sorted = True

    def filter_by_name(self, *names: str, inverse_mask: bool = False) -> ParameterView:
        """Filter parameters by name.

        Args:
            *names: Parameter names to match. Use "re:" prefix for regex patterns.
            inverse_mask: If True, return parameters NOT matching the names.

        Returns:
            New view with filtered parameters.
        """
        # Separate exact matches from regex patterns
        exact_names: list[str] = []
        regex_patterns: list[re.Pattern[str]] = []
        for name in names:
            if name.startswith(REGEX_PREFIX):
                regex_patterns.append(re.compile(name.removeprefix(REGEX_PREFIX)))
            else:
                exact_names.append(name)

        # If only exact matches and no inverse, use SQL
        if exact_names and not regex_patterns and not inverse_mask:
            current_uids = self._uids
            if current_uids is None:
                # Query directly from index
                filtered = self._repository.metadata_index.query(
                    TABLE_PARAMETERS,
                    where_in={"name": exact_names},
                )
            else:
                # Filter within current UIDs
                filtered = self._repository.metadata_index.query(
                    TABLE_PARAMETERS,
                    where_in={"name": exact_names, "uid": current_uids},
                )
            return ParameterView(repository=self._repository, uids=filtered)

        # Fall back to in-memory filtering for regex or inverse
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        matches = build_matcher(names)

        if not inverse_mask:
            filtered = [
                uid
                for uid in uids
                if matches(self._repository.get_proxy(uid).meta.name)
            ]
        else:
            filtered = [
                uid
                for uid in uids
                if not matches(self._repository.get_proxy(uid).meta.name)
            ]

        return ParameterView(repository=self._repository, uids=filtered)

    def filter_by_ptype(self, *ptypes: str | ParameterType) -> ParameterView:
        """Filter parameters by parameter type.

        Args:
            *ptypes: Parameter types to include (strings or ParameterType values).

        Returns:
            New view with filtered parameters.
        """
        # Convert to string names for SQL query
        ptype_names: list[str] = []
        for ptype in ptypes:
            if isinstance(ptype, ParameterType):
                ptype_names.append(ptype.name)
            else:
                ptype_names.append(ptype.upper())

        current_uids = self._uids
        if current_uids is None:
            # Query directly from index
            filtered = self._repository.metadata_index.query(
                TABLE_PARAMETERS,
                where_in={"ptype": ptype_names},
            )
        else:
            # Filter within current UIDs
            filtered = self._repository.metadata_index.query(
                TABLE_PARAMETERS,
                where_in={"ptype": ptype_names, "uid": current_uids},
            )

        return ParameterView(repository=self._repository, uids=filtered)

    def step_labels(self) -> dict[tuple[str, int], str]:
        """Return the display labels recorded for checkpoints in the store."""
        return self._repository.instances.step_labels()

    def coordinates(self, *, whole_store: bool = False) -> list[tuple[str, int]]:
        """List distinct (instance_uid, step) pairs in numeric step order."""
        scope = None if whole_store or self._uids is None else {"uid": self._uids}
        rows = self._repository.metadata_index.query_rows(
            TABLE_PARAMETERS,
            columns=("instance_uid", "step"),
            where_in=scope,
            distinct=True,
            order_by=("step", "instance_uid"),
        )
        return [(row["instance_uid"], row["step"]) for row in rows]

    def filter_by_coordinates(
        self, coordinates: list[tuple[str, int]]
    ) -> ParameterView:
        """Keep parameters at the selected model and step pairs."""
        by_instance: dict[str, list[int]] = {}
        for instance, step in coordinates:
            by_instance.setdefault(instance, []).append(step)
        selected: list[str] = []
        scope = None if self._uids is None else set(self._uids)
        for instance, steps in by_instance.items():
            matches = self._repository.metadata_index.query(
                TABLE_PARAMETERS,
                where={"instance_uid": instance},
                where_in={"step": steps},
            )
            selected.extend(uid for uid in matches if scope is None or uid in scope)
        return ParameterView(repository=self._repository, uids=selected)

    def filter_by_model_id(
        self, *model_ids: ModelID, inverse_mask: bool = False
    ) -> ParameterView:
        """Filter parameters by model id.

        Args:
            *model_ids: Model IDs to match. Use "re:" prefix for regex patterns.
            inverse_mask: If True, return parameters NOT matching the model IDs.

        Returns:
            New view with filtered parameters.
        """
        instances = self._repository.instances.resolve(model_ids, inverse=inverse_mask)
        where_in = {"instance_uid": instances}
        if self._uids is not None:
            where_in["uid"] = self._uids
        filtered = self._repository.metadata_index.query(
            TABLE_PARAMETERS, where_in=where_in
        )
        return ParameterView(repository=self._repository, uids=filtered)

    def pending_outputs(self, *fields: str) -> ParameterView | None:
        """Select incomplete parameters using a whole-store lookup.

        Return None when the caller must fall back to filtering the view.
        An empty view means the lookup found no incomplete parameters.
        """
        storage = self._repository.storage_manager
        lookup = getattr(type(storage), "owners_missing_outputs", None)
        if self._uids is not None or lookup is None:
            return None
        uids = lookup(storage, fields, table=TABLE_PARAMETERS)
        return (
            None
            if uids is None
            else ParameterView(repository=self._repository, uids=uids)
        )

    def filter_by_fields(
        self,
        *fields: FieldName,
        inverse_mask: bool = False,
        parallel: ParallelContext | None = None,
    ) -> ParameterView:
        """Filter parameters by the presence of one or more fields.

        Args:
            *fields: Field names that must be present.
            inverse_mask: If True, return parameters WITHOUT the specified fields.
            parallel: Optional parallel context for concurrent checks.

        Returns:
            New view with filtered parameters.
        """
        result = super().filter_by_fields(
            *fields, inverse_mask=inverse_mask, parallel=parallel
        )
        return ParameterView(repository=self._repository, uids=result._uids)

    def read_catalog_fields(
        self,
        fields: tuple[str, ...],
        *,
        accept: Callable[[ParameterMetadata, str], bool] | None = None,
        run_ids: dict[tuple[str, str], str] | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        """Read the current live catalog selection for export, when supported."""
        storage = self._repository.storage_manager
        if getattr(storage, "uses_value_revisions", False) is not True:
            return None
        whole_repository = self._uids is None
        uids = self.list_uids()
        result = {uid: {} for uid in uids}

        def selected_pointers() -> Iterator[tuple[tuple[str, str], ValuePointer]]:
            resolve_many = getattr(type(storage), "live_pointers", None)
            if resolve_many is not None:
                unscoped = whole_repository and (
                    getattr(storage, "supports_unscoped_pointers", False) is True
                )
                pointers = resolve_many(
                    storage,
                    None if unscoped else uids,
                    fields,
                    table=TABLE_PARAMETERS,
                    **({"include_generation": True} if run_ids is not None else {}),
                )
                yield from pointers.items()
                return
            for uid in uids:
                for field in dict.fromkeys(fields):
                    try:
                        pointer = storage.live_pointer(uid, field, TABLE_PARAMETERS)
                    except KeyError:  # noqa: S112 - omit absent logical fields
                        continue
                    yield (uid, field), pointer

        for (uid, field), pointer in selected_pointers():
            if uid not in result:
                continue
            param = self._repository.get_proxy(uid)
            if accept is None or accept(param.meta, field):
                result[uid][field] = param.read_resolved_field(
                    field, pointer, auto_prefetch=True, best_effort=True
                )
                if run_ids is not None and pointer.generation is not None:
                    run_ids[uid, field] = pointer.generation.run_id
        return result

    def _uncached_kernel_catalog(self) -> CatalogStorageManager | None:
        storage = self._repository.storage_manager
        # Subclasses opt in explicitly because direct pointer reads bypass proxy hooks.
        if (
            self._repository.cache_manager is None
            and vars(type(self._repository)).get("supports_direct_kernel_reads") is True
            and isinstance(storage, CatalogStorageManager)
            and vars(type(storage)).get("supports_direct_kernel_reads") is True
        ):
            return storage
        return None

    def kernel_inputs_are_resident(self) -> bool:
        """Return whether uncached input reads borrow values already in RAM."""
        storage = self._uncached_kernel_catalog()
        return storage is not None and storage.kernel_inputs_are_resident

    def kernel_trace_scope(
        self,
    ) -> tuple[CatalogStorageManager, list[str], bool] | None:
        """Return the catalog, scope snapshot, and whether it began unfiltered."""
        storage = self._repository.storage_manager
        return (
            (storage, self._ensure_uids(), self._starts_unscoped)
            if isinstance(storage, CatalogStorageManager)
            and isinstance(storage.journal, RunJournal)
            else None
        )

    def kernel_comparison_budget(self) -> int:
        """Return the byte budget shared by input reads and output comparison."""
        return self._get_cache_budget_bytes()

    def resolve_kernel_inputs(
        self, fields: tuple[FieldName, ...]
    ) -> dict[tuple[ParameterUID, FieldName], ValuePointer] | None:
        """Resolve an uncached compute chunk without one catalog query per input."""
        storage = self._uncached_kernel_catalog()
        if storage is None:
            return None
        if not fields:
            return {}
        return storage.live_pointers(self.list_uids(), fields, table=TABLE_PARAMETERS)

    def iter_chunks_by_read_budget(
        self,
        *,
        budget_bytes: int | None = None,
        required_fields_by_uid: dict[ParameterUID, list[FieldName]],
        default_field_bytes: int = 1024 * 1024,
        parallel: ParallelContext | None = None,
    ) -> Iterator[ParameterView]:
        """Yield sub-views chunked by approximate read budget.

        Args:
            budget_bytes: Maximum bytes per chunk. If None, auto-detects from
                cache_manager.get_available_bytes() with 40% headroom factor.
            required_fields_by_uid: Mapping of uid -> required fields to include
                in the estimate.
            default_field_bytes: Fallback size estimate when metadata is missing.
            parallel: Optional parallel context for metadata estimation.
        """
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        if not uids:
            return

        if not any(required_fields_by_uid.get(uid) for uid in uids):
            yield self
            return

        if budget_bytes is None:
            budget_bytes = self._get_cache_budget_bytes()

        if budget_bytes <= 0:
            yield self
            return

        storage = self._repository.storage_manager
        bulk = getattr(type(storage), "field_metadata_for", None)
        sizes = None
        metadata = None
        read_sizes = getattr(type(storage), "read_sizes", None)
        if read_sizes is not None or bulk is not None:
            requested = {
                uid: required_fields_by_uid[uid]
                for uid in uids
                if uid in required_fields_by_uid
            }
            try:
                if read_sizes is not None:
                    sizes = read_sizes(storage, requested, table=TABLE_PARAMETERS)
                else:
                    metadata = bulk(storage, requested, table=TABLE_PARAMETERS)
            except OSError:
                logger.debug("Bulk metadata read failed", exc_info=True)

        def _estimate_field_bytes(uid: ParameterUID, field: FieldName) -> int:
            if sizes is not None:
                return sizes.get((uid, field), default_field_bytes)
            meta = (
                self._repository.get_proxy(uid).get_field_metadata(field)
                if metadata is None
                else metadata.get((uid, field))
            )
            if not meta:
                return default_field_bytes

            shape = meta.get("shape")
            dtype = meta.get("dtype")
            if not shape or not dtype:
                return default_field_bytes

            try:
                itemsize = int(np.dtype(dtype).itemsize)
                n = 1
                for x in shape:
                    n *= int(x)
                return max(1, n * itemsize)
            except Exception:  # noqa: BLE001
                return default_field_bytes

        def _estimate_uid(uid: ParameterUID) -> int:
            fields = required_fields_by_uid.get(uid, [])
            if not fields:
                return 0
            return sum(_estimate_field_bytes(uid, f) for f in fields)

        estimates = (
            map_maybe_parallel(
                uids, _estimate_uid, parallel=self._read_parallelism(parallel)
            )
            if metadata is None and sizes is None
            else map(_estimate_uid, uids)
        )

        chunk: list[ParameterUID] = []
        used = 0
        for uid, estimated_bytes in zip(uids, estimates, strict=True):
            if chunk and (used + estimated_bytes > budget_bytes):
                yield ParameterView(repository=self._repository, uids=list(chunk))
                chunk.clear()
                used = 0
            chunk.append(uid)
            used += estimated_bytes
        if chunk:
            yield ParameterView(repository=self._repository, uids=chunk)

    def erase_fields(self, *fields: FieldName) -> None:
        """Erase fields from all parameters in this view."""
        uids = self._ensure_uids()
        if not uids:
            return

        with self:
            for uid in uids:
                for field in fields:
                    self._repository.storage_manager.erase_field(
                        uid, field, table=TABLE_PARAMETERS
                    )
                    if self._repository.cache_manager is not None:
                        self._repository.cache_manager.erase_field(uid, field)

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Erase fields matching regex patterns from all parameters in this view."""
        uids = self._ensure_uids()
        if not uids:
            return

        all_fields = set(
            self._repository.storage_manager.list_fields(table=TABLE_PARAMETERS)
        )

        with self:
            for uid in uids:
                for pattern in patterns:
                    for field_name in all_fields:
                        if re.fullmatch(pattern, field_name):
                            self._repository.storage_manager.erase_field(
                                uid, field_name, table=TABLE_PARAMETERS
                            )
                            if self._repository.cache_manager is not None:
                                self._repository.cache_manager.erase_field(
                                    uid, field_name
                                )

    def clear(self, erase: bool = False) -> None:
        """Clear this view and optionally erase corresponding data.

        Args:
            erase: If True, also erase underlying storage data.
                   If False, only clear membership (metadata index).
        """
        uids = self._ensure_uids()

        with self._repository:
            if erase:
                self._repository.erase_values(uids)
            self._repository.remove_membership(uids)

        if self._uids is not None:
            self._uids.clear()
        else:
            self._uids = []
        self._sorted = True

    def _get_cache_budget_bytes(self, headroom_factor: float = 0.4) -> int:
        """Get read budget based on available cache capacity.

        Args:
            headroom_factor: Fraction of available cache to use (default 40%).

        Returns:
            Budget in bytes, or fallback value if cache unavailable.
        """
        fallback = 256 * 1024 * 1024  # 256MB

        cache = self._repository.cache_manager
        if cache is None:
            return fallback

        available = cache.get_available_bytes()
        if available is None:
            return fallback

        return max(1024 * 1024, int(available * headroom_factor))
