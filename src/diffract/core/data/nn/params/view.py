"""Parameter view for batch operations on filtered parameter collections.

This module provides the ParameterView class that extends the generic DataView
with parameter-specific filtering methods.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from diffract.core.compute.parallel import ParallelContext, map_maybe_parallel
from diffract.core.constants import REGEX_PREFIX, TABLE_PARAMETERS
from diffract.core.data.utils import build_matcher
from diffract.core.data.view import DataView

from .metadata import ParameterMetadata
from .proxy import ParameterDataProxy
from .schema import FieldName, ModelID, ParameterType, ParameterUID

if TYPE_CHECKING:
    from .interface import IParameterProxy, IParameterRepository


logger = logging.getLogger(__name__)


class ParameterView(DataView[ParameterMetadata, ParameterDataProxy]):
    """A numpy-like view over a subset of parameters owned by a repository.

    Extends the generic DataView with parameter-specific filtering methods
    such as filter_by_name, filter_by_ptype, and filter_by_model_id.
    Uses SQL-based filtering via MetadataIndex for efficiency.
    """

    def __init__(
        self,
        *,
        repository: IParameterRepository,
        uids: list[ParameterUID] | None = None,
    ) -> None:
        super().__init__(repository=repository, uids=uids)

    def __iter__(self) -> Iterator[IParameterProxy]:
        """Iterate parameter proxies in view order."""
        return super().__iter__()

    def _sort_in_place(self) -> None:
        """Sort UIDs by model_id, name, then uid for deterministic ordering."""
        uids = self._ensure_uids()
        uids.sort(
            key=lambda uid: (
                self._repository.get_proxy(uid).meta.model_id,
                self._repository.get_proxy(uid).meta.name,
                uid,
            )
        )
        self._sorted = True

    def filter_by_name(
        self, *names: str, inverse_mask: bool = False
    ) -> ParameterView:
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
        # Separate exact matches from regex patterns
        exact_ids: list[str] = []
        regex_patterns: list[re.Pattern[str]] = []
        for model_id in model_ids:
            if model_id.startswith(REGEX_PREFIX):
                regex_patterns.append(re.compile(model_id.removeprefix(REGEX_PREFIX)))
            else:
                exact_ids.append(model_id)

        # If only exact matches and no inverse, use SQL
        if exact_ids and not regex_patterns and not inverse_mask:
            current_uids = self._uids
            if current_uids is None:
                # Query directly from index
                filtered = self._repository.metadata_index.query(
                    TABLE_PARAMETERS,
                    where_in={"model_id": exact_ids},
                )
            else:
                # Filter within current UIDs
                filtered = self._repository.metadata_index.query(
                    TABLE_PARAMETERS,
                    where_in={"model_id": exact_ids, "uid": current_uids},
                )
            return ParameterView(repository=self._repository, uids=filtered)

        # Fall back to in-memory filtering for regex or inverse
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        matches = build_matcher(model_ids)

        if not inverse_mask:
            filtered = [
                uid
                for uid in uids
                if matches(self._repository.get_proxy(uid).meta.model_id)
            ]
        else:
            filtered = [
                uid
                for uid in uids
                if not matches(self._repository.get_proxy(uid).meta.model_id)
            ]

        return ParameterView(repository=self._repository, uids=filtered)

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

        if budget_bytes is None:
            budget_bytes = self._get_cache_budget_bytes()

        if budget_bytes <= 0:
            yield self
            return

        def _estimate_field_bytes(uid: ParameterUID, field: FieldName) -> int:
            meta = self._repository.get_proxy(uid).get_field_metadata(field)
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

        estimates = map_maybe_parallel(uids, _estimate_uid, parallel=parallel)
        uid_to_est = dict(zip(uids, estimates, strict=True))

        chunk: list[ParameterUID] = []
        used = 0
        for uid in uids:
            estimated_bytes = uid_to_est[uid]
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
            for field in fields:
                self._repository.storage_manager.erase_field_for_all(
                    field, table=TABLE_PARAMETERS
                )
                if self._repository.cache_manager is not None:
                    self._repository.cache_manager.erase_field_for_all(field)

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Erase fields matching regex patterns from all parameters in this view."""
        uids = self._ensure_uids()
        if not uids:
            return

        all_fields = set(
            self._repository.storage_manager.list_fields(table=TABLE_PARAMETERS)
        )

        with self:
            for pattern in patterns:
                for field_name in all_fields:
                    if re.fullmatch(pattern, field_name):
                        self._repository.storage_manager.erase_field_for_all(
                            field_name, table=TABLE_PARAMETERS
                        )
                        if self._repository.cache_manager is not None:
                            self._repository.cache_manager.erase_field_for_all(
                                field_name
                            )

    def clear(self, erase: bool = False) -> None:
        """Clear this view and optionally erase corresponding data.
        
        Args:
            erase: If True, also erase underlying storage data.
                   If False, only clear membership (metadata index).
        """
        uids = self._ensure_uids()

        if erase:
            if self._repository.cache_manager is not None:
                self._repository.cache_manager.clear()

        with self._repository:
            for uid in list(uids):
                self._repository._proxy_cache.pop(uid, None)
                self._repository.metadata_index.delete(
                    TABLE_PARAMETERS, uid
                )
                if erase:
                    self._repository.storage_manager.erase_obj(
                        uid, table=TABLE_PARAMETERS
                    )

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

