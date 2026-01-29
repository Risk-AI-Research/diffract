"""View for aggregate entities.

This module provides the AggregateView class for batch operations
on filtered aggregate collections.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from diffract.core.constants import REGEX_PREFIX, TABLE_AGGREGATES
from diffract.core.data.interface import EntityUID
from diffract.core.data.utils import build_matcher
from diffract.core.data.view import DataView

from .metadata import AggregateMetadata
from .proxy import AggregateProxy

if TYPE_CHECKING:
    from .repository import AggregateRepository


class AggregateView(DataView[AggregateMetadata, AggregateProxy]):
    """A numpy-like view over a subset of aggregates owned by a repository.

    Extends the generic DataView with aggregate-specific filtering methods.
    Uses SQL-based filtering via MetadataIndex for efficiency.
    """

    def __init__(
        self,
        *,
        repository: AggregateRepository,
        uids: list[EntityUID] | None = None,
    ) -> None:
        super().__init__(repository=repository, uids=uids)

    def __iter__(self) -> Iterator[AggregateProxy]:
        """Iterate aggregate proxies in view order."""
        return super().__iter__()

    def _sort_in_place(self) -> None:
        """Sort UIDs by field_name, context for deterministic ordering."""
        uids = self._ensure_uids()
        uids.sort(
            key=lambda uid: (
                self._repository.get_proxy(uid).meta.field_name,
                ",".join(sorted(self._repository.get_proxy(uid).meta.context_models)),
                ",".join(sorted(self._repository.get_proxy(uid).meta.context_params)),
                uid,
            )
        )
        self._sorted = True

    def filter_by_field_name(
        self, *names: str, inverse_mask: bool = False
    ) -> AggregateView:
        """Filter aggregates by field name.

        Args:
            *names: Field names to match. Use "re:" prefix for regex patterns.
            inverse_mask: If True, return aggregates NOT matching the names.

        Returns:
            New view with filtered aggregates.
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
                    TABLE_AGGREGATES,
                    where_in={"field_name": exact_names},
                )
            else:
                # Filter within current UIDs
                filtered = self._repository.metadata_index.query(
                    TABLE_AGGREGATES,
                    where_in={"field_name": exact_names, "uid": current_uids},
                )
            return AggregateView(repository=self._repository, uids=filtered)

        # Fall back to in-memory filtering for regex or inverse
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        matches = build_matcher(names)

        if not inverse_mask:
            filtered = [
                uid
                for uid in uids
                if matches(self._repository.get_proxy(uid).meta.field_name)
            ]
        else:
            filtered = [
                uid
                for uid in uids
                if not matches(self._repository.get_proxy(uid).meta.field_name)
            ]

        return AggregateView(repository=self._repository, uids=filtered)

    def filter_by_context_models(
        self, *model_ids: str, require_all: bool = False
    ) -> AggregateView:
        """Filter aggregates by participating models.

        Args:
            *model_ids: Model IDs that must be in the aggregate's context_models.
            require_all: If True, all model_ids must be present. If False,
                         at least one must be present.

        Returns:
            New view with filtered aggregates.
        """
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        model_set = set(model_ids)

        def _matches(context_models: tuple[str, ...]) -> bool:
            context_set = set(context_models)
            if require_all:
                return model_set <= context_set
            return bool(model_set & context_set)

        filtered = [
            uid
            for uid in uids
            if _matches(self._repository.get_proxy(uid).meta.context_models)
        ]

        return AggregateView(repository=self._repository, uids=filtered)

    def filter_by_context_params(
        self, *param_names: str, require_all: bool = False
    ) -> AggregateView:
        """Filter aggregates by participating parameters.

        Args:
            *param_names: Parameter names that must be in the aggregate's context_params.
            require_all: If True, all param_names must be present. If False,
                         at least one must be present.

        Returns:
            New view with filtered aggregates.
        """
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        param_set = set(param_names)

        def _matches(context_params: tuple[str, ...]) -> bool:
            context_set = set(context_params)
            if require_all:
                return param_set <= context_set
            return bool(param_set & context_set)

        filtered = [
            uid
            for uid in uids
            if _matches(self._repository.get_proxy(uid).meta.context_params)
        ]

        return AggregateView(repository=self._repository, uids=filtered)

    def erase_fields(self, *fields: str) -> None:
        """Erase fields from all aggregates in this view."""
        uids = self._ensure_uids()
        if not uids:
            return

        with self:
            for field in fields:
                self._repository.storage_manager.erase_field_for_all(
                    field, table=TABLE_AGGREGATES
                )
                if self._repository.cache_manager is not None:
                    self._repository.cache_manager.erase_field_for_all(field)

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
                    TABLE_AGGREGATES, uid
                )
                if erase:
                    self._repository.storage_manager.erase_obj(
                        uid, table=TABLE_AGGREGATES
                    )

        if self._uids is not None:
            self._uids.clear()
        else:
            self._uids = []
        self._sorted = True
