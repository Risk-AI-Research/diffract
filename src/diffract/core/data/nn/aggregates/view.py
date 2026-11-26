"""View for aggregate entities.

This module provides the AggregateView class for batch operations
on filtered aggregate collections.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

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

    _repository: AggregateRepository

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

    def read_catalog_values(self) -> dict[str, tuple[Any, str | None]] | None:
        """Read each value and its run ID from the same catalog pointer."""
        storage = self._repository.storage_manager
        if getattr(storage, "uses_value_revisions", False) is not True:
            return None
        pointers = storage.live_pointers(
            self.list_uids(),
            ("value",),
            table=TABLE_AGGREGATES,
            include_generation=True,
        )
        return {
            uid: (
                self._repository.get_proxy(uid).read_resolved_field(
                    field, pointer, auto_prefetch=True, best_effort=True
                ),
                pointer.generation.run_id if pointer.generation else None,
            )
            for (uid, field), pointer in pointers.items()
        }

    def _sort_in_place(self) -> None:
        """Sort UIDs by field_name, context for deterministic ordering."""
        uids = self._ensure_uids()

        def order(uid: str) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
            meta = self._repository.get_proxy(uid).meta
            return meta.field_name, meta.context_models, meta.context_params, uid

        uids.sort(key=order)
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

    def within_parameters(self, parameter_uids: list[str]) -> AggregateView:
        """Keep aggregates whose complete, nonempty membership is in the scope."""
        candidates = self._repository.referencing(parameter_uids)
        if self._uids is not None:
            allowed = set(self._uids)
            candidates = [uid for uid in candidates if uid in allowed]
        rows = self._repository.metadata_index.query_rows(
            self._repository.MEMBERS,
            columns=("aggregate_uid", "member_uid"),
            where_in={"aggregate_uid": candidates},
        )
        scope = set(parameter_uids)
        outside = {
            row["aggregate_uid"] for row in rows if row["member_uid"] not in scope
        }
        return AggregateView(
            repository=self._repository,
            uids=[uid for uid in candidates if uid not in outside],
        )

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
        exact_names: set[str] = set()
        regex_patterns: list[re.Pattern[str]] = []
        for model_id in model_ids:
            if model_id.startswith(REGEX_PREFIX):
                regex_patterns.append(re.compile(model_id.removeprefix(REGEX_PREFIX)))
            else:
                exact_names.add(model_id)

        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()

        def _matches(context_models: tuple[str, ...]) -> bool:
            context_set = set(context_models)

            if exact_names:
                if require_all:
                    if not exact_names <= context_set:
                        return False
                elif not (exact_names & context_set):
                    return False

            if regex_patterns:
                if require_all:
                    for pattern in regex_patterns:
                        if not any(
                            pattern.fullmatch(cm) is not None for cm in context_models
                        ):
                            return False
                elif not any(
                    any(pattern.fullmatch(cm) is not None for pattern in regex_patterns)
                    for cm in context_models
                ):
                    return False

            return True

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
            *param_names: Parameter names that must be in the aggregate's
                context_params.
            require_all: If True, all param_names must be present. If False,
                         at least one must be present.

        Returns:
            New view with filtered aggregates.
        """
        exact_names: set[str] = set()
        regex_patterns: list[re.Pattern[str]] = []
        for param_name in param_names:
            if param_name.startswith(REGEX_PREFIX):
                regex_patterns.append(re.compile(param_name.removeprefix(REGEX_PREFIX)))
            else:
                exact_names.add(param_name)

        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()

        def _matches(context_params: tuple[str, ...]) -> bool:
            context_set = set(context_params)

            if exact_names:
                if require_all:
                    if not exact_names <= context_set:
                        return False
                elif not (exact_names & context_set):
                    return False

            if regex_patterns:
                if require_all:
                    for pattern in regex_patterns:
                        if not any(
                            pattern.fullmatch(cp) is not None for cp in context_params
                        ):
                            return False
                elif not any(
                    any(pattern.fullmatch(cp) is not None for pattern in regex_patterns)
                    for cp in context_params
                ):
                    return False

            return True

        filtered = [
            uid
            for uid in uids
            if _matches(self._repository.get_proxy(uid).meta.context_params)
        ]

        return AggregateView(repository=self._repository, uids=filtered)

    def filter_by_exact_context(
        self,
        *,
        models: tuple[str, ...] | None = None,
        params: tuple[str, ...] | None = None,
    ) -> AggregateView:
        """Filter aggregates by exact context match.

        Unlike filter_by_context_models/params which use set intersection,
        this method requires exact equality of the context tuples.

        Args:
            models: Exact context_models tuple to match. If None, any models match.
            params: Exact context_params tuple to match. If None, any params match.

        Returns:
            New view with aggregates matching the exact context.
        """
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()

        def _matches(proxy: AggregateProxy) -> bool:
            if models is not None and proxy.meta.context_models != models:
                return False
            return not (params is not None and proxy.meta.context_params != params)

        filtered = [uid for uid in uids if _matches(self._repository.get_proxy(uid))]

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

        with self._repository:
            if erase:
                self._repository.erase_values(uids)
            self._repository.metadata_index.delete_rows(
                self._repository.MEMBERS, where_in={"aggregate_uid": uids}
            )
            self._repository.remove_membership(uids)

        if self._uids is not None:
            self._uids.clear()
        else:
            self._uids = []
        self._sorted = True
