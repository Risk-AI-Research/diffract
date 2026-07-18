"""Session-level cache for field availability.

Caches the result of list_fields_by_uid() to avoid repeated expensive
storage scans when multiple export_metrics() calls are made without
intervening compute.apply() or other mutation operations.

The cache uses a generation counter for lazy invalidation: when a mutation
occurs, the generation increments and cached data is considered stale.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionFieldCache:
    """Cache for field availability across session read operations.

    This cache stores the mapping of parameter UID -> list of available fields,
    avoiding repeated calls to list_fields_by_uid() which can be expensive
    for large parameter collections.

    The cache is automatically invalidated when mutating operations occur
    (compute, add, erase, merge). It can also be incrementally updated
    after compute() to avoid full rescans.

    Example:
        cache = SessionFieldCache()

        # First export_metrics() populates the cache
        fields_by_uid = view.list_fields_by_uid()
        cache.set(fields_by_uid)

        # Second export_metrics() uses cached data
        if cache.is_valid:
            fields_by_uid = cache.get()

        # compute.apply() invalidates the cache
        cache.invalidate()
    """

    _generation: int = field(default=0, repr=False)
    _fields_by_uid: dict[str, list[str]] | None = field(default=None, repr=False)

    def get(self) -> dict[str, list[str]] | None:
        """Return cached field mapping or None if not populated.

        Returns:
            Dictionary mapping parameter UID to list of field names,
            or None if cache is not populated.
        """
        return self._fields_by_uid

    def set(self, fields_by_uid: dict[str, list[str]]) -> None:
        """Store field mapping in cache.

        Args:
            fields_by_uid: Mapping of parameter UID to list of available fields.
        """
        self._fields_by_uid = fields_by_uid
        logger.debug(
            "Field cache populated with %d entries at generation %d",
            len(fields_by_uid),
            self._generation,
        )

    def update(self, fields_by_uid: dict[str, list[str]]) -> None:
        """Merge additional field mappings into the existing cache.

        Args:
            fields_by_uid: Mapping of parameter UID to list of available
                fields to merge into the current cache contents.
        """
        self._fields_by_uid.update(**fields_by_uid)

    def invalidate(self) -> None:
        """Invalidate cache (called after mutations).

        Increments the generation counter and clears cached data.
        This should be called after any operation that modifies storage:
        compute.apply(), models.add(), models.erase(), results.erase(),
        utils.merge_other_session().
        """
        self._fields_by_uid = None
        self._generation += 1
        logger.debug("Field cache invalidated, new generation: %d", self._generation)

    def add_computed_fields(
        self,
        affected_uids: Iterable[str],
        new_fields: Iterable[str],
    ) -> None:
        """Incrementally update cache after compute.apply() operation.

        Instead of full invalidation, this method adds newly computed fields
        to existing cache entries. This is an optimization for cases where
        we know exactly which fields were added.

        If cache is not populated, this is a no-op.

        Args:
            affected_uids: UIDs of parameters that were computed.
            new_fields: Field names that were produced by the computation.
        """
        if self._fields_by_uid is None:
            logger.debug("Skipping incremental update: cache not populated")
            return

        new_fields_set = set(new_fields)
        updated_count = 0

        for uid in affected_uids:
            if uid in self._fields_by_uid:
                existing = set(self._fields_by_uid[uid])
                if not new_fields_set.issubset(existing):
                    existing.update(new_fields_set)
                    self._fields_by_uid[uid] = sorted(existing)
                    updated_count += 1

        logger.debug(
            "Incrementally updated %d cache entries with fields: %s",
            updated_count,
            ", ".join(sorted(new_fields_set)),
        )

    def remove_fields_by_uids(
        self, uids: Iterable[str], fields_to_remove: Iterable[str]
    ) -> None:
        """Remove fields from cache entries after results.erase().

        Instead of full invalidation, this method removes specified fields
        from specified cache entries. This is an optimization for results.erase().

        If cache is not populated, this is a no-op; UIDs absent from the cache
        are skipped, matching add_computed_fields and remove_uids.

        Args:
            uids: UIDs of cache entries to remove fields from.
            fields_to_remove: Field names that were erased.
        """
        if self._fields_by_uid is None:
            logger.debug("Skipping field removal: cache not populated")
            return

        fields_set = set(fields_to_remove)
        updated_count = 0

        for uid in uids:
            if uid not in self._fields_by_uid:
                # A scoped read may have populated the cache with only a
                # subset of uids; an absent uid has nothing to update.
                continue
            cached = self._fields_by_uid[uid]
            remaining = [f for f in cached if f not in fields_set]
            if len(remaining) != len(cached):
                self._fields_by_uid[uid] = remaining
                updated_count += 1

        logger.debug(
            "Removed fields from %d cache entries: %s",
            updated_count,
            ", ".join(sorted(fields_set)),
        )

    def remove_uids(self, uids_to_remove: Iterable[str]) -> None:
        """Remove UIDs from cache after models.erase().

        Instead of full invalidation, this method removes specified UIDs
        from the cache. This is an optimization for models.erase().

        If cache is not populated, this is a no-op.

        Args:
            uids_to_remove: UIDs that were erased from storage.
        """
        if self._fields_by_uid is None:
            logger.debug("Skipping UID removal: cache not populated")
            return

        removed_count = 0
        for uid in uids_to_remove:
            if uid in self._fields_by_uid:
                del self._fields_by_uid[uid]
                removed_count += 1

        logger.debug("Removed %d UIDs from cache", removed_count)

    @property
    def is_valid(self) -> bool:
        """Check if cache contains valid data.

        Returns:
            True if cache is populated and ready for use.
        """
        return self._fields_by_uid is not None

    @property
    def generation(self) -> int:
        """Current generation counter.

        Increments on each invalidation. Useful for debugging.

        Returns:
            Current generation number.
        """
        return self._generation

    def clear(self) -> None:
        """Clear cache completely and reset generation.

        Use this for full session reset, not for normal invalidation.
        """
        self._fields_by_uid = None
        self._generation = 0
        logger.debug("Field cache cleared and reset")
