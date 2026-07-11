"""Generic data view for batch operations on filtered collections.

This module provides the base DataView class that can be specialized
for different data types (parameters, relations, etc.).
"""

from __future__ import annotations

import logging
import re
import types
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Self, TypeVar

from tqdm.auto import tqdm

from diffract.core.constants import (
    PROGRESS_BAR_DELAY_SEC,
    PROGRESS_BAR_MIN_ITEMS,
)
from diffract.core.parallel import ParallelContext, map_maybe_parallel
from diffract.core.utils.exceptions import format_exception_message

from .interface import EntityIndex, EntityUID, FieldName

if TYPE_CHECKING:
    from .interface import IDataRepository

TMetadata = TypeVar("TMetadata")
TProxy = TypeVar("TProxy")

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class DataView[TMetadata, TProxy]:
    """Generic view over a subset of entities owned by a repository.

    Provides numpy-like view semantics with support for filtering,
    iteration, prefetching, and batch operations. Subclasses can add
    domain-specific filtering methods.

    Type Parameters:
        TMetadata: The metadata type for entities in this view.
        TProxy: The proxy type for entities in this view.
    """

    def __init__(
        self,
        *,
        repository: IDataRepository[TMetadata, TProxy],
        uids: list[EntityUID] | None = None,
    ) -> None:
        self._repository = repository
        self._uids: list[EntityUID] | None = list(uids) if uids is not None else None
        self._sorted: bool = False

    def _ensure_uids(self) -> list[EntityUID]:
        """Ensure UIDs are loaded (lazy load from repository if None)."""
        if self._uids is None:
            self._uids = self._repository.list_uids()
            self._sorted = False
        return self._uids

    def _sort_in_place(self) -> None:
        """Sort UIDs for deterministic ordering."""
        uids = self._ensure_uids()
        uids.sort()
        self._sorted = True

    def __len__(self) -> int:
        """Return number of entities in the view."""
        return len(self._ensure_uids())

    def __iter__(self) -> Iterator[TProxy]:
        """Iterate entity proxies in view order."""
        if not self._sorted:
            self._sort_in_place()

        for uid in self._ensure_uids():
            yield self._repository.get_proxy(uid)

    def __getitem__(
        self,
        index: EntityIndex
        | slice
        | EntityUID
        | Sequence[EntityIndex]
        | Sequence[EntityUID],
    ) -> TProxy | Self:
        """Index into the view by position(s) or uid(s)."""
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        match index:
            case int():
                return self._repository.get_proxy(uids[index])
            case slice():
                return self.__class__(
                    repository=self._repository,
                    uids=uids[index],
                )
            case str():
                return self._repository.get_proxy(index)
            case _:
                if isinstance(index, Sequence) and index:
                    first = index[0]
                    if isinstance(first, int):
                        result_uids = [uids[i] for i in index]
                    elif isinstance(first, str):
                        result_uids = list(index)
                    else:
                        msg = f"Invalid index type: {type(first)}"
                        raise ValueError(msg)
                    return self.__class__(repository=self._repository, uids=result_uids)

                return self.__class__(repository=self._repository, uids=[])

    def __enter__(self) -> Self:
        """Enter a batch write context for this view's repository storage."""
        self._repository.storage_manager.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the batch write context for this view's repository storage."""
        self._repository.storage_manager.__exit__(exc_type, exc_val, exc_tb)

    def list_uids(self) -> list[EntityUID]:
        """Return the list of entity UIDs contained in this view."""
        if not self._sorted:
            self._sort_in_place()
        return list(self._ensure_uids())

    def list_fields_by_uid(
        self,
        *,
        parallel: ParallelContext | None = None,
    ) -> dict[EntityUID, list[FieldName]]:
        """Return mapping uid -> available fields for entities in this view."""
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        if not uids:
            return {}

        def _safe_list_fields(
            uid: EntityUID,
        ) -> tuple[EntityUID, list[FieldName], str | None]:
            try:
                return uid, self._repository.get_proxy(uid).list_fields(), None
            except Exception as e:  # noqa: BLE001
                return uid, [], format_exception_message(e)

        dict_items = map_maybe_parallel(
            uids,
            _safe_list_fields,
            parallel=parallel,
        )

        total = len(uids)
        field_map: dict[EntityUID, list[FieldName]] = {}
        failures: list[tuple[EntityUID, str]] = []
        for uid, fields, error_type in tqdm(
            dict_items,
            desc="Listing fields...",
            delay=PROGRESS_BAR_DELAY_SEC,
            total=total,
            disable=total < PROGRESS_BAR_MIN_ITEMS,
        ):
            field_map[uid] = fields
            if error_type is not None:
                failures.append((uid, error_type))

        if failures:
            preview = ", ".join(
                f"{uid}({err})" for uid, err in failures[:_ERROR_PREVIEW_LIMIT]
            )
            suffix = "..." if len(failures) > _ERROR_PREVIEW_LIMIT else ""
            logger.warning(
                "Failed to list fields for %d entities: %s%s",
                len(failures),
                preview,
                suffix,
            )

        return field_map

    def prefetch_fields(
        self,
        *,
        fields_by_uid: dict[EntityUID, list[FieldName]] | None = None,
        fields: list[FieldName] | None = None,
        verify_prefetch: bool = False,
        parallel: ParallelContext | None = None,
    ) -> bool:
        """Prefetch specified fields for entities in this view into cache."""
        if fields_by_uid is not None and fields is not None:
            msg = "fields_by_uid and fields are mutually exclusive."
            raise ValueError(msg)

        if fields is None and fields_by_uid is None:
            return True

        if fields is not None and not fields:
            return True

        if fields_by_uid is not None and not fields_by_uid:
            return True

        if not self._sorted:
            self._sort_in_place()

        if fields is not None:
            fields_by_uid = dict.fromkeys(self._ensure_uids(), fields)

        if fields_by_uid is None:
            return True

        def _prefetch_one(
            uid: EntityUID,
        ) -> tuple[EntityUID, bool, str | None]:
            uid_fields = fields_by_uid.get(uid) if fields_by_uid else []
            if not uid_fields:
                return uid, True, None
            try:
                proxy = self._repository.get_proxy(uid)
                for field_name in uid_fields:
                    if not proxy.try_prefetch_field(field_name):
                        return uid, False, f"Missing field: {field_name}"
            except Exception as e:  # noqa: BLE001
                return uid, False, format_exception_message(e)
            else:
                return uid, True, None

        results = map_maybe_parallel(
            list(fields_by_uid.keys()),
            _prefetch_one,
            parallel=parallel,
        )

        total = len(fields_by_uid)
        all_success = True
        failures: list[tuple[EntityUID, str]] = []

        for uid, success, error_msg in tqdm(
            results,
            desc="Prefetching fields...",
            delay=PROGRESS_BAR_DELAY_SEC,
            total=total,
            disable=total < PROGRESS_BAR_MIN_ITEMS,
        ):
            if not success:
                all_success = False
                if error_msg:
                    failures.append((uid, error_msg))

        if failures:
            preview = ", ".join(
                f"{uid}({err})" for uid, err in failures[:_ERROR_PREVIEW_LIMIT]
            )
            suffix = "..." if len(failures) > _ERROR_PREVIEW_LIMIT else ""
            logger.warning(
                "Failed to prefetch fields for %d entities: %s%s",
                len(failures),
                preview,
                suffix,
            )

        if verify_prefetch and all_success:
            for uid in fields_by_uid:
                proxy = self._repository.get_proxy(uid)
                for field_name in fields_by_uid[uid]:
                    if not proxy.is_field_prefetched(field_name):
                        return False

        return all_success

    def erase_fields(self, *fields: FieldName) -> None:
        """Remove specified fields from entities in this view."""
        if not fields:
            return

        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        total = len(uids)
        for uid in tqdm(
            uids,
            desc="Erasing fields...",
            delay=PROGRESS_BAR_DELAY_SEC,
            total=total,
            disable=total < PROGRESS_BAR_MIN_ITEMS,
        ):
            proxy = self._repository.get_proxy(uid)
            for field_name in fields:
                if proxy.has_field(field_name):
                    proxy.erase_field(field_name)

    def erase_fields_with_regexp(self, *patterns: str) -> None:
        """Remove fields matching regex patterns from entities in this view."""
        if not patterns:
            return

        if not self._sorted:
            self._sort_in_place()

        compiled = [re.compile(p) for p in patterns]

        uids = self._ensure_uids()
        total = len(uids)
        for uid in tqdm(
            uids,
            desc="Erasing fields by pattern...",
            delay=PROGRESS_BAR_DELAY_SEC,
            total=total,
            disable=total < PROGRESS_BAR_MIN_ITEMS,
        ):
            proxy = self._repository.get_proxy(uid)
            for field_name in proxy.list_fields():
                if any(p.fullmatch(field_name) for p in compiled):
                    proxy.erase_field(field_name)

    def filter_by_fields(
        self,
        *fields: FieldName,
        inverse_mask: bool = False,
        parallel: ParallelContext | None = None,
    ) -> Self:
        """Filter entities by the presence of one or more fields."""
        if not self._sorted:
            self._sort_in_place()

        uids = self._ensure_uids()
        need_fields = tuple(fields)
        if not need_fields:
            return self.__class__(
                repository=self._repository,
                uids=uids if not inverse_mask else [],
            )

        def _filter(uid: EntityUID) -> EntityUID | None:
            proxy = self._repository.get_proxy(uid)
            for field in need_fields:
                if not proxy.has_field(field) ^ inverse_mask:
                    return None
            return uid

        filtered_uids = map_maybe_parallel(uids, _filter, parallel=parallel)

        total = len(uids)
        out = [
            uid
            for uid in tqdm(
                filtered_uids,
                desc="Filtering by fields...",
                delay=PROGRESS_BAR_DELAY_SEC,
                total=total,
                disable=total < PROGRESS_BAR_MIN_ITEMS,
            )
            if uid is not None
        ]

        return self.__class__(repository=self._repository, uids=out)

    def clear(self, erase: bool = False) -> None:
        """Clear the view.

        Args:
            erase: If True, also erase data from storage and cache.
        """
        if erase:
            for uid in self._ensure_uids():
                proxy = self._repository.get_proxy(uid)
                for field_name in proxy.list_fields():
                    proxy.erase_field(field_name)
        if self._uids is not None:
            self._uids.clear()
        else:
            self._uids = []
        self._sorted = True
