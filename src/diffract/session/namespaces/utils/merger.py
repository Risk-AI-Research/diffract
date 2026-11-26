from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from diffract.core.constants import (
    PROGRESS_BAR_DELAY_SEC,
    PROGRESS_BAR_MIN_ITEMS,
)

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import (
        IParameterProxy,
        IParameterRepository,
        IParameterView,
    )
    from diffract.core.parallel import ParallelContext

logger = logging.getLogger(__name__)


class SessionMerger:
    """Transfer parameters in chunks and resolve conflicts at the target."""

    def __init__(self, parallel: ParallelContext) -> None:
        """Initialize the merger.

        Args:
            parallel: Parallel context for concurrent operations.
        """
        self._parallel = parallel

    def merge(
        self,
        *,
        source: IParameterView,
        source_fields_by_uid: dict[str, list[str]],
        target_repository: IParameterRepository,
        verify: bool = False,
        fields: list[str] | None = None,
        read_budget_bytes: int = 512 * 1024 * 1024,
    ) -> dict[str, str]:
        """Merge parameters from source view into target repository.

        Args:
            source: Source parameter view to merge from.
            source_fields_by_uid: Precomputed mapping of uid -> available fields.
            target_repository: Target repository to merge into.
            verify: Preserve existing field values when a coordinate matches.
            fields: Specific fields to merge. If None, merges all fields.
            read_budget_bytes: Maximum bytes to read per chunk.
        """
        mapping: dict[str, str] = {}
        if not source:
            return mapping

        labels = source.step_labels()
        total = len(source)
        field_allowlist = set(fields) if fields else None

        required_fields_by_uid = self._compute_required_fields(
            source, source_fields_by_uid, field_allowlist
        )

        with tqdm(
            total=total,
            desc="Parameters migration...",
            delay=PROGRESS_BAR_DELAY_SEC,
            disable=total < PROGRESS_BAR_MIN_ITEMS,
        ) as pbar:
            for chunk in source.iter_chunks_by_read_budget(
                budget_bytes=read_budget_bytes,
                required_fields_by_uid=required_fields_by_uid,
                parallel=self._parallel,
            ):
                mapping.update(
                    self._process_chunk(
                        chunk=chunk,
                        source_fields_by_uid=source_fields_by_uid,
                        target_repository=target_repository,
                        verify=verify,
                        labels=labels,
                        field_allowlist=field_allowlist,
                        required_fields_by_uid=required_fields_by_uid,
                        pbar=pbar,
                    )
                )
        return mapping

    def _compute_required_fields(
        self,
        source: IParameterView,
        source_fields_by_uid: dict[str, list[str]],
        field_allowlist: set[str] | None,
    ) -> dict[str, list[str]]:
        """Compute required fields per parameter for prefetching."""
        required_fields_by_uid: dict[str, list[str]] = {}

        for param in source:
            uid = param.meta.uid
            available = source_fields_by_uid.get(uid, [])

            required: set[str] = set()
            for f in available:
                if field_allowlist is not None and f not in field_allowlist:
                    continue
                required.add(f)

            required_fields_by_uid[uid] = sorted(required)

        return required_fields_by_uid

    def _process_chunk(
        self,
        *,
        chunk: IParameterView,
        source_fields_by_uid: dict[str, list[str]],
        target_repository: IParameterRepository,
        verify: bool,
        labels: dict[tuple[str, int], str],
        field_allowlist: set[str] | None,
        required_fields_by_uid: dict[str, list[str]],
        pbar: Any,
    ) -> dict[str, str]:
        """Process a single chunk of parameters during merge."""
        chunk_required_fields_by_uid = {
            p.meta.uid: required_fields_by_uid.get(p.meta.uid, []) for p in chunk
        }
        chunk.prefetch_fields(
            fields_by_uid=chunk_required_fields_by_uid,
            parallel=self._parallel,
        )

        mapping = {}
        with target_repository:
            for pending_parameter in chunk:
                mapping[pending_parameter.meta.uid] = self._transfer_parameter(
                    pending_parameter=pending_parameter,
                    source_fields_by_uid=source_fields_by_uid,
                    target_repository=target_repository,
                    verify=verify,
                    labels=labels,
                    field_allowlist=field_allowlist,
                )
                pbar.update(1)
        return mapping

    def _transfer_parameter(
        self,
        *,
        pending_parameter: IParameterProxy,
        source_fields_by_uid: dict[str, list[str]],
        target_repository: IParameterRepository,
        verify: bool,
        labels: dict[tuple[str, int], str],
        field_allowlist: set[str] | None,
    ) -> str:
        """Transfer a single parameter to the target repository."""
        meta = pending_parameter.meta
        new_parameter = target_repository.merge_target(meta, overwrite=not verify)
        label = labels.pop((meta.instance_uid, meta.step), None)
        if label is not None:
            target_repository.instances.set_step_label(
                new_parameter.meta.instance_uid, meta.step, label, overwrite=not verify
            )
        existing_fields = set(new_parameter.list_fields()) if verify else set()

        available_fields = source_fields_by_uid.get(meta.uid, [])
        for field_name in available_fields:
            if field_allowlist is not None and field_name not in field_allowlist:
                continue
            if field_name in existing_fields:
                continue

            new_parameter.set_field(
                field_name,
                pending_parameter.get_field(field_name, auto_prefetch=True),
            )
        return new_parameter.meta.uid


class AggregateMerger:
    """Transfer aggregate values using the source-to-target parameter mapping."""

    def merge(
        self,
        *,
        source_aggregates: AggregateView,
        target_repository: AggregateRepository,
        member_mapping: dict[str, str],
        model_ids: list[str] | None = None,
        fields: list[str] | None = None,
        verify: bool = True,
    ) -> None:
        """Merge aggregates from source view into target repository.

        Args:
            member_mapping: Source-to-target UIDs for the transferred parameters.
            source_aggregates: Source aggregate view to merge from.
            target_repository: Target repository to merge into.
            model_ids: Filter source aggregates by context_models.
            fields: Filter source aggregates by field_name.
            verify: If True, skip existing aggregates.
        """
        filtered = source_aggregates
        logger.debug("Merging aggregates: source has %d aggregates", len(filtered))

        # Filter by model_ids if specified
        if model_ids:
            filtered = filtered.filter_by_context_models(*model_ids)

        # Filter by fields if specified
        if fields:
            filtered = filtered.filter_by_field_name(*fields)

        if not filtered:
            logger.debug("No source aggregates to merge after filtering")
            return

        # Prefetch values from source
        filtered.prefetch_fields(fields=["value"], parallel=None)

        merged_count = 0
        skipped_count = 0

        with target_repository.storage_manager:
            for source_agg in filtered:
                source_members = source_agg.meta.member_uids
                if any(member not in member_mapping for member in source_members):
                    skipped_count += 1
                    continue
                members = tuple(member_mapping[member] for member in source_members)
                target_agg = target_repository.get_or_create(
                    source_agg.meta.field_name,
                    member_uids=members,
                    directed=source_agg.meta.directed,
                )
                if verify and target_agg.has_field("value"):
                    skipped_count += 1
                    continue

                # Copy value if exists
                if source_agg.has_field("value"):
                    target_agg.set_field("value", source_agg.get_field("value"))
                    merged_count += 1

        if merged_count > 0 or skipped_count > 0:
            logger.debug(
                "Aggregates merge: %d copied, %d skipped (existing)",
                merged_count,
                skipped_count,
            )
