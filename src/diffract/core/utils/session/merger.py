"""Session merge utilities for transferring parameters and aggregates between sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from diffract.core.constants import (
    PROGRESS_BAR_DELAY_SEC,
    PROGRESS_BAR_MIN_ITEMS,
)
from diffract.core.data.nn.params.proxy import ParameterDataProxy

if TYPE_CHECKING:
    from diffract.core.compute.parallel import ParallelContext
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import (
        IParameterProxy,
        IParameterRepository,
        IParameterView,
    )

logger = logging.getLogger(__name__)


@dataclass
class MergeTargetState:
    """State of target session for conflict detection during merge."""

    models_mapping: dict[str, set[str]] = field(default_factory=dict)
    parameters_uids_mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    parameters_fields_mapping: dict[tuple[str, str], list[str]] = field(
        default_factory=dict
    )


class SessionMerger:
    """Utility for merging parameters between sessions.

    Handles chunked transfer of parameters with prefetching and conflict detection.

    Example:
        >>> merger = SessionMerger(parallel=parallel_context)
        >>> merger.merge(
        ...     source=source_session._get_view(),
        ...     source_fields_by_uid=source.list_fields_by_uid(),
        ...     target_repository=target_session._parameter_repository,
        ...     target_state=target_session._build_merge_target_state(),
        ...     fields=["metric", "loss"],
        ... )
    """

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
        target_state: MergeTargetState | None = None,
        fields: list[str] | None = None,
        read_budget_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Merge parameters from source view into target repository.

        Args:
            source: Source parameter view to merge from.
            source_fields_by_uid: Precomputed mapping of uid -> available fields.
            target_repository: Target repository to merge into.
            target_state: Current state of target for conflict detection.
                If None, no conflict checking is performed.
            fields: Specific fields to merge. If None, merges all fields.
            read_budget_bytes: Maximum bytes to read per chunk.
        """
        if not source:
            return

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
                self._process_chunk(
                    chunk=chunk,
                    source_fields_by_uid=source_fields_by_uid,
                    target_repository=target_repository,
                    target_state=target_state,
                    field_allowlist=field_allowlist,
                    required_fields_by_uid=required_fields_by_uid,
                    pbar=pbar,
                )

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
        target_state: MergeTargetState | None,
        field_allowlist: set[str] | None,
        required_fields_by_uid: dict[str, list[str]],
        pbar: Any,
    ) -> None:
        """Process a single chunk of parameters during merge."""
        chunk_required_fields_by_uid = {
            p.meta.uid: required_fields_by_uid.get(p.meta.uid, []) for p in chunk
        }
        chunk.prefetch_fields(
            fields_by_uid=chunk_required_fields_by_uid,
            parallel=self._parallel,
        )

        with target_repository:
            for pending_parameter in chunk:
                self._transfer_parameter(
                    pending_parameter=pending_parameter,
                    source_fields_by_uid=source_fields_by_uid,
                    target_repository=target_repository,
                    target_state=target_state,
                    field_allowlist=field_allowlist,
                )
                pbar.update(1)

    def _transfer_parameter(
        self,
        *,
        pending_parameter: IParameterProxy,
        source_fields_by_uid: dict[str, list[str]],
        target_repository: IParameterRepository,
        target_state: MergeTargetState | None,
        field_allowlist: set[str] | None,
    ) -> None:
        """Transfer a single parameter to the target repository."""
        meta = pending_parameter.meta
        create_new = True

        if (
            target_state is not None
            and meta.model_id in target_state.models_mapping
            and meta.name in target_state.models_mapping[meta.model_id]
        ):
            uid = target_state.parameters_uids_mapping[(meta.model_id, meta.name)]
            new_parameter = target_repository.get_proxy(uid)
            create_new = False

        if create_new:
            new_parameter = ParameterDataProxy.create_and_store(
                meta,
                repository=target_repository,
            )

        available_fields = source_fields_by_uid.get(meta.uid, [])
        for field_name in available_fields:
            if field_allowlist is not None and field_name not in field_allowlist:
                continue
            if (not create_new) and target_state is not None:
                existing_fields = target_state.parameters_fields_mapping.get(
                    (meta.model_id, meta.name), []
                )
                if field_name in existing_fields:
                    continue

            new_parameter.set_field(
                field_name,
                pending_parameter.get_field(field_name, auto_prefetch=True),
            )


class AggregateMerger:
    """Utility for merging aggregates between sessions.

    Handles transfer of aggregate entries with conflict detection.

    Example:
        >>> merger = AggregateMerger()
        >>> merger.merge(
        ...     source_aggregates=source_session._aggregate_repository.create_view(),
        ...     target_repository=target_session._aggregate_repository,
        ...     model_ids=["model_a"],
        ...     fields=["l_overlap"],
        ...     verify=True,
        ... )
    """

    def merge(
        self,
        *,
        source_aggregates: AggregateView,
        target_repository: AggregateRepository,
        model_ids: list[str] | None = None,
        fields: list[str] | None = None,
        verify: bool = True,
    ) -> None:
        """Merge aggregates from source view into target repository.

        Args:
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

        # Get existing aggregate UIDs in target
        target_uids = set(target_repository.create_view().list_uids())

        merged_count = 0
        skipped_count = 0

        with target_repository.storage_manager:
            for source_agg in filtered:
                if source_agg.meta.uid in target_uids:
                    if verify:
                        skipped_count += 1
                        continue

                # Create or get aggregate in target
                target_agg = target_repository.get_or_create(
                    field_name=source_agg.meta.field_name,
                    context_models=source_agg.meta.context_models,
                    context_params=source_agg.meta.context_params,
                )

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
