"""Session utility namespace."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from diffract.session._identifiers import check_identifier
from diffract.session.namespaces.utils.merger import (
    AggregateMerger,
    MergeTargetState,
    SessionMerger,
)
from diffract.session.session import Session, SessionContext

if TYPE_CHECKING:
    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.aggregates.view import AggregateView
    from diffract.core.data.nn.params.interface import (
        IParameterRepository,
        IParameterView,
    )
    from diffract.core.parallel import ParallelContext

logger = logging.getLogger(__name__)


class UtilsNamespace:
    """Utility operations API for Session."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        aggregate_repository: AggregateRepository = Provide["nn.aggregate_repository"],
        parallel_context_factory: Callable[[], ParallelContext] = Provide[
            "parallel_singleton.thread_pool_context.provider"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository
        self.__parallel_context_factory = parallel_context_factory

    def merge_other_session(
        self,
        other_session_or_context: Session | SessionContext,
        fields: list[str] | None = None,
        *,
        verify: bool = True,
        read_budget_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Merge parameters from another session into this one.

        Copies parameters and their computed fields from another session.
        When verify is True, handles conflicts by skipping duplicate fields.

        Args:
            other_session_or_context: Source session to merge from.
            fields: Specific fields to merge. If None, merges all fields.
            verify: If True, check for conflicts and skip duplicates.
            read_budget_bytes: Maximum bytes to read during merge operation.
        """
        with self.__session_or_context:
            parallel = self.__parallel_context_factory()
            target_state: MergeTargetState | None = None
            if verify:
                target_state = MergeTargetState()

                models = self.__session_or_context.models
                for parameter_info in models.parameters.list(verbose=True):
                    model_id = parameter_info["model_id"]
                    name = parameter_info["name"]
                    uid = parameter_info["uid"]
                    available_fields = parameter_info["available_fields"]

                    if model_id not in target_state.models_mapping:
                        target_state.models_mapping[model_id] = set()

                    target_state.models_mapping[model_id].add(name)
                    target_state.parameters_fields_mapping[(model_id, name)] = (
                        available_fields
                    )
                    target_state.parameters_uids_mapping[(model_id, name)] = uid

            if isinstance(other_session_or_context, Session):
                source_context = other_session_or_context.filter()
            else:
                source_context = other_session_or_context

            source_params = source_context._param_filter_context
            source_aggs = source_context._agg_filter_context

            self._validate_source_identifiers(source_params, source_aggs)

            # Merge parameters if any exist
            if source_params:
                logger.info(
                    "Starting merge from session '%s' (%d parameters)",
                    other_session_or_context,
                    len(source_params),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Merge source models: %s",
                        other_session_or_context.models.list(),
                    )

                source_fields_by_uid = source_params.list_fields_by_uid(
                    parallel=parallel
                )

                merger = SessionMerger(parallel=parallel)
                merger.merge(
                    source=source_params,
                    source_fields_by_uid=source_fields_by_uid,
                    target_repository=self.__param_repo,
                    target_state=target_state,
                    fields=fields,
                    read_budget_bytes=read_budget_bytes,
                )
            else:
                logger.debug("No parameters to merge from source session")

            # Merge aggregates from source session (independent of parameters)
            if source_aggs:
                logger.info(
                    "Starting merge from session '%s' (%d aggregates)",
                    other_session_or_context,
                    len(source_aggs),
                )

                aggregate_merger = AggregateMerger()
                aggregate_merger.merge(
                    source_aggregates=source_aggs,
                    target_repository=self.__agg_repo,
                    fields=fields,
                    verify=verify,
                )
            else:
                logger.debug("No aggregates to merge from source session")

            # Invalidate field cache since parameters/fields may have been added
            self.__session_or_context._field_cache.invalidate()

    @staticmethod
    def _validate_source_identifiers(
        source_params: IParameterView,
        source_aggs: AggregateView,
    ) -> None:
        """Reject any exotic identity string the source carries before merging.

        Raises:
            InvalidIdentifierError: On the first unaccepted stored identifier.
        """
        for param in source_params:
            check_identifier(param.meta.model_id, kind="model id")
            check_identifier(param.meta.name, kind="parameter name")

        for aggregate in source_aggs:
            meta = aggregate.meta
            check_identifier(meta.field_name, kind="aggregate field name")
            for model_id in meta.context_models:
                check_identifier(model_id, kind="model id")
            for param_name in meta.context_params:
                check_identifier(param_name, kind="parameter name")
