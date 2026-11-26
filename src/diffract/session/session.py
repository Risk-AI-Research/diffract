"""Main Session API for the diffract package.

This module implements the primary user interface for neural network parameter
analysis workflows, providing a high-level API for model management, kernel
execution, and result retrieval through dependency injection.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Self

from diffract.containers import MainContainer, create_main_container
from diffract.core.data.metadata.migrations import (
    SchemaVersionError,
    StoreIdentityError,
)
from diffract.core.data.nn.aggregates.view import AggregateView
from diffract.core.data.nn.params.interface import IParameterView
from diffract.core.data.nn.params.schema import ParameterType, StepSelection
from diffract.session.errors import IncompatibleStoreError, UnrecognizedStoreError
from diffract.session.field_cache import SessionFieldCache
from diffract.session.utils import filter_aggregate_view, filter_parameter_view

logger = logging.getLogger(__name__)


def _aggregate_scope(
    source: Callable[[], AggregateView],
    parameter_names: list[str] | None,
    model_ids: list[str] | None,
    step_members: list[str] | None,
) -> AggregateView:
    view = filter_aggregate_view(
        source(), parameter_names=parameter_names, model_ids=model_ids
    )
    return view if step_members is None else view.within_parameters(step_members)


class Session:
    """Main session class for neural network parameter analysis.

    Provides a high-level interface for adding models, applying computational
    kernels, and retrieving results. Manages the complete lifecycle of parameter
    analysis workflows through dependency injection and modular components.

    Example:
        >>> session = Session()
        >>> session.models.add(my_model, model_id="bert-base")
        >>> session.compute.apply("frob_norm", "stable_rank")
        >>> results = session.results.export_metrics(
        ...     "frob_norm", "stable_rank", export_format="pandas"
        ... )
    """

    def __init__(
        self,
        profile: str | None = "ram",
        config_path: str | Path | None = None,
        container: MainContainer | None = None,
        other_session: Session | None = None,
    ) -> None:
        """Initialize a new session.

        Args:
            profile: Built-in profile name for quick setup:
                - "ram": RAM storage, no persistence (fast experiments)
                - "local": SQLite storage in .diffract/ (persistent, simple)
                - "hybrid": SQLite + HDF5 (persistent, optimized for large arrays)
            config_path: Path to configuration file (YAML/JSON/INI). Takes
                priority over profile if both are provided.
            container: Pre-configured MainContainer instance. If None,
                creates new container using config_path or profile.
            other_session: Another Session instance to copy configuration from.
        """
        # For namespaces
        self._field_cache = SessionFieldCache()

        if container is None:
            self.__container = create_main_container(config_path, profile=profile)
        else:
            self.__container = container

        self.__context_depth = 0

        with self:
            logger.debug("Loading parameters...")
            self.__param_repo = self.__container.nn.parameter_repository()
            self.__agg_repo = self.__container.nn.aggregate_repository()
            logger.info(
                "Session initialized with %d existing parameters, %d aggregates",
                len(self.__param_repo),
                len(self.__agg_repo),
            )

        self._bind_namespaces(self)

        if other_session is not None:
            self.utils.merge_other_session(other_session, verify=False)

    def __enter__(self) -> Self:
        """Enter session context and initialize resources.

        Raises:
            IncompatibleStoreError: If the persistent metadata index has an
                incompatible or incomplete schema. Resources initialized
                before the failure are released first.
            UnrecognizedStoreError: If the configured metadata path is not a
                diffract store; the file is left unchanged.
        """
        self.__context_depth += 1
        if self.__context_depth == 1:
            logger.debug("Init resources called")
            try:
                self.__container.init_resources()
                self.__container.nn.parameter_repository().refresh_names()
                self.__container.nn.aggregate_repository().refresh_names()
            except BaseException as exc:
                # A resource init failure (e.g. an incompatible store version)
                # leaves earlier resources open; release them and undo the
                # depth bump so the failure is clean and retryable.
                self.__container.shutdown_resources()
                self.__context_depth -= 1
                if isinstance(exc, StoreIdentityError):
                    raise UnrecognizedStoreError(str(exc)) from exc
                if isinstance(exc, SchemaVersionError):
                    raise IncompatibleStoreError(str(exc)) from exc
                raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit session context and shutdown resources."""
        self.__context_depth -= 1
        if self.__context_depth == 0:
            logger.debug("Shutdown resources called")
            self.__container.shutdown_resources()

    def _bind_namespaces(self, target: Session | SessionContext) -> None:
        """Bind each namespace and its factories to the owning session."""
        from diffract.session.namespaces import (
            ComputeNamespace,
            ModelsNamespace,
            ResultsNamespace,
            UtilsNamespace,
            VizNamespace,
        )
        from diffract.session.namespaces.models.parameters import ParametersNamespace

        container = self.__container
        parallel = container.parallel_singleton.thread_pool_context
        registry = container.compute.kernel_registry()
        target.models = ModelsNamespace(
            target,
            parameter_repository=self.__param_repo,
            aggregate_repository=self.__agg_repo,
            extractor_factory=container.nn.extractor_factory,
            parameters=ParametersNamespace(
                target,
                parameter_repository=self.__param_repo,
                parallel_context_factory=parallel,
            ),
        )
        target.compute = ComputeNamespace(
            target,
            parameter_repository=self.__param_repo,
            aggregate_repository=self.__agg_repo,
            kernel_registry=registry,
            kernel_executor_factory=container.compute.kernel_executor,
        )
        target.results = ResultsNamespace(
            target,
            parameter_repository=self.__param_repo,
            aggregate_repository=self.__agg_repo,
            kernel_registry=registry,
            kernel_executor_factory=container.compute.kernel_executor,
            parallel_context_factory=parallel,
            results_exporter_factory=container.export.result_exporter,
            default_export_format=container.export.config.default_export_format(),
        )
        target.utils = UtilsNamespace(
            target,
            parameter_repository=self.__param_repo,
            aggregate_repository=self.__agg_repo,
            parallel_context_factory=parallel,
        )
        target.viz = VizNamespace(target)

    def filter(
        self,
        param_ids: list[str] | None = None,
        param_names: list[str] | None = None,
        param_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
        steps: StepSelection | None = None,
    ) -> SessionContext:
        """Create a filtered view of the session's parameters and aggregates.

        Args:
            param_ids: Parameter IDs to keep, or None for no ID filtering.
            param_names: Parameter names to keep, or None for no name filtering.
            param_types: Parameter types to keep, or None for no type filtering.
            model_ids: Model IDs to keep, or None for no model filtering.
            steps: Checkpoint numbers, or "final" for each model's maximum step.

        Returns:
            A SessionContext scoped to the filtered parameter and aggregate views.
        """
        with self:
            param_view = self.__param_repo.create_view()
            filtered_param_view = filter_parameter_view(
                param_view,
                parameter_ids=param_ids,
                parameter_names=param_names,
                parameter_types=param_types,
                steps=steps,
                model_ids=model_ids,
            )

            return SessionContext(
                self,
                filtered_param_view,
                partial(
                    _aggregate_scope,
                    self.__agg_repo.create_view,
                    param_names[:] if param_names else None,
                    model_ids[:] if model_ids else None,
                    filtered_param_view.list_uids() if steps is not None else None,
                ),
            )


class SessionContext:
    """Filtered, scoped view over a session's parameters and aggregates.

    Wraps a parent Session together with pre-filtered parameter and aggregate
    views, exposing the same namespaces (models, compute, results, viz, utils)
    restricted to the filtered scope. Supports further chained filtering.
    """

    def __init__(
        self,
        session: Session,
        param_filter_context: IParameterView,
        agg_filter_context: AggregateView | Callable[[], AggregateView],
    ) -> None:
        """Initialize a filtered session context.

        Args:
            session: The parent Session this context is derived from.
            param_filter_context: Pre-filtered parameter view for this scope.
            agg_filter_context: Aggregate view or a callable that resolves the scope.
        """
        self.__session = session

        # For namespaces
        self._param_filter_context = param_filter_context
        self.__aggregate_source = agg_filter_context
        self._field_cache = session._field_cache

        session._bind_namespaces(self)

    def __enter__(self) -> Self:
        """Enter the underlying session context."""
        self.__session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the underlying session context."""
        self.__session.__exit__(exc_type, exc_val, exc_tb)

    def _current_aggregates(self) -> AggregateView:
        source = self.__aggregate_source
        return source() if callable(source) else source

    @property
    def _agg_filter_context(self) -> AggregateView:
        return self._current_aggregates()

    def filter(
        self,
        param_ids: list[str] | None = None,
        param_names: list[str] | None = None,
        param_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
        steps: StepSelection | None = None,
    ) -> Self:
        """Further filter this context's parameters and aggregates.

        Args:
            param_ids: Parameter IDs to keep, or None for no ID filtering.
            param_names: Parameter names to keep, or None for no name filtering.
            param_types: Parameter types to keep, or None for no type filtering.
            model_ids: Model IDs to keep, or None for no model filtering.
            steps: Checkpoint numbers, or "final" for each model's maximum step.

        Returns:
            A new SessionContext scoped to the further-filtered views.
        """
        with self:
            filtered_param_view = filter_parameter_view(
                self._param_filter_context,
                parameter_ids=param_ids,
                parameter_names=param_names,
                parameter_types=param_types,
                steps=steps,
                model_ids=model_ids,
            )

            return SessionContext(
                self.__session,
                filtered_param_view,
                partial(
                    _aggregate_scope,
                    self._current_aggregates,
                    param_names[:] if param_names else None,
                    model_ids[:] if model_ids else None,
                    filtered_param_view.list_uids() if steps is not None else None,
                ),
            )
