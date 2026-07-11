"""Main Session API for the diffract package.

This module implements the primary user interface for neural network parameter
analysis workflows, providing a high-level API for model management, kernel
execution, and result retrieval through dependency injection.
"""

from __future__ import annotations

import logging
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Self

from dependency_injector.wiring import Provide, inject

from diffract.containers import MainContainer, create_main_container
from diffract.core.data.nn.aggregates.repository import AggregateRepository
from diffract.core.data.nn.aggregates.view import AggregateView
from diffract.core.data.nn.params.interface import IParameterRepository, IParameterView
from diffract.core.data.nn.params.schema import ParameterType
from diffract.session.field_cache import SessionFieldCache
from diffract.session.utils import filter_aggregate_view, filter_parameter_view

logger = logging.getLogger(__name__)


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
        from diffract.session.namespaces import (
            ComputeNamespace,
            ModelsNamespace,
            ResultsNamespace,
            UtilsNamespace,
            VizNamespace,
        )

        # For namespaces
        self._field_cache = SessionFieldCache()

        if container is None:
            self.__container = create_main_container(config_path, profile=profile)
        else:
            self.__container = container

        self.__context_depth = 0

        with self:
            logger.debug("Loading parameters...")
            self.__init_repos()
            logger.info(
                "Session initialized with %d existing parameters, %d aggregates",
                len(self.__param_repo),
                len(self.__agg_repo),
            )

        self.models = ModelsNamespace(self)
        self.compute = ComputeNamespace(self)
        self.results = ResultsNamespace(self)
        self.viz = VizNamespace(self)
        self.utils = UtilsNamespace(self)

        if other_session is not None:
            self.utils.merge_other_session(other_session, verify=False)

    def __enter__(self) -> Self | nullcontext[Self]:
        """Enter session context and initialize resources."""
        self.__context_depth += 1
        if self.__context_depth == 1:
            logger.debug("Init resources called")
            self.__container.init_resources()
            return self
        return nullcontext()

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

    @inject
    def __init_repos(
        self,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        aggregate_repository: AggregateRepository = Provide["nn.aggregate_repository"],
    ) -> None:
        """Inject dependencies from the container."""
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository

    def filter(
        self,
        param_ids: list[str] | None = None,
        param_names: list[str] | None = None,
        param_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
    ) -> SessionContext:
        """Create a filtered view of the session's parameters and aggregates.

        Args:
            param_ids: Parameter IDs to keep, or None for no ID filtering.
            param_names: Parameter names to keep, or None for no name filtering.
            param_types: Parameter types to keep, or None for no type filtering.
            model_ids: Model IDs to keep, or None for no model filtering.

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
                model_ids=model_ids,
            )

            agg_view = self.__agg_repo.create_view()
            filtered_agg_view = filter_aggregate_view(
                agg_view,
                parameter_names=param_names,
                model_ids=model_ids,
            )

            return SessionContext(self, filtered_param_view, filtered_agg_view)


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
        agg_filter_context: AggregateView,
    ) -> None:
        """Initialize a filtered session context.

        Args:
            session: The parent Session this context is derived from.
            param_filter_context: Pre-filtered parameter view for this scope.
            agg_filter_context: Pre-filtered aggregate view for this scope.
        """
        from diffract.session.namespaces import (
            ComputeNamespace,
            ModelsNamespace,
            ResultsNamespace,
            UtilsNamespace,
            VizNamespace,
        )

        self.__session = session

        # For namespaces
        self._param_filter_context = param_filter_context
        self._agg_filter_context = agg_filter_context
        self._field_cache = session._field_cache

        self.models = ModelsNamespace(self)
        self.compute = ComputeNamespace(self)
        self.results = ResultsNamespace(self)
        self.viz = VizNamespace(self)
        self.utils = UtilsNamespace(self)

    def __enter__(self) -> Self | nullcontext[Self]:
        """Enter the underlying session context."""
        return self.__session.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the underlying session context."""
        self.__session.__exit__(exc_type, exc_val, exc_tb)

    def filter(
        self,
        param_ids: list[str] | None = None,
        param_names: list[str] | None = None,
        param_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
    ) -> Self:
        """Further filter this context's parameters and aggregates.

        Args:
            param_ids: Parameter IDs to keep, or None for no ID filtering.
            param_names: Parameter names to keep, or None for no name filtering.
            param_types: Parameter types to keep, or None for no type filtering.
            model_ids: Model IDs to keep, or None for no model filtering.

        Returns:
            A new SessionContext scoped to the further-filtered views.
        """
        with self:
            filtered_param_view = filter_parameter_view(
                self._param_filter_context,
                parameter_ids=param_ids,
                parameter_names=param_names,
                parameter_types=param_types,
                model_ids=model_ids,
            )

            filtered_agg_view = filter_aggregate_view(
                self._agg_filter_context,
                parameter_names=param_names,
                model_ids=model_ids,
            )

            return SessionContext(
                self.__session, filtered_param_view, filtered_agg_view
            )
