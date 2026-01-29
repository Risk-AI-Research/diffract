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
from typing import TYPE_CHECKING, Any, Callable, Literal, Self

from dependency_injector.wiring import Provide, inject

from .containers import MainContainer, create_main_container
from .core.compute.parallel import ParallelContext
from .core.data.nn.aggregates.repository import AggregateRepository
from .core.data.nn.extractors.base import (
    ExtractorOverrides,
    ParameterOverrides,
)
from .core.data.nn.params.interface import IParameterRepository, IParameterView
from .core.data.nn.params.schema import ParameterType
from .core.utils.exceptions import format_exception_message
from .core.utils.session import (
    AggregateIngester,
    AggregateIngestionError,
    AggregateMerger,
    FieldIngester,
    FieldIngestionError,
    MergeTargetState,
    MetadataPatcher,
    MetadataPatchError,
    ResultsEraser,
    ResultsEraserError,
    SessionMerger,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from diffract.core.compute.execution.executor import KernelExecutor

    from .core.compute.registry import KernelRegistry
    from .core.data.nn.extractors.handlers.base import ParameterHandler
    from .core.data.nn.extractors.interface import IParameterExtractor

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base exception class for session-related errors."""


class ModelNotFoundError(SessionError):
    """Raised when a model is not found in the session."""


class ModelAlreadyExistsError(SessionError):
    """Raised when a model with the same id already exists."""


class KernelNotFoundError(SessionError):
    """Raised when a kernel is not found in the registry."""


class Session:
    """Main session class for neural network parameter analysis.

    Provides a high-level interface for adding models, applying computational
    kernels, and retrieving results. Manages the complete lifecycle of parameter
    analysis workflows through dependency injection and modular components.

    Example:
        >>> session = Session()
        >>> session.add(my_model, model_id="bert-base")
        >>> session.compute("frob_norm", "stable_rank")
        >>> results = session.get_results(
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
        self._active_context = False

        if container is None:
            self._container = create_main_container(config_path, profile=profile)
        else:
            self._container = container

        self._inject_dependencies()

        with self._own_context():
            # Load existing parameters from storage
            logger.debug("Loading parameters...")
            self._parameter_repository: IParameterRepository = (
                self._parameter_repository_factory()
            )
            self._aggregate_repository: AggregateRepository = (
                self._aggregate_repository_factory()
            )

            if other_session is not None:
                self.merge(other_session, verify=False)

            logger.info(
                "Session initialized with %d existing parameters, %d aggregates",
                len(self._parameter_repository),
                len(self._aggregate_repository),
            )

    @inject
    def _inject_dependencies(
        self,
        kernel_registry: KernelRegistry = Provide["compute.kernel_registry"],
        kernel_executor: Callable[[], KernelExecutor] = Provide[
            "compute.kernel_executor.provider"
        ],
        parallel_context: Callable[[], ParallelContext] = Provide[
            "parallel_singleton.thread_pool_context.provider"
        ],
        parameter_extractor_factory: Callable[..., IParameterExtractor] = Provide[
            "nn.extractor_factory.provider"
        ],
        parameter_repository_factory: Callable[[], IParameterRepository] = Provide[
            "nn.parameter_repository.provider"
        ],
        aggregate_repository_factory: Callable[[], AggregateRepository] = Provide[
            "nn.aggregate_repository.provider"
        ],
        results_exporter: Callable[
            [tuple[str, ...], IParameterView, str], Any
        ] = Provide["export.export_results.provider"],
    ) -> None:
        """Inject dependencies from the container."""
        self._kernel_registry = kernel_registry
        self._kernel_executor = kernel_executor
        self._parallel_context = parallel_context
        self._parameter_extractor_factory = parameter_extractor_factory
        self._parameter_repository_factory = parameter_repository_factory
        self._aggregate_repository_factory = aggregate_repository_factory
        self._results_exporter = results_exporter

    def __enter__(self) -> Self:
        """Enter session context and initialize resources."""
        self._active_context = True
        logger.debug("Init resources called")
        self._container.init_resources()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit session context and shutdown resources."""
        logger.debug("Shutdown resources called")
        self._container.shutdown_resources()
        self._active_context = False

    def _own_context(self) -> Session | nullcontext[Session]:
        """Return self as context manager or nullcontext if already active."""
        if self._active_context:
            return nullcontext(self)
        return self

    def add(
        self,
        model: Any,
        model_id: str | None = None,
        parameter_overrides: dict[str, ParameterOverrides] | None = None,
        custom_handlers: Iterable[ParameterHandler] | None = None,
    ) -> None:
        """Add a neural network model to the session.

        Extracts parameters from the provided model and stores them for analysis.
        Parameters are automatically persisted and can be retrieved in future sessions.

        Args:
            model: Neural network model object (PyTorch, TensorFlow, etc.).
            model_id: Unique identifier for the model. Auto-generated if None.
            parameter_overrides: Custom parameter extraction overrides by name.
            custom_handlers: Additional parameter handlers for custom parameter types.

        Raises:
            ModelAlreadyExistsError: If model_id already exists in the session.
            SessionError: If parameter extraction fails.
        """
        with self._own_context():
            try:
                kwargs = {}

                if model_id:
                    if len(self._get_view(model_ids=[model_id])) > 0:
                        msg = f"Model with uid {model_id} already exists"
                        raise ModelAlreadyExistsError(msg)
                    kwargs["model_id"] = model_id

                if parameter_overrides:
                    kwargs["parameter_overrides"] = parameter_overrides

                extractor_overrides = ExtractorOverrides(**kwargs)

                parameter_extractor = self._parameter_extractor_factory(
                    model=model,
                    overrides=extractor_overrides,
                    custom_handlers=custom_handlers,
                )

                previous_len = len(self._parameter_repository)
                parameter_extractor.extract_parameters(
                    parameter_repository=self._parameter_repository,
                )
                new_len = len(self._parameter_repository)

                logger.info(
                    "Added model '%s' with %d parameters",
                    model_id or "<auto-generated>",
                    new_len - previous_len,
                )

            except SessionError:
                raise
            except Exception as e:
                msg = f"Failed to add model: {format_exception_message(e)}"
                raise SessionError(msg) from e

    def compute(
        self,
        *fields_to_produce: str,
        parameter_uids: list[str] | None = None,
        parameter_names: list[str] | None = None,
        parameter_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
    ) -> None:
        """Apply computational kernels to stored parameters.

        Executes the specified kernels on filtered parameters in dependency order.
        Results are automatically stored and can be retrieved using get_results().

        Args:
            *fields_to_produce: Names of fields to compute using registered kernels.
            parameter_uids: Filter by specific parameter UIDs.
            parameter_names: Filter by parameter names (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).
            parameter_types: Filter by parameter types.
            model_ids: Filter by model IDs (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).

        Raises:
            KernelNotFoundError: If any specified field cannot be produced.
            SessionError: If kernel execution fails.
        """
        with self._own_context():
            if not fields_to_produce:
                logger.warning("No fields specified for computation")
                return

            for field_name in fields_to_produce:
                if not self._kernel_registry.can_produce_field(field_name):
                    msg = f"Cannot produce '{field_name}': not found in registry"
                    raise KernelNotFoundError(msg)

            pending = self._get_view(
                parameter_uids, parameter_names, parameter_types, model_ids
            )

            try:
                with self._kernel_executor() as executor:
                    for field_name in fields_to_produce:
                        executor.execute(
                            field_or_kernel_name=field_name, parameters=pending
                        )

                logger.info(
                    "Successfully produced fields: %s", ", ".join(fields_to_produce)
                )

            except SessionError:
                raise
            except Exception as e:
                msg = f"Failed to produce fields: {format_exception_message(e)}"
                raise SessionError(msg) from e

    def get_results(
        self,
        *fields: str,
        export_format: Literal["dict", "json", "pandas"],
        parameter_uids: list[str] | None = None,
        parameter_names: list[str] | None = None,
        parameter_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
    ) -> Any:
        """Retrieve computation results for specified fields.

        Returns a StructuredExportResult containing:
        - scalars: Per-parameter fields from parameter repository
        - aggregates: Contextual/aggregation fields from aggregate repository

        Args:
            *fields: Field names to retrieve. Can include both scalar fields
                (e.g., "frob_norm") and contextual fields (e.g., "l_overlap").
            export_format: Output format - "dict", "json", or "pandas".
            parameter_uids: Filter by specific parameter UIDs.
            parameter_names: Filter by parameter names (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).
            parameter_types: Filter by parameter types.
            model_ids: Filter by model IDs (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).

        Returns:
            StructuredExportResult with scalars and aggregates in the specified format.
        """
        with self._own_context():
            # Get parameter view with filters
            param_view = self._get_view(
                parameter_uids, parameter_names, parameter_types, model_ids
            )

            # Get aggregate view with model filter if specified
            aggregate_view = self._aggregate_repository.create_view()
            if model_ids:
                aggregate_view = aggregate_view.filter_by_context_models(*model_ids)

            parallel = self._parallel_context()

            return self._results_exporter(
                *fields,
                parameters=param_view,
                aggregates=aggregate_view,
                export_format=export_format,
                parallel=parallel,
            )

    def erase_models(
        self,
        *model_ids: str,
        erase_all: bool = False,
    ) -> None:
        """Erase all parameters and results for specified models.

        Removes model parameters from both memory and persistent storage.
        Use with caution as this operation cannot be undone.

        Args:
            *model_ids: Model IDs to erase. Required if erase_all is False.
            erase_all: If True, erases all models. model_ids must be empty.

        Raises:
            ValueError: If neither model_ids nor erase_all is specified,
                or if both are specified.
            ModelNotFoundError: If any specified model ID is not found.
        """
        with self._own_context():
            if (not model_ids) ^ erase_all:
                if not model_ids:
                    msg = "No model_ids provided and erase_all=False"
                else:
                    msg = "Cannot specify both model_ids and erase_all=True"
                raise ValueError(msg)

            if erase_all:
                model_ids = tuple(self.list_models())

            for model_id in model_ids:
                model_params = self._get_view(model_ids=[model_id])

                if len(model_params) == 0:
                    msg = f"Model '{model_id}' not found"
                    raise ModelNotFoundError(msg)

                model_params.clear(erase=True)

                # Also erase aggregates that reference this model in context_models
                model_aggregates = self._aggregate_repository.create_view()
                model_aggregates = model_aggregates.filter_by_context_models(model_id)
                if model_aggregates:
                    logger.debug(
                        "Erasing %d aggregates for model '%s'",
                        len(model_aggregates),
                        model_id,
                    )
                    model_aggregates.clear(erase=True)

                logger.info("Erased model '%s'", model_id)

    def erase_results(
        self,
        *fields_to_erase: str,
        erase_dependent_also: bool = False,
        erase_all: bool = False,
    ) -> None:
        """Erase computation results for specified fields.

        Removes computed field data from parameters while preserving the
        parameters themselves. Optionally erases dependent fields that
        rely on the specified fields as inputs.

        Args:
            *fields_to_erase: Field names to erase. Required if erase_all is False.
            erase_dependent_also: If True, also erases fields that depend on
                the specified fields.
            erase_all: If True, erases all computed fields. fields_to_erase
                must be empty.

        Raises:
            ValueError: If neither fields_to_erase nor erase_all is specified,
                or if both are specified.
            KernelNotFoundError: If any specified field cannot be produced
                by registered kernels.
        """
        with self._own_context():
            if (not fields_to_erase) ^ erase_all:
                if not fields_to_erase:
                    msg = "No fields_to_erase provided and erase_all=False"
                else:
                    msg = "Cannot specify both fields_to_erase and erase_all=True"
                raise ValueError(msg)

            eraser = ResultsEraser(kernel_registry=self._kernel_registry)

            try:
                if erase_all:
                    fields = set(self._kernel_registry.list_fields_can_produce())
                else:
                    fields = eraser.resolve_fields_to_erase(
                        fields_to_erase, erase_dependent_also
                    )
            except ResultsEraserError as e:
                raise KernelNotFoundError(str(e)) from e

            eraser.erase(
                view=self._get_view(),
                aggregates=self._aggregate_repository.create_view(),
                fields=fields,
            )

    def list_models(self) -> list[str]:
        """List all model IDs currently in the session.

        Returns:
            Sorted list of unique model IDs present in the session.
        """
        with self._own_context():
            model_ids = {param.meta.model_id for param in self._get_view()}
            return sorted(model_ids)

    def list_aggregates(
        self,
        field_names: list[str] | None = None,
        model_ids: list[str] | None = None,
        *,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """List aggregate metadata for inspection.

        Provides information about aggregates (aggregation results) currently
        stored in the session, including field names, context, and values.

        Args:
            field_names: Filter by aggregate field names.
            model_ids: Filter by models in context_models.
            verbose: If True, include the computed value.

        Returns:
            List of dictionaries containing aggregate metadata including
            field_name, context_models, context_params, and optionally value.
        """
        with self._own_context():
            view = self._aggregate_repository.create_view()

            if field_names:
                view = view.filter_by_field_name(*field_names)

            if model_ids:
                view = view.filter_by_context_models(*model_ids)

            if verbose:
                view.prefetch_fields(fields=["value"], parallel=None)

            result: list[dict[str, Any]] = []
            for aggregate in view:
                entry: dict[str, Any] = {
                    "uid": aggregate.meta.uid,
                    "field_name": aggregate.meta.field_name,
                    "context_models": aggregate.meta.context_models,
                    "context_params": aggregate.meta.context_params,
                }
                if verbose and aggregate.has_field("value"):
                    entry["value"] = aggregate.get_field("value")
                result.append(entry)

            return result

    def list_kernels(self, verbose: bool = False) -> list[str]:
        """List all available computational kernels.

        Args:
            verbose: If True, include detailed kernel information including
                dependencies and field requirements.

        Returns:
            List of kernel names or detailed kernel descriptions.
        """
        with self._own_context():
            return self._kernel_registry.list_kernels(verbose=verbose)

    def list_fields_can_compute(self, verbose: bool = False) -> list[str]:
        """List all fields that can be computed by registered kernels.

        Args:
            verbose: If True, include detailed field information including
                producing kernels and dependencies.

        Returns:
            List of computable field names or detailed field descriptions.
        """
        with self._own_context():
            return self._kernel_registry.list_fields_can_produce(verbose=verbose)

    def draw(
        self,
        *,
        plot: Any = None,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
        theme: Any = None,
        theme_path: str | Path | None = None,
    ) -> Any:
        """Render a Plotly figure using the viz module.

        Provide either `plot` (a Plot object) or `config_path` (a Hydra YAML file).

        Args:
            plot: A Plot instance to render.
            config_path: Path to a Hydra YAML config file.
            overrides: Hydra overrides to apply (only with config_path).
            theme: A Theme instance to apply.
            theme_path: Path to a YAML file with theme config.
        """
        if (plot is None) == (config_path is None):
            raise ValueError("Provide exactly one of: plot=... or config_path=...")

        if config_path is not None:
            from diffract.viz.renderer import render_from_config

            return render_from_config(
                session=self,
                config_path=config_path,
                overrides=overrides,
                theme=theme,
                theme_path=theme_path,
            )

        from diffract.viz.renderer import render

        return render(plot, session=self, theme=theme)

    def list_parameters(
        self,
        parameter_uids: list[str] | None = None,
        parameter_names: list[str] | None = None,
        parameter_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
        *,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """List parameter metadata for inspection.

        Provides detailed information about parameters currently stored
        in the session, including computed fields and metadata.

        Returns:
            List of dictionaries containing parameter metadata including
            UID, name, model ID, type, available fields, and custom metadata.
        """
        with self._own_context():
            filtered = self._get_view(
                parameter_uids, parameter_names, parameter_types, model_ids
            )

            if not verbose:
                return [
                    {
                        "uid": param.meta.uid,
                        "name": param.meta.name,
                        "model_id": param.meta.model_id,
                        "parameter_type": param.meta.ptype.name,
                    }
                    for param in filtered
                ]

            params = list(filtered)
            parallel = self._parallel_context()
            field_cache = filtered.list_fields_by_uid(parallel=parallel)

            return [
                {
                    "uid": param.meta.uid,
                    "name": param.meta.name,
                    "model_id": param.meta.model_id,
                    "parameter_type": param.meta.ptype.name,
                    "available_fields": field_cache.get(param.meta.uid, []),
                    "other_meta": param.meta.other_meta,
                }
                for param in params
            ]

    def patch_meta(
        self,
        *,
        updates: dict[str, Any],
        force: bool = False,
    ) -> None:
        """Update custom metadata (`other_meta`) for selected parameters.

        This method updates only the `other_meta` portion of parameter metadata.
        Changes are persisted by rewriting the stored `__metadata__` field, so
        subsequent sessions will see the updated metadata.

        Args:
            updates: Mapping of parameter UID to metadata updates. Each value
                must be a dict of {key: value} pairs to apply to that
                parameter's other_meta.
                Example: {"uid1": {"dataset": "imagenet", "epoch": 10}}
            force: If False (default), raise if any key already exists in
                `other_meta`. If True, overwrite existing keys.

        Raises:
            SessionError: If conflicts are detected (force=False), invalid keys
                are provided, or updates structure is invalid.
        """
        with self._own_context():
            if not updates:
                return

            for uid, value in updates.items():
                if not isinstance(value, dict):
                    msg = (
                        f"Invalid update for uid '{uid}': expected dict[str, Any], "
                        f"got {type(value).__name__}"
                    )
                    raise SessionError(msg)

            parameters = self._get_view(parameter_uids=list(updates.keys()))
            patcher = MetadataPatcher()

            try:
                patcher.patch(updates=updates, parameters=parameters, force=force)
            except MetadataPatchError as e:
                raise SessionError(str(e)) from e

    def ingest_fields(
        self,
        fields_by_uid: dict[str, dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        """Ingest precomputed fields into the session via a uid->field mapping.

        Field names are stored exactly as provided (including contextual suffixes
        like "metric@ctx"). By default, this method is strict and raises if any
        target field already exists; set force=True to overwrite.

        Args:
            fields_by_uid: Mapping of parameter uid -> {field_name: value}.
            force: If False (default), raise on any existing field conflict.
                If True, overwrite existing field values.

        Raises:
            SessionError: If unknown uids are provided, forbidden fields are
                requested, or conflicts are detected (force=False).
        """
        with self._own_context():
            if not fields_by_uid:
                return

            parameters = self._get_view(parameter_uids=list(fields_by_uid.keys()))
            ingester = FieldIngester()

            try:
                ingester.ingest(
                    fields_by_uid=fields_by_uid, parameters=parameters, force=force
                )
            except FieldIngestionError as e:
                raise SessionError(str(e)) from e

    def ingest_aggregates(
        self,
        aggregates: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        """Ingest precomputed aggregate values into the session.

        Each aggregate is identified by (field_name, context_models, context_params).
        Useful for importing precomputed aggregation results.

        Args:
            aggregates: List of aggregate dictionaries, each containing:
                - field_name: Base field name (e.g., "l_overlap").
                - context_models: Tuple/list of model IDs.
                - context_params: Tuple/list of parameter names (optional).
                - value: The computed value to store.
            force: If False (default), raise on existing aggregate conflict.
                If True, overwrite existing values.

        Raises:
            SessionError: If aggregate structure is invalid or conflicts are
                detected (force=False).
        """
        with self._own_context():
            if not aggregates:
                return

            ingester = AggregateIngester()
            try:
                ingester.ingest(
                    aggregates=aggregates,
                    repository=self._aggregate_repository,
                    force=force,
                )
            except AggregateIngestionError as e:
                raise SessionError(str(e)) from e

    def configure_kernel(self, kernel_name: str, **config: Any) -> None:
        """Configure parameters for a specific kernel.

        Updates the configuration for a registered kernel, affecting
        subsequent computations using that kernel.

        Args:
            kernel_name: Name of kernel to configure.
            **config: Kernel configuration parameters as keyword arguments.

        Raises:
            KernelNotFoundError: If kernel is not registered in the registry.
        """
        with self._own_context():
            if not self._kernel_registry.has_kernel(kernel_name):
                msg = f"Kernel '{kernel_name}' not found in registry"
                raise KernelNotFoundError(msg)

            from .core.compute.registry import KernelConfig

            kernel_config = KernelConfig(**config)
            self._kernel_registry.configure_kernel(kernel_name, kernel_config)

            logger.info("Configured kernel '%s' with: %s", kernel_name, config)

    def merge(
        self,
        other_session: Session,
        fields: list[str] | None = None,
        parameter_uids: list[str] | None = None,
        parameter_names: list[str] | None = None,
        parameter_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
        *,
        verify: bool = True,
        read_budget_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Merge parameters from another session into this one.

        Copies parameters and their computed fields from another session.
        When verify is True, handles conflicts by skipping duplicate fields.

        Args:
            other_session: Source session to merge from.
            fields: Specific fields to merge. If None, merges all fields.
            parameter_uids: Filter source parameters by UIDs.
            parameter_names: Filter source parameters by names (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).
            parameter_types: Filter source parameters by types.
            model_ids: Filter source parameters by model IDs (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).
            verify: If True, check for conflicts and skip duplicates.
            read_budget_bytes: Maximum bytes to read during merge operation.
        """
        with self._own_context():
            parallel = self._parallel_context()
            target_state: MergeTargetState | None = None
            if verify:
                target_state = MergeTargetState()

                for parameter_info in self.list_parameters(verbose=True):
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

            source = other_session._get_view(
                parameter_uids, parameter_names, parameter_types, model_ids
            )

            # Merge parameters if any exist
            if source:
                logger.info(
                    "Starting merge from session '%s' (%d parameters)",
                    other_session,
                    len(source),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Merge source models: %s", other_session.list_models())

                source_fields_by_uid = source.list_fields_by_uid(parallel=parallel)

                merger = SessionMerger(parallel=parallel)
                merger.merge(
                    source=source,
                    source_fields_by_uid=source_fields_by_uid,
                    target_repository=self._parameter_repository,
                    target_state=target_state,
                    fields=fields,
                    read_budget_bytes=read_budget_bytes,
                )
            else:
                logger.debug("No parameters to merge from source session")

            # Merge aggregates from source session (independent of parameters)
            aggregate_merger = AggregateMerger()
            aggregate_merger.merge(
                source_aggregates=other_session._aggregate_repository.create_view(),
                target_repository=self._aggregate_repository,
                model_ids=model_ids,
                fields=fields,
                verify=verify,
            )

    def kernel(
        self,
        *,
        name: str | None = None,
        require_fields: tuple[str, ...] | None = None,
        produce_fields: tuple[str, ...] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering custom kernels in the session registry.

        Creates a kernel that can compute metrics on neural network parameters.
        Dependencies are automatically inferred from function arguments.

        Args:
            name: Custom kernel name (defaults to function name).
            require_fields: Input field names (inferred from signature if None).
            produce_fields: Output field names (defaults to kernel name).

        Returns:
            Decorator function for registering the kernel.

        Example:
            >>> session = Session()
            >>> with session:
            ...     @session.kernel()
            ...     def my_metric(frob_norm: float, *, scale: float = 1.0) -> float:
            ...         return frob_norm * scale
        """
        from .core.compute.execution import (
            KernelApplyLevel,
            KernelExecutionProtocol,
            KernelRestrictions,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            req_auto, cfg = self._kernel_registry._split_signature(func)  # noqa: SLF001
            final_name = name or func.__name__
            final_require = require_fields or req_auto
            final_produce = produce_fields or (final_name,)

            self._kernel_registry.register_kernel(
                name=final_name,
                require_fields=final_require,
                produce_fields=final_produce,
                implementation=func,
                apply_level=KernelApplyLevel.PARAMETER,
                execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
                restrictions=None,
                config=cfg,
                info=None,
            )
            return func

        return decorator

    def _get_view(
        self,
        parameter_uids: list[str] | None = None,
        parameter_names: list[str] | None = None,
        parameter_types: list[ParameterType] | None = None,
        model_ids: list[str] | None = None,
    ) -> IParameterView:
        """Filter parameters based on provided criteria.

        Args:
            parameter_uids: Filter by specific parameter UIDs.
            parameter_names: Filter by parameter names (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).
            parameter_types: Filter by parameter types.
            model_ids: Filter by model IDs (exact by default).
                To use regular expressions, prefix an entry with "re:" and provide
                a Python regular expression pattern (matched via re.fullmatch).

        Returns:
            Filtered ParameterView containing only parameters
            matching all specified criteria.
        """
        with self._own_context():
            view: IParameterView = self._parameter_repository.create_view()

            if parameter_names:
                view = view.filter_by_name(*parameter_names)

            if parameter_types:
                view = view.filter_by_ptype(*parameter_types)

            if model_ids:
                view = view.filter_by_model_id(*model_ids)

            if parameter_uids:
                view = view[parameter_uids]

            return view
