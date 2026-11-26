"""Model management namespace for Session."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

from dependency_injector.wiring import Provide, inject

from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.identity import (
    DuplicateParameterError,
    IdentifierValidationError,
    training_step,
)
from diffract.core.data.metadata.errors import UniqueConstraintError
from diffract.core.data.nn.extractors.base import ExtractorOverrides, ParameterOverrides
from diffract.core.data.nn.extractors.report import ExtractionReport
from diffract.core.storage.catalog_manager import storage_activity
from diffract.core.utils.exceptions import format_exception_message
from diffract.core.utils.hashing import get_unique_id
from diffract.session._identifiers import check_identifier
from diffract.session.errors import (
    EmptyExtractionError,
    InvalidIdentifierError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    SessionError,
)
from diffract.session.resolver import resolve_models
from diffract.session.session import Session, SessionContext
from diffract.session.summaries import EraseSummary

from .parameters import ParametersNamespace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from diffract.core.data.nn.aggregates.repository import AggregateRepository
    from diffract.core.data.nn.extractors.handlers.base import ParameterHandler
    from diffract.core.data.nn.extractors.interface import IParameterExtractor
    from diffract.core.data.nn.params.interface import IParameterRepository


logger = logging.getLogger(__name__)

__all__ = [
    "ModelsNamespace",
    "ParametersNamespace",
]


class ModelsNamespace:
    """Model management API."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        aggregate_repository: AggregateRepository = Provide["nn.aggregate_repository"],
        extractor_factory: Callable[..., IParameterExtractor] = Provide[
            "nn.extractor_factory.provider"
        ],
        parameters: ParametersNamespace | None = None,
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__agg_repo = aggregate_repository
        self.__extractor_factory = extractor_factory

        self.parameters = (
            parameters
            if parameters is not None
            else ParametersNamespace(session_or_context)
        )

    def add(
        self,
        model: Any,
        model_id: str | None = None,
        parameter_overrides: dict[str, ParameterOverrides] | None = None,
        custom_handlers: Iterable[ParameterHandler] | None = None,
        *,
        step: int = 0,
        display_label: str | None = None,
        required_names: Iterable[str] = (),
    ) -> ExtractionReport:
        """Add a neural network model to the session.

        Stores a snapshot of the weights at this step. Once this call returns,
        the caller can update the model or reuse its arrays for the next step.
        Persistent profiles also make the checkpoint available in later sessions.

        Args:
            model: Neural network model object (PyTorch, TensorFlow, etc.)
                or a dict mapping parameter names to NumPy arrays.
            model_id: Unique identifier for the model. Auto-generated if None.
            parameter_overrides: Custom parameter extraction overrides by name.
            custom_handlers: Additional parameter handlers for custom parameter types.
            step: Nonnegative training step. Defaults to zero.
            display_label: Optional display text for this checkpoint.
            required_names: Exact source parameter names that must be extractable.
                Their converted weights are held in memory until all required
                names pass. Other eligible parameters are also stored.

        Returns:
            Per-name extraction outcomes, reasons and source and storage dtypes.

        Raises:
            ModelAlreadyExistsError: If the model already has parameters at this step.
            TypeError: If the step is not an integer or the label is not text.
            ValueError: If the step is negative or exceeds the storage range.
            InvalidIdentifierError: If model_id is not over the accepted
                identifier alphabet.
            EmptyExtractionError: If there are no extracted parameters. Its
                coverage attribute contains the per-input reasons.
            SessionError: If required names cannot be extracted or the import fails.
        """
        step = training_step(step)
        if display_label is not None and not isinstance(display_label, str):
            raise TypeError("display_label must be a string or None.")
        with self.__session_or_context:
            try:
                name = model_id or get_unique_id()
                check_identifier(name, kind="model id")
                with (
                    storage_activity(self.__param_repo.storage_manager, "ingest"),
                    self.__param_repo,
                ):
                    report = self.__add_at_coordinate(
                        model,
                        name,
                        model_id,
                        parameter_overrides,
                        custom_handlers,
                        step,
                        display_label,
                        frozenset(
                            (required_names,)
                            if isinstance(required_names, str)
                            else required_names
                        ),
                    )
            except (SessionError, DuplicateParameterError):
                raise
            except IdentifierValidationError as e:
                raise InvalidIdentifierError(str(e)) from e
            except Exception as e:
                msg = f"Failed to add model: {format_exception_message(e)}"
                raise SessionError(msg) from e

            self.__session_or_context._field_cache.invalidate()
            logger.info("Added model '%s'", name)
            return report

    def __add_at_coordinate(
        self,
        model: Any,
        name: str,
        model_id: str | None,
        parameter_overrides: dict[str, ParameterOverrides] | None,
        custom_handlers: Iterable[ParameterHandler] | None,
        step: int,
        display_label: str | None,
        required_names: frozenset[str],
    ) -> ExtractionReport:
        instances = self.__param_repo.instances
        instance = instances.get_or_create_instance(name)
        coordinate = {"instance_uid": instance.instance_uid, "step": step}
        index = self.__param_repo.metadata_index
        if index.query(TABLE_PARAMETERS, where=coordinate, limit=1):
            raise ModelAlreadyExistsError(
                f"Model '{name}' already exists at step {step}"
            )
        extractor = self.__extractor_factory(
            model=model,
            overrides=ExtractorOverrides(
                model_id=model_id, parameter_overrides=parameter_overrides or {}
            ),
            custom_handlers=custom_handlers,
        )
        report = extractor.extract_parameters(
            self.__param_repo,
            instance=instance,
            step=step,
            required_names=required_names,
        )
        if not report.counts["extracted"]:
            raise EmptyExtractionError(name, report)

        instances.set_extraction(instance.instance_uid, step, report)
        if display_label is not None:
            instances.set_step_label(
                instance.instance_uid, step, display_label, overwrite=True
            )
        return report

    def set_display_label(self, model_id: str, step: int, label: str) -> None:
        """Set the display text for an existing checkpoint in this scope."""
        if not isinstance(label, str):
            raise TypeError("label must be a string.")
        with self.__session_or_context, self.__param_repo:
            global_scope = isinstance(self.__session_or_context, Session)
            view = (
                self.__param_repo.create_view()
                if global_scope
                else self.__session_or_context._param_filter_context
            )
            selection = resolve_models(
                view,
                self.__param_repo.instances,
                [(model_id, step)],
                allow_empty_instances=False,
            )
            (coordinate,) = selection.coordinates
            self.__param_repo.instances.set_step_label(
                *coordinate, label, overwrite=True
            )
        self.__session_or_context._field_cache.invalidate()

    def rename(
        self,
        model_id: str,
        new_model_id: str,
    ) -> None:
        """Rename a model in the session.

        Args:
            model_id: ID of the model to rename.
            new_model_id: New ID for the model.

        Raises:
            ModelNotFoundError: If model_id does not exist.
            ModelAlreadyExistsError: If new_model_id already exists.
            InvalidIdentifierError: If new_model_id is not over the accepted
                identifier alphabet.
        """
        with self.__session_or_context:
            check_identifier(new_model_id, kind="model id")

            instances = self.__param_repo.instances
            with self.__param_repo.metadata_index:
                instance = instances.get_by_name(model_id)
                if instance is None:
                    raise ModelNotFoundError(f"Model '{model_id}' not found")
                if model_id == new_model_id:
                    raise ModelAlreadyExistsError(
                        f"Model '{new_model_id}' already exists"
                    )
                try:
                    instances.rename(instance.instance_uid, new_model_id)
                except UniqueConstraintError as exc:
                    raise ModelAlreadyExistsError(
                        f"Model '{new_model_id}' already exists"
                    ) from exc
            self.__agg_repo.refresh_names()
            self.__param_repo.refresh_names()
            self.__session_or_context._field_cache.invalidate()

            logger.info("Renamed model '%s' to '%s'", model_id, new_model_id)

    def erase(
        self,
        *model_ids: str | tuple[str, int],
        erase_all: bool = False,
    ) -> EraseSummary:
        """Erase selected parameters and the aggregates that reference them.

        The catalog tombstones their fields. A model remains registered while
        any checkpoint still has parameters.

        Args:
            *model_ids: Exact model names or (model name, step) pairs to erase.
            erase_all: Select every model visible in this scope. Pass no model IDs.

        Returns:
            An EraseSummary naming the affected models and counting removed parameters.

        Raises:
            ValueError: If neither model_ids nor erase_all is specified,
                or if both are specified.
            ModelNotFoundError: If any specified model ID is not found.
        """
        with self.__session_or_context:
            if (not model_ids) ^ erase_all:
                if not model_ids:
                    msg = "No model_ids provided and erase_all=False"
                else:
                    msg = "Cannot specify both model_ids and erase_all=True"
                raise ValueError(msg)

            global_scope = isinstance(self.__session_or_context, Session)
            with self.__param_repo:
                if erase_all:
                    model_ids = tuple(self.list())
                if global_scope:
                    params = self.__param_repo.create_view()
                else:
                    params = self.__session_or_context._param_filter_context

                selection = resolve_models(
                    params,
                    self.__param_repo.instances,
                    model_ids,
                    allow_empty_instances=global_scope,
                )
                selected = params.filter_by_coordinates(list(selection.coordinates))
                erased_uids = selected.list_uids()
                aggregate_uids = self.__agg_repo.referencing(erased_uids)
                self.__param_repo.erase_values(erased_uids)
                self.__agg_repo.erase_values(aggregate_uids)
                self.__agg_repo.remove_membership(aggregate_uids)
                removed = self.__param_repo.remove_membership(erased_uids)
                self.__param_repo.instances.prune_empty(
                    selection.coordinates, selection.instances
                )

            self.__param_repo.refresh_names()
            self.__agg_repo.refresh_names()
            self.__session_or_context._field_cache.remove_uids(erased_uids)
            return EraseSummary(models=selection.names, affected_uids=removed)

    @overload
    def list(self, *, layout: Literal["names"] = "names") -> list[str]: ...

    @overload
    def list(self, *, layout: Literal["coordinates"]) -> list[tuple[str, int]]: ...

    @overload
    def list(self, *, layout: Literal["grouped"]) -> dict[str, list[int]]: ...

    def list(
        self, *, layout: Literal["names", "coordinates", "grouped"] = "names"
    ) -> list[str] | list[tuple[str, int]] | dict[str, list[int]]:
        """List model names or their training checkpoints within the current scope.

        Args:
            layout: ``"names"`` returns sorted model names. ``"coordinates"`` returns
                (name, step) pairs sorted by name and numeric step. ``"grouped"``
                maps each model name to its ordered list of steps.
        """
        if layout not in {"names", "coordinates", "grouped"}:
            raise ValueError("layout must be 'names', 'coordinates' or 'grouped'")
        with self.__session_or_context:
            global_scope = isinstance(self.__session_or_context, Session)
            if global_scope and layout == "names":
                return self.__param_repo.instances.names()
            params = (
                self.__param_repo.create_view()
                if global_scope
                else self.__session_or_context._param_filter_context
            )
            coordinates = params.coordinates()
            instances = self.__param_repo.instances
            if layout == "names":
                return instances.names(list({uid for uid, _ in coordinates}))
            names = {
                instance.instance_uid: instance.name for instance in instances.list()
            }
            ordered = sorted((names[uid], step) for uid, step in coordinates)
            if layout == "coordinates":
                return ordered
            grouped: dict[str, list[int]] = (
                {name: [] for name in sorted(names.values())} if global_scope else {}
            )
            for name, step in ordered:
                grouped.setdefault(name, []).append(step)
            return grouped
