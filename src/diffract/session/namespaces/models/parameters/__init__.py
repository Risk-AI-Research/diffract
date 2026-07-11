"""Parameter management namespace for Session models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from diffract.core.parallel import ParallelContext
from diffract.session.errors import SessionError
from diffract.session.session import Session, SessionContext

from .meta_patcher import MetadataPatcher, MetadataPatchError

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterRepository

logger = logging.getLogger(__name__)


class ParametersNamespace:
    """Parameter management API for Session models."""

    @inject
    def __init__(
        self,
        session_or_context: Session | SessionContext,
        parameter_repository: IParameterRepository = Provide["nn.parameter_repository"],
        parallel_context_factory: Callable[[], ParallelContext] = Provide[
            "parallel_singleton.thread_pool_context.provider"
        ],
    ) -> None:
        self.__session_or_context = session_or_context
        self.__param_repo = parameter_repository
        self.__parallel_context_factory = parallel_context_factory

    def list(
        self,
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
        with self.__session_or_context:
            if isinstance(self.__session_or_context, Session):
                params = self.__param_repo.create_view()
            else:
                params = self.__session_or_context._param_filter_context

            if not verbose:
                return [
                    {
                        "uid": param.meta.uid,
                        "name": param.meta.name,
                        "model_id": param.meta.model_id,
                        "parameter_type": param.meta.ptype.name,
                    }
                    for param in params
                ]

            if self.__session_or_context._field_cache.is_valid:
                field_cache = self.__session_or_context._field_cache.get()
            else:
                field_cache = params.list_fields_by_uid(
                    parallel=self.__parallel_context_factory()
                )
                self.__session_or_context._field_cache.set(field_cache)

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
        with self.__session_or_context:
            if not updates:
                return

            for uid, value in updates.items():
                if not isinstance(value, dict):
                    msg = (
                        f"Invalid update for uid '{uid}': expected dict[str, Any], "
                        f"got {type(value).__name__}"
                    )
                    raise SessionError(msg)

            param_view = self.__param_repo.create_view()
            pending = param_view[updates.keys()]
            patcher = MetadataPatcher()

            try:
                patcher.patch(updates=updates, parameters=pending, force=force)
            except MetadataPatchError as e:
                raise SessionError(str(e)) from e
