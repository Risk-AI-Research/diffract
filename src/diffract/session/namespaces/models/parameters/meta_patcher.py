"""Parameter management namespace for Session models."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from diffract.core.constants import (
    PROGRESS_BAR_DELAY_SEC,
    PROGRESS_BAR_MIN_ITEMS,
    TABLE_PARAMETERS,
)

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.proxy import ParameterDataProxy

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class MetadataPatchError(Exception):
    """Error during metadata patching."""


class MetadataPatcher:
    """Utility for patching parameter metadata.

    Handles conflict detection and applies updates to the other_meta
    portion of parameter metadata.

    Example:
        >>> patcher = MetadataPatcher()
        >>> patcher.patch(
        ...     updates={"uid1": {"dataset": "imagenet", "epoch": 10}},
        ...     parameters=session._get_view(),
        ...     force=False,
        ... )
    """

    def patch(
        self,
        *,
        updates: dict[str, dict[str, Any]],
        parameters: IParameterView,
        force: bool = False,
    ) -> None:
        """Patch other_meta with conflict detection.

        Args:
            updates: Mapping of parameter uid -> {meta_key: value}.
            parameters: Parameter view containing target parameters.
            force: If True, overwrite existing metadata keys.

        Raises:
            MetadataPatchError: If unknown UIDs or conflicts are detected.
        """
        if not updates:
            logger.debug("patch called with empty updates, skipping")
            return

        resolved, unknown_uids = self._resolve_parameters(updates, parameters)

        if unknown_uids:
            preview = ", ".join(sorted(unknown_uids)[:_ERROR_PREVIEW_LIMIT])
            suffix = "..." if len(unknown_uids) > _ERROR_PREVIEW_LIMIT else ""
            raise MetadataPatchError(
                f"Unknown parameter UIDs (not found in session): {preview}{suffix}"
            )

        conflicts = self._validate_conflicts(updates, resolved, force)

        if conflicts:
            preview = ", ".join(
                f"{uid}:{k}" for uid, k in conflicts[:_ERROR_PREVIEW_LIMIT]
            )
            suffix = "..." if len(conflicts) > _ERROR_PREVIEW_LIMIT else ""
            raise MetadataPatchError(
                f"Metadata conflicts (existing keys) for {len(conflicts)} entries: "
                f"{preview}{suffix}"
            )

        self._apply_updates(updates, resolved, parameters)
        logger.info("Patched metadata for %d parameters", len(updates))

    def _resolve_parameters(
        self,
        updates: dict[str, dict[str, Any]],
        parameters: IParameterView,
    ) -> tuple[dict[str, ParameterDataProxy], list[str]]:
        """Resolve parameter proxies from UIDs."""
        resolved: dict[str, ParameterDataProxy] = {}
        unknown_uids: list[str] = []

        for uid in updates:
            try:
                resolved[uid] = parameters[uid]
            except KeyError:
                unknown_uids.append(uid)

        return resolved, unknown_uids

    def _validate_conflicts(
        self,
        updates: dict[str, dict[str, Any]],
        resolved: dict[str, ParameterDataProxy],
        force: bool,
    ) -> list[tuple[str, str]]:
        """Check for conflicting metadata keys."""
        conflicts: list[tuple[str, str]] = []

        for uid, meta_map in updates.items():
            param = resolved[uid]
            conflicts.extend(
                (param.meta.uid, key)
                for key in meta_map
                if (not force) and (key in param.meta.other_meta)
            )

        return conflicts

    def _apply_updates(
        self,
        updates: dict[str, dict[str, Any]],
        resolved: dict[str, ParameterDataProxy],
        parameters: IParameterView,
    ) -> None:
        """Apply metadata updates to parameters.

        Updates other_meta in both the proxy and the metadata index.
        """
        total = len(updates)
        repository = parameters._repository

        with repository:
            for uid, meta_map in tqdm(
                updates.items(),
                desc="Updating metadata...",
                delay=PROGRESS_BAR_DELAY_SEC,
                disable=total < PROGRESS_BAR_MIN_ITEMS,
                total=total,
            ):
                param = resolved[uid]
                old_meta = param.meta

                # Build updated other_meta
                new_other_meta = dict(old_meta.other_meta)
                new_other_meta.update(meta_map)

                # Create new metadata with updated other_meta
                new_meta = replace(old_meta, other_meta=new_other_meta)

                # Update the proxy's meta reference (frozen, so we use
                # object.__setattr__)
                object.__setattr__(param, "meta", new_meta)

                new_meta_dict = new_meta.to_dict()
                new_meta_dict.pop("uid")
                repository.metadata_index.upsert(
                    TABLE_PARAMETERS,
                    uid=uid,
                    **new_meta_dict,
                )
