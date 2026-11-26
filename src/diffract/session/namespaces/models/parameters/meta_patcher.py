"""Parameter management namespace for Session models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from diffract.core.constants import (
    PROGRESS_BAR_DELAY_SEC,
    PROGRESS_BAR_MIN_ITEMS,
)

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterProxy, IParameterView

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_LIMIT = 10


class MetadataPatchError(Exception):
    """Error during metadata patching."""


class MetadataPatcher:
    """Validate metadata updates before applying them to parameter records."""

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
    ) -> tuple[dict[str, IParameterProxy], list[str]]:
        """Resolve parameter proxies from UIDs."""
        resolved: dict[str, IParameterProxy] = {}
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
        resolved: dict[str, IParameterProxy],
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
        resolved: dict[str, IParameterProxy],
        parameters: IParameterView,
    ) -> None:
        """Apply metadata updates to parameters.

        Updates other_meta in both the proxy and the metadata index.
        """
        total = len(updates)
        with parameters:
            for uid, meta_map in tqdm(
                updates.items(),
                desc="Updating metadata...",
                delay=PROGRESS_BAR_DELAY_SEC,
                disable=total < PROGRESS_BAR_MIN_ITEMS,
                total=total,
            ):
                resolved[uid].update_other_meta(meta_map)
