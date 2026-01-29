"""Utilities to describe stored fields (shape/kind/dtype) for plotting/inspection."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from diffract.core.compute.execution.enums import KernelApplyLevel
from diffract.core.compute.registry import KernelRegistry
from diffract.core.storage.interface import IStorageManager
from diffract.core.storage.metadata import ValueMetadata, infer_value_metadata
from diffract.core.utils.exceptions import format_exception_message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FieldSignature:
    """Metadata describing a stored field's characteristics."""

    field: str
    kind: str
    dtype: str | None
    shape: tuple[int, ...] | None
    ndim: int | None
    is_numeric: bool
    apply_level: KernelApplyLevel | None = None
    source: str | None = None  # e.g., "metadata", "inferred"
    available: bool = True


def _signature_from_meta(
    field: str,
    meta: dict[str, Any],
    *,
    apply_level: KernelApplyLevel | None = None,
    source: str = "metadata",
) -> FieldSignature:
    vm = ValueMetadata.from_jsonable(meta)
    return FieldSignature(
        field=field,
        kind=vm.kind,
        dtype=vm.dtype,
        shape=vm.shape,
        ndim=vm.ndim,
        is_numeric=vm.is_numeric,
        apply_level=apply_level,
        source=source,
    )


def _signature_from_value(
    field: str,
    value: Any,
    *,
    apply_level: KernelApplyLevel | None = None,
    source: str = "value",
) -> FieldSignature:
    vm = infer_value_metadata(value)
    return FieldSignature(
        field=field,
        kind=vm.kind,
        dtype=vm.dtype,
        shape=vm.shape,
        ndim=vm.ndim,
        is_numeric=vm.is_numeric,
        apply_level=apply_level,
        source=source,
    )


def collect_field_signatures(
    storage: IStorageManager,
    field_names: Iterable[str] | None = None,
    *,
    sample_limit: int = 8,
    apply_level_hint: KernelApplyLevel | None = None,
) -> dict[str, FieldSignature]:
    """Collect signatures for fields stored in a storage backend.

    Sampling is limited to avoid loading all parameters.
    """
    signatures: dict[str, FieldSignature] = {}

    fields = list(field_names) if field_names is not None else storage.list_fields()

    for field in fields:
        if field in signatures:
            continue

        try:
            obj_ids = storage.list_objs_has_field(field)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "Failed to list objects for field '%s': %s",
                field,
                format_exception_message(e),
                exc_info=True,
            )
            obj_ids = []

        obj_ids = obj_ids[:sample_limit]
        signature: FieldSignature | None = None

        for obj_uid in obj_ids:
            meta = storage.get_field_metadata(obj_uid, field)
            if meta is not None:
                signature = _signature_from_meta(
                    field, meta, apply_level=apply_level_hint, source="metadata"
                )
                break

            try:
                value = storage.get_field(obj_uid, field)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to read field '%s' for object '%s': %s",
                    field,
                    obj_uid,
                    format_exception_message(e),
                    exc_info=True,
                )
                continue

            signature = _signature_from_value(
                field, value, apply_level=apply_level_hint, source="value"
            )
            break

        if signature is not None:
            signatures[field] = signature

    return signatures


def _registry_apply_levels(registry: KernelRegistry) -> dict[str, KernelApplyLevel]:
    mapping: dict[str, KernelApplyLevel] = {}
    for kernel_name in registry.list_kernels():
        apply_level = registry.get_kernel_apply_level(kernel_name)
        for field in registry.get_fields_kernel_produce(kernel_name):
            mapping[field] = apply_level
    return mapping


def collect_field_catalog(
    storage: IStorageManager,
    registry: KernelRegistry,
    field_names: Iterable[str] | None = None,
    *,
    sample_limit: int = 8,
) -> dict[str, FieldSignature]:
    """Merge observed signatures with registry hints.

    Includes computable-only fields.
    """
    observed = collect_field_signatures(
        storage, field_names=field_names, sample_limit=sample_limit
    )
    apply_levels = _registry_apply_levels(registry)

    # Enrich observed with apply_level if known.
    for field, level in apply_levels.items():
        if field in observed and observed[field].apply_level is None:
            observed[field].apply_level = level

    # Add registry-only fields (not yet computed).
    for field, level in apply_levels.items():
        if field not in observed:
            observed[field] = FieldSignature(
                field=field,
                kind="object",
                dtype=None,
                shape=None,
                ndim=None,
                is_numeric=False,
                apply_level=level,
                source="registry",
                available=False,
            )

    return observed
