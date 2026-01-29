"""Centralized constants for magic strings used throughout diffract.

This module defines all special string patterns and field names to avoid
scattered string literals across the codebase.
"""

from __future__ import annotations

# --- Filter pattern prefix ---
REGEX_PREFIX = "re:"
"""Prefix for regex patterns in filter functions (e.g., 're:layer\\.\\d+')."""

# --- Reserved field names ---
WEIGHTS_FIELD = "weights"
"""Field name for storing parameter weight arrays."""

# --- HDF5 storage internals ---
HDF5_INDEX_GROUP = "__index__"
"""HDF5 group name for storing object index datasets."""

HDF5_INDEX_DATASET = "all_objs"
"""HDF5 dataset name within index group for object UIDs."""

HDF5_INDEX_TOMBSTONE = ""
"""Marker value for deleted entries in HDF5 index."""

# --- Storage attribute names ---
STORAGE_ATTR_TYPE = "type"
"""Attribute name for stored value type."""

STORAGE_ATTR_META = "value_meta"
"""Attribute name for stored value metadata."""

# --- Zarr storage internals ---

ZARR_INDEX_GROUP = "_index"
"""Zarr group name for storing object and field index datasets."""

ZARR_INDEX_OBJS = "objs"
"""Zarr dataset name for object UID index."""

ZARR_INDEX_FIELDS = "fields"
"""Zarr dataset name for field name index."""

# --- Contextual field naming for aggregations ---
CONTEXT_SEPARATOR = "@"
"""Separator between field name and context suffix (e.g., 'metric@models[...]')."""

MODELS_CONTEXT_PREFIX = "models"
"""Context key for model identifiers in aggregated field names."""

PARAMS_CONTEXT_PREFIX = "params"
"""Context key for parameter names in aggregated field names."""

# --- Progress bar settings ---
PROGRESS_BAR_DELAY_SEC = 1.0
"""Delay before showing progress bar (seconds). Avoids clutter for fast ops."""

PROGRESS_BAR_MIN_ITEMS = 50
"""Minimum items before showing a progress bar."""

# --- Storage table names ---
TABLE_PARAMETERS = "parameters"
"""Table name for neural network parameter data."""

TABLE_AGGREGATES = "aggregates"
"""Table name for aggregate data between parameters/models."""


def format_context_part(key: str, values: tuple[str, ...]) -> str:
    """Format a single context part like 'models[m1,m2]'."""
    return f"{key}[{','.join(sorted(values))}]"


def format_field_suffix(
    models: tuple[str, ...] | None = None,
    params: tuple[str, ...] | None = None,
) -> str:
    """Build a deterministic suffix from context identifiers.

    Args:
        models: Model identifiers participating in aggregation.
        params: Parameter names participating in aggregation.

    Returns:
        Formatted suffix like 'models[m1,m2]@params[p1,p2]' or empty string.
    """
    parts: list[str] = []
    if models:
        parts.append(format_context_part(MODELS_CONTEXT_PREFIX, models))
    if params:
        parts.append(format_context_part(PARAMS_CONTEXT_PREFIX, params))
    return CONTEXT_SEPARATOR.join(parts) if parts else ""


def format_contextual_field_name(
    field_name: str,
    models: tuple[str, ...] | None = None,
    params: tuple[str, ...] | None = None,
) -> str:
    """Compose contextual field name with optional context suffix.

    Args:
        field_name: Base field name.
        models: Model identifiers for context.
        params: Parameter names for context.

    Returns:
        Field name with context suffix if any context provided.
    """
    suffix = format_field_suffix(models, params)
    return f"{field_name}{CONTEXT_SEPARATOR}{suffix}" if suffix else field_name


def parse_contextual_field_name(
    full_field_name: str,
) -> tuple[str, tuple[str, ...] | None, tuple[str, ...] | None]:
    """Parse a contextual field name into base name and context components.

    Args:
        full_field_name: Field name like 'metric@models[m1,m2]@params[p1]'.

    Returns:
        Tuple of (base_name, models, params) where models and params are None
        if not present in the suffix.
    """
    import re

    if CONTEXT_SEPARATOR not in full_field_name:
        return full_field_name, None, None

    parts = full_field_name.split(CONTEXT_SEPARATOR)
    base_name = parts[0]

    models: tuple[str, ...] | None = None
    params: tuple[str, ...] | None = None

    models_pattern = re.compile(rf"^{MODELS_CONTEXT_PREFIX}\[(.+)\]$")
    params_pattern = re.compile(rf"^{PARAMS_CONTEXT_PREFIX}\[(.+)\]$")

    for part in parts[1:]:
        if m := models_pattern.match(part):
            models = tuple(m.group(1).split(","))
        elif m := params_pattern.match(part):
            params = tuple(m.group(1).split(","))

    return base_name, models, params
