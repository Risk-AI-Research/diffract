"""Centralized constants for magic strings used throughout diffract.

This module defines all special string patterns and field names to avoid
scattered string literals across the codebase.
"""

from __future__ import annotations

# --- Filter pattern prefix ---
# Prefix for regex patterns in filter functions (e.g., 're:layer\.\d+').
REGEX_PREFIX = "re:"

# --- Reserved field names ---
# Field name for storing parameter weight arrays.
WEIGHTS_FIELD = "weights"

# --- HDF5 storage internals ---
# HDF5 group name for storing object index datasets.
HDF5_INDEX_GROUP = "__index__"

# HDF5 dataset name within index group for object UIDs.
HDF5_INDEX_DATASET = "all_objs"

# Marker value for deleted entries in HDF5 index.
HDF5_INDEX_TOMBSTONE = ""

# --- Storage attribute names ---
# Attribute name for stored value type.
STORAGE_ATTR_TYPE = "type"

# Attribute name for stored value metadata.
STORAGE_ATTR_META = "value_meta"

# --- Zarr storage internals ---

# Zarr group name for storing object and field index datasets.
ZARR_INDEX_GROUP = "_index"

# Zarr dataset name for object UID index.
ZARR_INDEX_OBJS = "objs"

# Zarr dataset name for field name index.
ZARR_INDEX_FIELDS = "fields"

# --- Progress bar settings ---
# Delay before showing progress bar (seconds). Avoids clutter for fast ops.
PROGRESS_BAR_DELAY_SEC = 1.0

# Minimum items before showing a progress bar.
PROGRESS_BAR_MIN_ITEMS = 50

# --- Storage table names ---
# Table name for neural network parameter data.
TABLE_PARAMETERS = "parameters"

# Table name for aggregate data between parameters/models.
TABLE_AGGREGATES = "aggregates"
