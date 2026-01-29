"""Session utility helpers for complex operations.

This module provides utility classes for Session operations like merging,
field ingestion, metadata patching, and results erasure.
"""

from .ingester import (
    AggregateIngester,
    AggregateIngestionError,
    FieldIngester,
    FieldIngestionError,
)
from .merger import AggregateMerger, MergeTargetState, SessionMerger
from .meta_patcher import MetadataPatcher, MetadataPatchError
from .results_eraser import ResultsEraser, ResultsEraserError

__all__ = [
    "AggregateIngester",
    "AggregateIngestionError",
    "AggregateMerger",
    "FieldIngester",
    "FieldIngestionError",
    "MergeTargetState",
    "MetadataPatchError",
    "MetadataPatcher",
    "ResultsEraser",
    "ResultsEraserError",
    "SessionMerger",
]
