"""Parameter repository for managing neural network parameters.

This module provides the ParameterRepository class that extends the generic
DataRepository with parameter-specific functionality.
"""

from __future__ import annotations

from diffract.core.constants import TABLE_PARAMETERS
from diffract.core.data.repository import DataRepository

from .metadata import ParameterMetadata
from .proxy import ParameterDataProxy
from .view import ParameterView


class ParameterRepository(DataRepository[ParameterMetadata, ParameterDataProxy]):
    """Repository owning storage/cache/metadata and managing parameter membership.

    Extends the generic DataRepository with parameter-specific configuration
    for metadata class, proxy class, view class, storage table, and metadata schema.
    """

    METADATA_CLASS = ParameterMetadata
    PROXY_CLASS = ParameterDataProxy
    VIEW_CLASS = ParameterView
    TABLE = TABLE_PARAMETERS

    # Schema for MetadataIndex
    METADATA_COLUMNS = {
        "name": str,
        "model_id": str,
        "ptype": str,
    }
    METADATA_INDEXES = ["model_id", "ptype", "name"]

    def create_view(self) -> ParameterView:
        """Create a view over all parameters in this repository.

        Returns:
            ParameterView containing all parameters.
        """
        return ParameterView(repository=self, uids=None)
