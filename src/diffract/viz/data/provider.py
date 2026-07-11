from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .extraction import get_field_data
from .filtering import apply_filter
from .types import DataShape, DataType, Entry

if TYPE_CHECKING:
    from diffract.session import Session


class DataProvider:
    """Data provider for visualization.

    Fetches data from session and provides typed field access.
    Always fetches aggregates merged with entries.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._entries: dict[str, Entry] = {}
        self._cached_fields: set[str] = set()

    def fetch(
        self,
        fields: list[str],
        *,
        value_filter: dict[str, tuple[str, Any]] | None = None,
    ) -> dict[str, Entry]:
        """Fetch data from session with aggregates merged.

        Uses unified export API that handles aggregate merge internally.
        Always expands contextual fields into entries.

        Args:
            fields: Computed field names to fetch.
            value_filter: Optional filter {field: (op, threshold)}.

        Returns:
            Entries dict: {uid: Entry}
        """
        fields_to_fetch = set(fields)
        if value_filter:
            fields_to_fetch.update(value_filter.keys())

        missing_fields = fields_to_fetch - self._cached_fields
        if missing_fields:
            update = self._session.results.export(
                *sorted(missing_fields),
                sources="all",
                export_format="dict",
                expand_contextual=True,
            )
            self._merge_entries(update)
            self._cached_fields.update(missing_fields)

        if value_filter:
            return apply_filter(self._entries, value_filter)

        return self._entries

    def get_field_data(self, field: str) -> tuple[list[Any], DataType, DataShape]:
        """Fetch a field and return its values, data type, and data shape.

        Args:
            field: Computed field name to fetch.

        Returns:
            Tuple of (values, data type, data shape).
        """
        return get_field_data(self.fetch(fields=[field]), field)

    def _merge_entries(self, update: dict[str, Any]) -> None:
        for uid, item in update.items():
            fields = item.get("fields")
            metadata = item.get("metadata")

            if not isinstance(fields, dict):
                raise TypeError(
                    f"Expected 'fields' mapping for entry '{uid}', "
                    f"got {type(fields).__name__}"
                )
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Expected 'metadata' mapping for entry '{uid}', "
                    f"got {type(metadata).__name__}"
                )

            merged_fields = dict(fields)
            merged_fields.update(metadata)

            existing = self._entries.get(uid)
            if existing is not None:
                existing_fields = existing.get("fields", {})
                if isinstance(existing_fields, dict):
                    merged_fields = dict(existing_fields) | merged_fields

            self._entries[uid] = Entry(fields=merged_fields)
