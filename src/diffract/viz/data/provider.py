from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diffract.session.resolver import FieldAddress, FieldSelector, resolve

from .extraction import get_field_data, matching_fields, plot_address
from .filtering import apply_filter
from .types import DataShape, DataType, Entry, EntryContext

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
        self._cached_fields: set[str | FieldSelector] = set()
        self._generation = session.results.generation

    def fetch(
        self,
        fields: list[FieldAddress],
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
        generation = self._session.results.generation
        if generation != self._generation:
            self._entries.clear()
            self._cached_fields.clear()
            self._generation = generation
        addresses = (plot_address(f) for f in [*fields, *(value_filter or {})])
        fields_to_fetch = {
            "name" if isinstance(f, FieldSelector) and f.field == "name" else f
            for f in addresses
        }

        missing_fields = fields_to_fetch - self._cached_fields
        if missing_fields:
            update = self._session.results.export(
                *sorted(missing_fields, key=repr),
                sources="all",
                export_format="dict",
                expand_contextual=True,
            )
            self._merge_entries(update)
            self._cached_fields.update(missing_fields)

        entries = self._entries
        contextual = [f for f in fields if resolve(f).is_contextual]
        if contextual:
            entries = {
                uid: entry
                for uid, entry in entries.items()
                if all(
                    matching_fields(EntryContext.from_entry(entry), f)
                    for f in contextual
                )
            }
        if any(resolve(f).field in {"step", "step_label"} for f in fields):
            entries = dict(
                sorted(
                    entries.items(), key=lambda item: item[1]["fields"].get("step", 0)
                )
            )
        return apply_filter(entries, value_filter) if value_filter else entries

    def get_field_data(
        self, field: FieldAddress
    ) -> tuple[list[Any], DataType, DataShape]:
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

            contexts = dict(existing.get("field_contexts", {})) if existing else {}
            contexts.update(item.get("field_contexts", {}))
            self._entries[uid] = Entry(fields=merged_fields, field_contexts=contexts)
