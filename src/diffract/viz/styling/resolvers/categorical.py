from __future__ import annotations

from dataclasses import dataclass

from diffract.viz.data import DataType, Entry, FieldRef
from diffract.viz.data.extraction import get_field_data


@dataclass
class CategoricalPropertyResolver:
    """Resolves categorical FieldRef values with optional mapping to candidates."""

    candidates: list[str] | None = None

    def resolve(
        self,
        source: FieldRef | str | None,
        entries: dict[str, Entry] | None = None,
    ) -> str | list[str] | None:
        """Resolve a categorical source into a value or list of values.

        Args:
            source: A literal string, a FieldRef, or None.
            entries: Entries used to resolve a FieldRef, if given.

        Returns:
            The literal string, the resolved per-entry values, or None.
        """
        match source:
            case None:
                return None
            case str():
                return source
            case FieldRef():
                return self._resolve_field_ref(source, entries)

    def _resolve_field_ref(
        self,
        ref: FieldRef,
        entries: dict[str, Entry] | None,
    ) -> list[str] | None:
        if entries is None:
            return None

        values, data_type, _ = get_field_data(entries, ref.field)
        if ref.data_type is not None:
            data_type = ref.data_type

        if data_type != DataType.CATEGORICAL:
            raise ValueError(
                f"Field '{ref.field}' must be categorical for this property, "
                f"got {data_type.name}"
            )

        str_values = [str(v) for v in values]

        if self.candidates is None:
            return str_values

        unique = list(dict.fromkeys(str_values))
        order_indices = ref.ordering.argsort(unique)
        sorted_unique = [unique[i] for i in order_indices]
        mapping = {
            v: self.candidates[i % len(self.candidates)]
            for i, v in enumerate(sorted_unique)
        }

        return [mapping[v] for v in str_values]
