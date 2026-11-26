from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

from diffract.session import Session
from diffract.session.resolver import FieldSelector, render_label, resolve
from diffract.viz.data import DataProvider, Entry, FieldRef
from diffract.viz.data.extraction import field_label
from diffract.viz.plots.base.overlay import Overlay
from diffract.viz.styling import apply_theme

from .configurator import Configurator

if TYPE_CHECKING:
    from diffract.viz.styling import Theme


@dataclass(kw_only=True)
class Plot(ABC):
    """A configurable plot that can render itself given a Session."""

    title: str | None = None
    value_filter: dict[str, tuple[str, Any]] | None = None

    _traces_data: dict[str, dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )
    _field_labels: dict[str | FieldSelector, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _theme: Theme | None = field(default=None, repr=False, compare=False)

    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render the plot using data from the session.

        Args:
            session: The diffract session containing data to plot.
            theme: Optional theme for styling. If provided, palettes from theme
                   are used for colors/symbols/dashes, and figure-level styling
                   is applied after configurators.

        Returns:
            A Plotly Figure object.
        """
        self._theme = theme
        data_provider = DataProvider(session)
        entries = self._collect_entries(data_provider)
        self._field_labels = {
            f: field_label(entries, f) for f in self._collect_fields_to_fetch()
        }
        self._traces_data = self._build_traces_data(entries)
        fig: go.Figure = self._build_figure()
        self._add_overlay_traces_by_mro(fig)
        self._configure_by_mro(fig)

        if theme is not None:
            apply_theme(fig, theme)

        return fig

    def _collect_entries(self, data_provider: DataProvider) -> dict[str, Entry]:
        """Collect all entries needed for the plot."""
        fields_to_fetch = self._collect_fields_to_fetch()
        return data_provider.fetch(fields_to_fetch, value_filter=self.value_filter)

    def _collect_fields_to_fetch(self) -> list[str]:
        """Collect all field names needed for the plot.

        Walks the MRO and inspects dataclass fields, collecting the ``field``
        attribute from every value that is a :class:`FieldRef`.
        """
        result: list[str] = []
        seen: set[str] = set()

        for cls in type(self).mro():
            if not hasattr(cls, "__dataclass_fields__"):
                continue
            for f in fields(cls):
                value = getattr(self, f.name, None)
                if isinstance(value, FieldRef) and value.field not in seen:
                    seen.add(value.field)
                    result.append(value.field)

        return result

    def _field_label(self, ref: FieldRef | str | FieldSelector) -> str:
        address = ref.field if isinstance(ref, FieldRef) else ref
        if address in self._field_labels:
            return self._field_labels[address]
        return render_label(resolve(address))

    @abstractmethod
    def _build_traces_data(
        self, entries: dict[str, Entry]
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_figure(self) -> go.Figure:
        raise NotImplementedError

    def _add_overlay_traces_by_mro(self, fig: go.Figure) -> None:
        """Run each distinct overlay hook once in MRO order."""
        seen = {Overlay.add_overlay_traces}
        for cls in type(self).mro():
            if not issubclass(cls, Overlay):
                continue
            hook = cls.add_overlay_traces
            if hook not in seen:
                seen.add(hook)
                hook(self, fig)

    def _configure_by_mro(self, fig: go.Figure) -> None:
        """Apply configurator mixins in MRO order."""
        for cls in type(self).mro():
            if cls is Configurator:
                continue
            if issubclass(cls, Configurator):
                cls.configure(self, fig)
