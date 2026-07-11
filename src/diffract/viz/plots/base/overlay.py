from __future__ import annotations

from abc import ABC, abstractmethod

import plotly.graph_objects as go


class Overlay(ABC):
    """A mixin that can add overlay traces to a Plotly figure."""

    @abstractmethod
    def add_overlay_traces(self, fig: go.Figure) -> None:
        """Add overlay traces to the given figure."""
        raise NotImplementedError
