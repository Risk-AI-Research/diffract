from __future__ import annotations

from abc import ABC, abstractmethod

import plotly.graph_objects as go


class Configurator(ABC):
    """A mixin that can configure a Plotly figure.

    In `viz`, configurators are intended to be used as mixins on a plot
    class and invoked in MRO order by `Plot._configure_by_mro(...)`.
    """

    @abstractmethod
    def configure(self, fig: go.Figure) -> None:
        """Apply configuration to the given figure."""
        raise NotImplementedError
