"""Flexible color mapping for plots.

Maps metadata keys (model_id, layer_id, head_id, etc.) to colors using
either discrete colormaps or continuous colorscales.

Example:
    from diffract.viz.colors import ColorMapper
    from diffract.viz.themes import DEFAULT_THEME

    mapper = ColorMapper(theme=DEFAULT_THEME)
    color = mapper.get_color("model_id", "gpt2", all_values=["gpt2", "llama"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from diffract.core.utils import imports as import_utils

if TYPE_CHECKING:  # pragma: no cover
    from diffract.viz.themes import Theme


# Default colorscales for known metadata keys
_DEFAULT_COLORSCALES: dict[str, str] = {
    "model_id": "discrete",
    "kind": "discrete",
    "ptype": "discrete",
    "parameter_type": "discrete",
}


@dataclass(slots=True)
class ColorMapper:
    """Maps metadata keys to colors with sensible defaults.

    For discrete keys (model_id, kind), cycles through theme.discrete_colormap.
    For continuous keys (layer_id, head_id), samples from colorscales.
    """

    theme: Theme | None = None
    overrides: dict[str, str | list[str]] = field(default_factory=dict)

    def _get_discrete_cmap(self) -> list[str]:
        if self.theme is not None:
            return self.theme.discrete_colormap
        return [
            "navy",
            "crimson",
            "green",
            "chocolate",
            "orange",
            "violet",
            "purple",
            "blue",
            "grey",
        ]

    def _get_colorscale_for_key(self, key: str) -> str:
        if key in self.overrides:
            v = self.overrides[key]
            if isinstance(v, str):
                return v
            # If list, treat as discrete
            return "discrete"

        if key in _DEFAULT_COLORSCALES:
            cs = _DEFAULT_COLORSCALES[key]
            if cs == "discrete":
                return "discrete"
            return cs

        # Unknown keys default to discrete
        return "discrete"

    def get_colorscale(self, key: str) -> str | list[str]:
        """Return colorscale name or discrete color list for a key."""
        cs = self._get_colorscale_for_key(key)
        if cs == "discrete":
            if key in self.overrides and isinstance(self.overrides[key], list):
                return self.overrides[key]
            return self._get_discrete_cmap()
        return cs

    def _get_param_type_color(self, value: Any) -> str | None:
        """Check if value is a known param type and return its color."""
        if self.theme is None:
            return None
        param_colors = getattr(self.theme, "param_type_colors", None)
        if not isinstance(param_colors, dict):
            return None
        str_val = str(value).lower()
        # Direct match
        if str_val in param_colors:
            return param_colors[str_val]
        # Partial match (e.g., "layers.0.ffn" matches "ffn")
        for ptype, color in param_colors.items():
            if ptype in str_val:
                return color
        return None

    def get_color(
        self,
        key: str,
        value: Any,
        all_values: list[Any],
    ) -> str:
        """Return a color for a specific (key, value) pair.

        Args:
            key: Metadata key (e.g., "model_id", "layer_id").
            value: The specific value to color.
            all_values: All unique values for this key (used for indexing).

        Returns:
            A color string (hex or named color).
        """
        # Check if value is a known param type (for parameter_type, kind, ptype keys)
        if key in ("parameter_type", "kind", "ptype", "parameter_name"):
            param_color = self._get_param_type_color(value)
            if param_color:
                return param_color

        cs = self._get_colorscale_for_key(key)

        if cs == "discrete":
            cmap = self.get_colorscale(key)
            if isinstance(cmap, list):
                try:
                    idx = list(all_values).index(value)
                except ValueError:
                    idx = 0
                return cmap[idx % len(cmap)]
            # Fallback
            return self._get_discrete_cmap()[0]

        # Continuous colorscale: sample based on position in sorted values
        return self._sample_colorscale(cs, value, all_values)

    def _sample_colorscale(
        self, colorscale: str, value: Any, all_values: list[Any]
    ) -> str:
        """Sample a color from a continuous colorscale."""
        px_colors = import_utils.require("plotly.colors")

        # Normalize value to [0, 1] range
        try:
            sorted_vals = sorted(set(all_values))
            if len(sorted_vals) <= 1:
                t = 0.5
            else:
                idx = sorted_vals.index(value)
                t = idx / (len(sorted_vals) - 1)
        except (ValueError, TypeError):
            t = 0.5

        # Get colorscale colors
        try:
            cs_colors = px_colors.get_colorscale(colorscale)
        except ValueError:
            # Fallback to viridis
            cs_colors = px_colors.get_colorscale("viridis")

        return self._interpolate_colorscale(cs_colors, t)

    def _interpolate_colorscale(
        self, colorscale: list[tuple[float, str]], t: float
    ) -> str:
        """Interpolate a color from a plotly colorscale at position t in [0, 1]."""
        px_colors = import_utils.require("plotly.colors")

        t = max(0.0, min(1.0, t))

        # Find the two surrounding colors
        lower = colorscale[0]
        upper = colorscale[-1]
        for i in range(len(colorscale) - 1):
            if colorscale[i][0] <= t <= colorscale[i + 1][0]:
                lower = colorscale[i]
                upper = colorscale[i + 1]
                break

        if lower[0] == upper[0]:
            return lower[1]

        # Interpolate
        local_t = (t - lower[0]) / (upper[0] - lower[0])

        # Parse colors
        def parse_rgb(c: str) -> tuple[int, int, int]:
            if c.startswith("rgb"):
                nums = c.replace("rgb(", "").replace(")", "").split(",")
                return int(nums[0]), int(nums[1]), int(nums[2])
            # Try hex
            if c.startswith("#"):
                c = c.lstrip("#")
                return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            # Named color - use plotly to convert
            try:
                hex_c = px_colors.label_rgb(px_colors.convert_to_RGB_255(c))
                return parse_rgb(hex_c)
            except (TypeError, ValueError):
                return 128, 128, 128

        r1, g1, b1 = parse_rgb(lower[1])
        r2, g2, b2 = parse_rgb(upper[1])

        r = int(r1 + (r2 - r1) * local_t)
        g = int(g1 + (g2 - g1) * local_t)
        b = int(b1 + (b2 - b1) * local_t)

        return f"rgb({r},{g},{b})"

    def get_colors_for_values(self, key: str, values: list[Any]) -> dict[Any, str]:
        """Get a color mapping for all values of a key."""
        unique = list(dict.fromkeys(values))  # Preserve order, remove duplicates
        return {v: self.get_color(key, v, unique) for v in unique}

    def get_symbol(self, index: int) -> str:
        """Get a marker symbol by index, cycling through available symbols."""
        if self.theme is not None:
            symbols = self.theme.marker_symbols
        else:
            symbols = [
                "circle",
                "square",
                "triangle-up",
                "diamond",
                "cross",
                "x",
                "star",
            ]
        return symbols[index % len(symbols)]

    def get_symbol_for_value(self, _key: str, value: Any, all_values: list[Any]) -> str:
        """Get a marker symbol for a specific (key, value) pair."""
        try:
            idx = list(all_values).index(value)
        except ValueError:
            idx = 0
        return self.get_symbol(idx)

    def get_dash(self, index: int) -> str:
        """Get a line dash by index, cycling through available dashes."""
        if self.theme is not None and getattr(self.theme, "line_dashes", None):
            dashes = self.theme.line_dashes
        else:
            dashes = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]
        return dashes[index % len(dashes)]

    def get_dash_for_value(self, _key: str, value: Any, all_values: list[Any]) -> str:
        """Get a line dash for a specific (key, value) pair."""
        try:
            idx = list(all_values).index(value)
        except ValueError:
            idx = 0
        return self.get_dash(idx)


def get_symbol(index: int, theme: Theme | None = None) -> str:
    """Get a marker symbol by index, cycling through available symbols."""
    if theme is not None:
        symbols = theme.marker_symbols
    else:
        symbols = ["circle", "square", "triangle-up", "diamond", "cross", "x", "star"]
    return symbols[index % len(symbols)]
