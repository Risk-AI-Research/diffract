"""Hydra-configurable rendering utilities for `diffract.viz`.

This module keeps rendering orchestration separate from individual plot classes.
It supports two rendering modes:

1. Already-constructed plot object via ``render(...)``.
2. Hydra YAML config via ``render_from_config(...)``.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from diffract.core.utils import imports as import_utils
from diffract.viz.styling.sources import StyleLiteralKind, is_style_literal

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    from diffract.session import Session
    from diffract.viz.styling import Theme

_PLOTLY_GO = "plotly.graph_objects"
_VIZ_EXTRA_HINT = (
    "diffract.viz requires the viz extra. "
    'Install it with: pip install "diffract-core[viz]"'
)
_THEME_STRUCTURED_KEYS = frozenset(
    {"layout", "typography", "background", "axes", "legend", "colorbar", "palettes"}
)


def require_plotly_go() -> Any:
    """Import and return ``plotly.graph_objects``."""
    return import_utils.require(_PLOTLY_GO)


class Plot(Protocol):
    """Protocol for renderable plot objects."""

    def render(self, session: Session, theme: Theme | None = None) -> go.Figure:
        """Render a figure from a session."""
        ...


def render(
    plot: Plot,
    *,
    session: Session,
    theme: Theme | None = None,
) -> go.Figure:
    """Render a Plotly figure from a plot object."""
    go = require_plotly_go()

    if _plot_supports_theme_kwarg(plot):
        fig = plot.render(session, theme=theme)
    else:
        fig = plot.render(session)
        if theme is not None:
            from diffract.viz.styling import apply_theme

            fig = apply_theme(fig, theme)

    if not isinstance(fig, go.Figure):
        raise TypeError("Plot.render() must return plotly.graph_objects.Figure")

    return fig


def load_theme(theme_path: str | Path) -> Theme:
    """Load a `viz` Theme from YAML."""
    try:
        yaml = import_utils.require("yaml")
    except ImportError as e:
        raise ImportError(_VIZ_EXTRA_HINT) from e

    path = Path(theme_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Theme YAML must be a mapping, got {type(data).__name__}")

    return _theme_from_dict(data)


def render_from_config(
    *,
    session: Session,
    config_path: str | Path,
    overrides: list[str] | None = None,
    theme: Theme | None = None,
    theme_path: str | Path | None = None,
) -> go.Figure:
    """Load a Hydra YAML config and render a figure."""
    try:
        hydra = import_utils.require("hydra")
        instantiate = import_utils.require("hydra.utils").instantiate
        omega_conf = import_utils.require("omegaconf").OmegaConf
    except ImportError as e:
        raise ImportError(_VIZ_EXTRA_HINT) from e
    compose = hydra.compose
    initialize_config_dir = hydra.initialize_config_dir

    cfg_path = Path(config_path).expanduser().resolve()
    config_dir = str(cfg_path.parent)
    config_name = cfg_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    omega_conf.resolve(cfg)

    plot_cfg = cfg.get("plot")
    if plot_cfg is None:
        raise ValueError("Config must define a top-level 'plot' node")

    plot = instantiate(plot_cfg, _convert_="object")
    _coerce_field_refs(plot)

    resolved_theme = theme
    if resolved_theme is None and theme_path is not None:
        resolved_theme = load_theme(theme_path)

    fig = render(plot, session=session, theme=resolved_theme)

    fig_cfg = cfg.get("config")
    if fig_cfg is not None:
        fig_cfg_plain = omega_conf.to_container(fig_cfg, resolve=True)
        if not isinstance(fig_cfg_plain, dict):
            raise TypeError("Config node 'config' must be a dict-like mapping")
        fig.update(fig_cfg_plain)

    return fig


def _coerce_field_refs(value: Any) -> None:
    """Recursively coerce string values to `FieldRef` where annotated.

    Two deterministic, config-time rules:

    - Annotations that contain ``FieldRef`` and do not legitimately accept
      a plain string always coerce strings to ``FieldRef``.
    - Style properties annotated with a :class:`StyleLiteralKind` (colors,
      symbols, dashes) keep strings that are valid plotly literals of that
      kind and coerce everything else to ``FieldRef``. Literals win over
      identically named fields; an explicit ``FieldRef`` (e.g. refs-style
      configs) is the escape hatch.
    """
    if is_dataclass(value):
        cls = type(value)
        try:
            type_hints = get_type_hints(cls, include_extras=True)
        except (AttributeError, NameError, TypeError):
            type_hints = {}

        for f in fields(value):
            item = getattr(value, f.name)
            annotation = type_hints.get(f.name, f.type)
            literal_kind = _style_literal_kind(annotation)
            if isinstance(item, str) and literal_kind is not None:
                if not _is_style_literal(item, literal_kind):
                    from diffract.viz.data import FieldRef

                    setattr(value, f.name, FieldRef(field=item))
                continue
            if isinstance(item, str) and _should_coerce_string_to_field_ref(annotation):
                from diffract.viz.data import FieldRef

                setattr(value, f.name, FieldRef(field=item))
                continue
            _coerce_field_refs(item)
        return

    if isinstance(value, list):
        for item in value:
            _coerce_field_refs(item)
        return

    if isinstance(value, tuple):
        for item in value:
            _coerce_field_refs(item)
        return

    if isinstance(value, dict):
        for item in value.values():
            _coerce_field_refs(item)


def _contains_field_ref(annotation: Any) -> bool:
    if annotation is None:
        return False

    if isinstance(annotation, str):
        return "FieldRef" in annotation

    from diffract.viz.data import FieldRef

    if annotation is FieldRef:
        return True

    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        return any(_contains_field_ref(arg) for arg in get_args(annotation))

    return False


def _annotation_allows_plain_string(annotation: Any) -> bool:
    if annotation is None:
        return False

    if isinstance(annotation, str):
        return "str" in annotation

    if annotation is str:
        return True

    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        return any(_annotation_allows_plain_string(arg) for arg in get_args(annotation))

    return False


def _should_coerce_string_to_field_ref(annotation: Any) -> bool:
    return _contains_field_ref(annotation) and not _annotation_allows_plain_string(
        annotation
    )


def _style_literal_kind(annotation: Any) -> StyleLiteralKind | None:
    if annotation is None:
        return None

    if isinstance(annotation, str):
        # Annotations are often unresolvable strings here (TYPE_CHECKING-only
        # names in the class hierarchy defeat get_type_hints), so match the
        # source aliases by name.
        by_alias = {
            "ColorSource": StyleLiteralKind.COLOR,
            "SymbolSource": StyleLiteralKind.SYMBOL,
            "DashSource": StyleLiteralKind.DASH,
        }
        return next(
            (kind for alias, kind in by_alias.items() if alias in annotation),
            None,
        )

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        for metadata in args[1:]:
            if isinstance(metadata, StyleLiteralKind):
                return metadata
        return _style_literal_kind(args[0])

    if get_origin(annotation) in (UnionType, Union):
        for arg in get_args(annotation):
            kind = _style_literal_kind(arg)
            if kind is not None:
                return kind

    return None


# The probe lives in styling.sources; keep the private name for local callers.
_is_style_literal = is_style_literal


def _plot_supports_theme_kwarg(plot: Plot) -> bool:
    try:
        signature = inspect.signature(plot.render)
    except (TypeError, ValueError):
        # If signature introspection fails, keep permissive behavior.
        return True

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "theme":
            return True

    return False


def _theme_from_dict(data: dict[str, Any]) -> Theme:
    from diffract.viz.styling.palettes import (
        DefaultColorPalette,
        DefaultDashPalette,
        DefaultSymbolPalette,
        PaletteBundle,
    )
    from diffract.viz.styling.theme import (
        AxesStyle,
        BackgroundStyle,
        ColorbarStyle,
        LayoutStyle,
        LegendStyle,
        Theme,
        TypographyStyle,
    )

    kwargs: dict[str, Any] = {}

    if any(key in data for key in _THEME_STRUCTURED_KEYS):
        if "layout" in data:
            kwargs["layout"] = LayoutStyle(**_require_mapping(data["layout"], "layout"))
        if "typography" in data:
            kwargs["typography"] = TypographyStyle(
                **_require_mapping(data["typography"], "typography")
            )
        if "background" in data:
            kwargs["background"] = BackgroundStyle(
                **_require_mapping(data["background"], "background")
            )
        if "axes" in data:
            kwargs["axes"] = AxesStyle(**_require_mapping(data["axes"], "axes"))
        if "legend" in data:
            kwargs["legend"] = LegendStyle(**_require_mapping(data["legend"], "legend"))
        if "colorbar" in data:
            kwargs["colorbar"] = ColorbarStyle(
                **_require_mapping(data["colorbar"], "colorbar")
            )
        if "palettes" in data:
            palettes_dict = _require_mapping(data["palettes"], "palettes")
            palette_kwargs: dict[str, Any] = {}

            color_cfg = palettes_dict.get("color")
            if color_cfg is not None:
                if isinstance(color_cfg, list):
                    palette_kwargs["color"] = DefaultColorPalette(
                        _colors=list(color_cfg)
                    )
                else:
                    color_dict = _require_mapping(color_cfg, "palettes.color")
                    colors = color_dict.get("colors", color_dict.get("_colors"))
                    if colors is not None:
                        palette_kwargs["color"] = DefaultColorPalette(
                            _colors=list(colors)
                        )

            symbols_cfg = palettes_dict.get("symbols")
            if symbols_cfg is not None:
                if isinstance(symbols_cfg, list):
                    palette_kwargs["symbols"] = DefaultSymbolPalette(
                        _symbols=list(symbols_cfg)
                    )
                else:
                    symbols_dict = _require_mapping(symbols_cfg, "palettes.symbols")
                    symbols = symbols_dict.get("symbols", symbols_dict.get("_symbols"))
                    if symbols is not None:
                        palette_kwargs["symbols"] = DefaultSymbolPalette(
                            _symbols=list(symbols)
                        )

            dashes_cfg = palettes_dict.get("dashes")
            if dashes_cfg is not None:
                if isinstance(dashes_cfg, list):
                    palette_kwargs["dashes"] = DefaultDashPalette(
                        _dashes=list(dashes_cfg)
                    )
                else:
                    dashes_dict = _require_mapping(dashes_cfg, "palettes.dashes")
                    dashes = dashes_dict.get("dashes", dashes_dict.get("_dashes"))
                    if dashes is not None:
                        palette_kwargs["dashes"] = DefaultDashPalette(
                            _dashes=list(dashes)
                        )

            kwargs["palettes"] = PaletteBundle(**palette_kwargs)

        return Theme(**kwargs)

    # Flat legacy-like format for convenience.
    layout_kwargs: dict[str, Any] = {}
    typography_kwargs: dict[str, Any] = {}
    background_kwargs: dict[str, Any] = {}
    axes_kwargs: dict[str, Any] = {}
    legend_kwargs: dict[str, Any] = {}
    colorbar_kwargs: dict[str, Any] = {}
    palettes_kwargs: dict[str, Any] = {}

    if "width" in data:
        layout_kwargs["width"] = data["width"]
    if "height" in data:
        layout_kwargs["height"] = data["height"]
    if "margin" in data:
        layout_kwargs["margin"] = _require_mapping(data["margin"], "margin")

    if "font_family" in data:
        typography_kwargs["font_family"] = data["font_family"]
    if "title_font_size" in data:
        typography_kwargs["title_font_size"] = data["title_font_size"]
    if "label_font_size" in data:
        typography_kwargs["label_font_size"] = data["label_font_size"]
    if "tick_font_size" in data:
        typography_kwargs["tick_font_size"] = data["tick_font_size"]

    if "background_color" in data:
        background_kwargs["plot_bgcolor"] = data["background_color"]
    if "paper_bgcolor" in data:
        background_kwargs["paper_bgcolor"] = data["paper_bgcolor"]

    if "grid_color" in data:
        axes_kwargs["grid_color"] = data["grid_color"]
    if "border_color" in data:
        axes_kwargs["line_color"] = data["border_color"]
    if "show_borders" in data:
        axes_kwargs["show_line"] = bool(data["show_borders"])
    if "mirror_axes" in data:
        axes_kwargs["mirror"] = bool(data["mirror_axes"])

    show_x_grid = data.get("show_x_grid")
    show_y_grid = data.get("show_y_grid")
    if show_x_grid is not None and show_y_grid is not None:
        sx = bool(show_x_grid)
        sy = bool(show_y_grid)
        if sx != sy:
            raise ValueError(
                "Flat theme format does not support different values for "
                "'show_x_grid' and 'show_y_grid'. Use structured 'axes' theme format."
            )
        axes_kwargs["show_grid"] = sx
    elif show_x_grid is not None:
        axes_kwargs["show_grid"] = bool(show_x_grid)
    elif show_y_grid is not None:
        axes_kwargs["show_grid"] = bool(show_y_grid)

    if "legend_bgcolor" in data:
        legend_kwargs["bgcolor"] = data["legend_bgcolor"]
    if "legend_border_color" in data:
        legend_kwargs["border_color"] = data["legend_border_color"]
    if "legend_border_width" in data:
        legend_kwargs["border_width"] = data["legend_border_width"]
    if "legend_font_size" in data:
        legend_kwargs["font_size"] = data["legend_font_size"]

    if "colorbar_orientation" in data:
        colorbar_kwargs["orientation"] = data["colorbar_orientation"]
    if "colorbar_x" in data:
        colorbar_kwargs["x"] = data["colorbar_x"]
    if "colorbar_y" in data:
        colorbar_kwargs["y"] = data["colorbar_y"]
    if "colorbar_xanchor" in data:
        colorbar_kwargs["xanchor"] = data["colorbar_xanchor"]
    if "colorbar_yanchor" in data:
        colorbar_kwargs["yanchor"] = data["colorbar_yanchor"]
    if "colorbar_thickness" in data:
        colorbar_kwargs["thickness"] = data["colorbar_thickness"]
    if "colorbar_len" in data:
        colorbar_kwargs["len"] = data["colorbar_len"]

    if "discrete_colormap" in data:
        palettes_kwargs["color"] = DefaultColorPalette(
            _colors=list(data["discrete_colormap"])
        )
    if "marker_symbols" in data:
        palettes_kwargs["symbols"] = DefaultSymbolPalette(
            _symbols=list(data["marker_symbols"])
        )
    if "line_dashes" in data:
        palettes_kwargs["dashes"] = DefaultDashPalette(
            _dashes=list(data["line_dashes"])
        )

    if layout_kwargs:
        kwargs["layout"] = LayoutStyle(**layout_kwargs)
    if typography_kwargs:
        kwargs["typography"] = TypographyStyle(**typography_kwargs)
    if background_kwargs:
        kwargs["background"] = BackgroundStyle(**background_kwargs)
    if axes_kwargs:
        kwargs["axes"] = AxesStyle(**axes_kwargs)
    if legend_kwargs:
        kwargs["legend"] = LegendStyle(**legend_kwargs)
    if colorbar_kwargs:
        kwargs["colorbar"] = ColorbarStyle(**colorbar_kwargs)
    if palettes_kwargs:
        kwargs["palettes"] = PaletteBundle(**palettes_kwargs)

    return Theme(**kwargs)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"Theme field '{field_name}' must be a mapping, got {type(value).__name__}"
        )
    return dict(value)
