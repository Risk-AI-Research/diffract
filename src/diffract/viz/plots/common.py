"""Common helpers for plots based on Session.get_results(dict).

Provides utilities for data fetching, grouping, type coercion, and jitter
overlays that are reused across multiple plot types.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from diffract.core.constants import CONTEXT_SEPARATOR, parse_contextual_field_name

if TYPE_CHECKING:  # pragma: no cover
    from diffract.core.data.nn.params.schema import ParameterType
    from diffract.session import Session


def as_float(value: Any) -> float | None:
    """Convert a scalar value to float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return float(value.item())
    return None


def as_int(value: Any) -> int | None:
    """Convert a scalar value to int, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        v = value.item()
        return as_int(v)
    return None


def sort_key(value: Any) -> tuple[int, str]:
    """Stable sort key for mixed scalar types (numbers first, then strings)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, f"{float(value):020.10f}")
    return (1, str(value))


def get_field_value(
    fields: dict[str, Any],
    base_name: str,
    *,
    model_id: str | None = None,
    parameter_name: str | None = None,
) -> Any:
    """Get field value by base name, matching contextual variants.

    Contextual fields have format `{base_name}@models[...]@params[...]`.
    This function looks for the base field first, then selects the best
    contextual variant based on the provided metadata.

    Selection priority:
    1. Base field (no context suffix)
    2. Contextual field that includes current parameter (by model_id/name)
    3. Among matching fields, prefer smaller context (more specific)

    Args:
        fields: Dict of field_name -> value from session results.
        base_name: Base field name to look for.
        model_id: Current parameter's model_id for context matching.
        parameter_name: Current parameter's name for context matching.

    Returns:
        Field value if found (base or contextual), None otherwise.
    """
    if base_name in fields:
        return fields[base_name]

    pattern = re.compile(rf"^{re.escape(base_name)}{re.escape(CONTEXT_SEPARATOR)}.+$")
    candidates: list[tuple[str, Any]] = []
    for key, value in fields.items():
        if pattern.match(key):
            candidates.append((key, value))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][1]

    # Multiple candidates: select best match based on context
    return _select_best_contextual_field(
        candidates, model_id=model_id, parameter_name=parameter_name
    )


def _select_best_contextual_field(
    candidates: list[tuple[str, Any]],
    *,
    model_id: str | None,
    parameter_name: str | None,
) -> Any:
    """Select best contextual field from multiple candidates.

    Priority:
    1. Fields whose context includes current parameter
    2. Among those, prefer smaller context size (more specific)
    3. Fallback: deterministic sort by field name
    """
    scored: list[tuple[int, int, str, Any]] = []

    for field_name, value in candidates:
        _, models, params = parse_contextual_field_name(field_name)

        # Check if current parameter is in context
        model_match = models is None or (model_id is not None and model_id in models)
        param_match = params is None or (
            parameter_name is not None and parameter_name in params
        )
        matches = model_match and param_match

        # Context size (smaller = more specific)
        context_size = (len(models) if models else 0) + (len(params) if params else 0)

        # Score: (matches, -context_size) — higher is better
        # matches=True (1) > matches=False (0)
        # smaller context_size is better, so we negate it
        scored.append((1 if matches else 0, -context_size, field_name, value))

    # Sort: highest score first, then by field name for determinism
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored[0][3]


def list_contextual_field_values(
    fields: dict[str, Any], base_name: str
) -> list[tuple[str, Any]]:
    """List all contextual field variants for a base name.

    Args:
        fields: Dict of field_name -> value from session results.
        base_name: Base field name to look for.

    Returns:
        List of (full_field_name, value) tuples for all matching fields.
    """
    result: list[tuple[str, Any]] = []
    pattern = re.compile(
        rf"^{re.escape(base_name)}({re.escape(CONTEXT_SEPARATOR)}.+)?$"
    )
    for key, value in fields.items():
        if pattern.match(key):
            result.append((key, value))
    return result


def density_scaled_jitter(
    *,
    y: np.ndarray,
    jitter: np.ndarray,
    window_scale: float = 0.4,
) -> np.ndarray:
    """Scale jitter by local point density (notebooks_src-style).

    Points in denser regions get larger jitter spread for better visibility.
    """
    if y.size <= 1:
        return jitter

    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        iqr = float(np.max(y) - np.min(y))
    if iqr <= 0:
        return jitter

    window = window_scale * iqr
    lower = y - window
    upper = y + window

    counts = np.asarray(
        [int(np.sum((y > lo) & (y < hi))) for lo, hi in zip(lower, upper, strict=True)],
        dtype=np.int64,
    )
    cmax = int(np.max(counts)) if counts.size else 0
    if cmax <= 1:
        return jitter

    norm = (counts - 1) / (cmax - 1)
    scale = np.log(norm * (np.e - 1.0) + 1.0)
    return jitter * scale


def fetch_data(
    session: Session,
    fields: list[str],
    *,
    parameter_uids: list[str] | None = None,
    parameter_names: list[str] | None = None,
    parameter_types: list[ParameterType] | None = None,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch data from session: compute fields and return dict results.

    This is a convenience wrapper around session.get_results().
    """
    return session.get_results(
        *fields,
        export_format="dict",
        parameter_uids=parameter_uids,
        parameter_names=parameter_names,
        parameter_types=parameter_types,
        model_ids=model_ids,
    ).scalars


def group_entries_by(
    results: dict[str, Any],
    group_by: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group result entries by a metadata key.

    Args:
        results: Dict from session.get_results(export_format="dict").
        group_by: Metadata key to group by (e.g., "model_id", "layer_id").

    Returns:
        Dict mapping group keys to lists of entries.
    """
    groups: dict[str, list[dict[str, Any]]] = {}

    for entry in results.values():
        meta = entry.get("metadata", {})

        if not group_by:
            key = "all"
        elif group_by == "parameter_name":
            key = str(meta.get("name", "unknown"))
        else:
            key = str(meta.get(group_by, "null"))

        groups.setdefault(key, []).append(entry)

    return groups


def extract_meta_value(entry: dict[str, Any], key: str) -> Any:
    """Extract a metadata value from an entry, handling special keys."""
    meta = entry.get("metadata", {})
    if key == "parameter_name":
        return meta.get("name")
    return meta.get(key)


def build_jitter_scatter(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    jitter_width: float = 0.25,
    jitter_offset: float = 0.0,
    jitter_seed: int = 42,
    density_scale: bool = True,
    marker_size: int = 4,
    opacity: float = 0.7,
    colors: np.ndarray | None = None,
    coloraxis: str | None = None,
    colorscale: str = "Viridis",
    name: str = "",
) -> dict[str, Any]:
    """Build a jitter scatter trace configuration.

    Returns a dict suitable for go.Scatter(**result).
    """
    from diffract.core.utils import imports as import_utils

    import_utils.require("plotly.graph_objects")

    rng = np.random.default_rng(jitter_seed)
    j = rng.uniform(-jitter_width, jitter_width, size=ys.size)

    if density_scale and ys.size > 1:
        j = density_scaled_jitter(y=ys, jitter=j)

    marker: dict[str, Any] = {"size": marker_size, "opacity": opacity}

    if colors is not None and colors.size == ys.size:
        marker["color"] = colors
        if coloraxis:
            marker["coloraxis"] = coloraxis
        else:
            marker["colorscale"] = colorscale

    return dict(
        x=xs + jitter_offset + j,
        y=ys,
        mode="markers",
        showlegend=False,
        marker=marker,
        name=name,
        meta={"trace_type": f"{name} jitter"},
    )


def collect_unique_meta_values(
    results: dict[str, Any],
    key: str,
) -> list[Any]:
    """Collect unique metadata values for a key, preserving order."""
    seen: dict[Any, None] = {}
    for entry in results.values():
        v = extract_meta_value(entry, key)
        if v is not None and v not in seen:
            seen[v] = None
    return list(seen.keys())
