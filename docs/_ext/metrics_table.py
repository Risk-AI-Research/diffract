"""Sphinx extension that generates the metric catalog from the kernel registry.

The published metric catalog is a projection of the live kernel registry, so it
can never drift from the registered kernels. At build time this extension
introspects the registry through its public API and writes a MyST page of
``list-table`` blocks grouped by category. The generated file is git-ignored and
regenerated on every build.

A kernel with no display formula -- neither a ``:math:`` role in its docstring
nor a per-name ``KernelInfo.formula`` override -- raises a Sphinx warning, which
the ``-W`` build turns into an error. Catalog completeness is therefore enforced
by the build itself, on top of the warning-free gate.

The build environment must provide the ``taichi`` extra: the three ``*_p_value``
kernels register only when the accelerated fitter is importable, and a build
without it would silently drop them from the catalog.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from sphinx.util import logging

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.config import Config

    from diffract.core.compute.registry import KernelRegistry

logger = logging.getLogger(__name__)

_MATH_ROLE = re.compile(r":math:`([^`]+)`")

# Source module basename -> (catalog category, handwritten formula page).
_CATEGORY: dict[str, tuple[str, str]] = {
    "mat_properties": ("Matrix properties", "spectral"),
    "mat_decomposition": ("Spectral decomposition", "spectral"),
    "norms": ("Norms", "norms"),
    "ranks": ("Ranks", "ranks"),
    "heavy_tailed": ("Heavy-tailed fits", "heavy_tailed"),
    "marchenko_pastur": ("Marchenko-Pastur", "rmt"),
    "tracy_widom": ("Tracy-Widom", "rmt"),
    "alignment": ("Alignment (cross-model)", "alignment"),
    "model_quality": ("Model quality", "quality"),
}

# Category render order (matches the order of the handwritten pages).
_CATEGORY_ORDER = [
    "Matrix properties",
    "Spectral decomposition",
    "Norms",
    "Ranks",
    "Heavy-tailed fits",
    "Marchenko-Pastur",
    "Tracy-Widom",
    "Alignment (cross-model)",
    "Model quality",
]

# Source/dependency order of the kernel modules, so a category spanning two
# modules (matrix properties + decomposition, MP + TW) reads base-to-derived.
_MODULE_ORDER = [
    "mat_properties",
    "mat_decomposition",
    "norms",
    "ranks",
    "heavy_tailed",
    "marchenko_pastur",
    "tracy_widom",
    "alignment",
    "model_quality",
]

_TAICHI_MARKER = "requires the taichi extra"
_OUTPUT = Path("reference/metrics/_generated_table.md")


def _build_registry() -> KernelRegistry:
    """Return a registry populated with the built-in kernels."""
    from diffract.core.compute.containers import ComputeSingletonContainer
    from diffract.core.compute.decorator import register_default_kernels

    container = ComputeSingletonContainer()
    container.wire(modules=["diffract.core.compute.decorator"])
    registry = container.kernel_registry()
    register_default_kernels(registry)
    return registry


def _formula(registry: KernelRegistry, name: str) -> str | None:
    """Resolve a kernel's display formula (LaTeX), or None if it has none.

    Precedence: a per-name ``KernelInfo.formula`` override wins over the
    ``:math:`` role parsed from the kernel docstring.
    """
    override = registry.get_kernel_info(name).formula
    if override:
        return override.strip()

    doc = registry.get_kernel_implementation(name).__doc__ or ""
    match = _MATH_ROLE.search(doc)
    return match.group(1).strip() if match else None


def _category(registry: KernelRegistry, name: str) -> tuple[str, str]:
    """Return the (category, page) a kernel belongs to from its source module."""
    module = registry.get_kernel_implementation(name).__module__ or ""
    basename = module.rsplit(".", 1)[-1]
    return _CATEGORY.get(basename, ("Other", "index"))


def _source_pos(registry: KernelRegistry, name: str) -> tuple[int, int, str]:
    """Sort key grouping a category by source order (kernel family), then name.

    Stacked registrations share one implementation, so every name of a family
    sorts together by the body's source line; the module index keeps modules
    that share a category in dependency order rather than alphabetically.
    """
    impl = registry.get_kernel_implementation(name)
    impl = getattr(impl, "__wrapped__", impl)
    module = (getattr(impl, "__module__", "") or "").rsplit(".", 1)[-1]
    line = getattr(getattr(impl, "__code__", None), "co_firstlineno", 0)
    order = (
        _MODULE_ORDER.index(module) if module in _MODULE_ORDER else len(_MODULE_ORDER)
    )
    # Within a family (same body/line), the real quantity precedes its
    # randomized-null counterpart, then alphabetical.
    return (order, line, "rand" in name, name)


def _is_taichi_only(registry: KernelRegistry, name: str) -> bool:
    """True when the kernel is documented as requiring the taichi extra."""
    doc = registry.get_kernel_implementation(name).__doc__ or ""
    return _TAICHI_MARKER in doc


def _code(values: tuple[str, ...]) -> str:
    """Render a tuple of field/config tokens as inline code, or an em dash."""
    return ", ".join(f"`{value}`" for value in values) if values else "—"


def _config_tokens(config: dict[str, object]) -> tuple[str, ...]:
    """Render a config mapping as ``key=value`` tokens."""
    return tuple(f"{key}={value}" for key, value in config.items())


def _row(registry: KernelRegistry, kernel: str) -> tuple[str, bool]:
    """Build one list-table row for a kernel, and whether it lacked a formula.

    One row per kernel: the produced field(s) are listed together so the
    formula appears once, rather than repeating across a multi-field kernel's
    rows. Returns the row text and a flag that is True when no formula was found.
    """
    formula = _formula(registry, kernel)
    missing = formula is None
    formula_cell = f"${formula}$" if formula else "—"
    level = registry.get_kernel_apply_level(kernel).name
    requires = _code(registry.get_fields_kernel_require(kernel))
    config = _code(_config_tokens(registry.get_kernel_config(kernel)))
    marker = " [^taichi]" if _is_taichi_only(registry, kernel) else ""
    fields = _code(registry.get_fields_kernel_produce(kernel))

    cells = [
        f"{fields}{marker}",
        f"`{kernel}`",
        formula_cell,
        level,
        requires,
        config,
    ]
    lines = "\n".join(
        f"     - {cell}" if index else f"   * - {cell}"
        for index, cell in enumerate(cells)
    )
    return lines, missing


def _render(registry: KernelRegistry) -> str:
    """Render the whole catalog as grouped MyST list-tables, one row per kernel."""
    buckets: dict[str, list[str]] = {c: [] for c in _CATEGORY_ORDER}
    buckets.setdefault("Other", [])
    for kernel in registry.list_kernels():
        category, _page = _category(registry, kernel)
        buckets.setdefault(category, []).append(kernel)

    missing_any = False
    out: list[str] = [
        "<!-- Generated by docs/_ext/metrics_table.py from the kernel registry.",
        "     Do not edit by hand; edit the kernel docstrings instead. -->",
        "",
    ]
    seen_taichi = False
    for category in [*_CATEGORY_ORDER, "Other"]:
        kernels = sorted(
            buckets.get(category, []), key=lambda k: _source_pos(registry, k)
        )
        if not kernels:
            continue
        # Resolve the handwritten page for the category header link.
        page_name = next((p for _m, (c, p) in _CATEGORY.items() if c == category), None)
        header = f"## [{category}]({page_name}.md)" if page_name else f"## {category}"
        out.append(header)
        out.append("")
        out.append("```{list-table}")
        out.append(":header-rows: 1")
        out.append(":class: metrics-catalog")
        out.append("")
        out.append("   * - Field(s)")
        out.append("     - Kernel")
        out.append("     - Formula")
        out.append("     - Level")
        out.append("     - Requires")
        out.append("     - Config")
        for kernel in kernels:
            row, missing = _row(registry, kernel)
            if missing:
                missing_any = True
                logger.warning(
                    "metric catalog: kernel %r has no display formula; add a "
                    "':math:` `' role to its docstring or a KernelInfo.formula "
                    "override.",
                    kernel,
                    type="metrics",
                    subtype="formula",
                )
            if _is_taichi_only(registry, kernel):
                seen_taichi = True
            out.append(row)
        out.append("```")
        out.append("")

    if seen_taichi:
        out.append(
            "[^taichi]: Registered only when the `taichi` extra is installed; "
            "the bootstrap p-value uses the accelerated fitter."
        )
        out.append("")

    if missing_any:
        logger.warning(
            "metric catalog is incomplete: one or more kernels have no display "
            "formula (see warnings above).",
            type="metrics",
            subtype="formula",
        )
    return "\n".join(out)


def _generate(app: Sphinx, _config: Config) -> None:
    """Write the generated catalog table before Sphinx reads the sources."""
    registry = _build_registry()
    kernels = registry.list_kernels()
    with_formula = sum(1 for name in kernels if _formula(registry, name))
    content = _render(registry)
    output = Path(app.srcdir) / _OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    logger.info(
        "metric catalog written to %s (%d/%d kernels have a formula)",
        output,
        with_formula,
        len(kernels),
    )


def setup(app: Sphinx) -> dict[str, object]:
    """Register the extension: generate the catalog at config-init time."""
    app.connect("config-inited", _generate)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
