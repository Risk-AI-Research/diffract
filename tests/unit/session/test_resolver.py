"""Field addresses and their resolution in export, merge and erase."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import diffract
from diffract.session import Session
from diffract.session.resolver import FieldSelector, render_label, resolve
from diffract.viz.data.types import FieldRef

pytestmark = pytest.mark.unit


# ---------------- resolution of every address form ----------------


def test_plain_name_resolves_unconstrained() -> None:
    selector = resolve("stable_rank")
    assert selector == FieldSelector(field="stable_rank")
    assert not selector.is_contextual


def test_grammar_string_resolves_components() -> None:
    selector = resolve("agreement@models[m1,m2]@params[w1,w2]")
    assert selector.field == "agreement"
    assert selector.models == ("m1", "m2")
    assert selector.params == ("w1", "w2")
    assert selector.is_contextual


def test_grammar_string_with_single_context_part() -> None:
    assert resolve("metric@models[m1,m2]") == FieldSelector(
        field="metric", models=("m1", "m2")
    )
    assert resolve("metric@params[p1,p2,p3]") == FieldSelector(
        field="metric", params=("p1", "p2", "p3")
    )


def test_component_order_is_preserved_not_canonicalized() -> None:
    # Only the mint and the renderer sort; resolution reports as written.
    assert resolve("k@models[m2,m1]").models == ("m2", "m1")


def test_non_canonical_suffix_contributes_no_context() -> None:
    selector = resolve("metric@garbage")
    assert selector.field == "metric"
    assert not selector.is_contextual


def test_field_ref_resolves_through_its_field() -> None:
    assert resolve(FieldRef(field="frob_norm")) == FieldSelector(field="frob_norm")
    assert resolve(FieldRef(field="k@models[a,b]")).models == ("a", "b")


def test_selector_input_is_idempotent() -> None:
    selector = FieldSelector(field="k", models=("a",))
    assert resolve(selector) is selector


def test_unresolvable_address_type_raises() -> None:
    with pytest.raises(TypeError, match="Cannot resolve"):
        resolve(42)


def test_selector_is_hashable_and_frozen() -> None:
    selector = resolve("k@models[a,b]")
    assert {selector: 1}[resolve("k@models[a,b]")] == 1
    with pytest.raises(AttributeError):
        selector.field = "other"  # type: ignore[misc]


def test_reserved_axes_default_to_none() -> None:
    selector = resolve("k@models[a]")
    assert selector.steps is None
    assert selector.roles is None


def test_legacy_input_matches_a_structured_set_selector() -> None:
    old = resolve("agreement@models[m2,m1]@params[w]")
    assert old.matches(FieldSelector("agreement", models=("m1", "m2"), params=("w",)))
    assert not old.matches(FieldSelector("agreement", models=("m1",), params=("w",)))


def test_render_label_sorts_context_members() -> None:
    label = render_label(
        FieldSelector(field="k", models=("m2", "m1"), params=("p2", "p1"))
    )
    assert label == "k [models=m1, m2, parameters=p1, p2]"


def test_render_label_of_plain_selector_is_the_field() -> None:
    assert render_label(FieldSelector(field="frob_norm")) == "frob_norm"


# ---------------- the symbol/import gate ----------------

_GATED_SYMBOLS = frozenset(
    {
        "parse_contextual_field_name",
        "format_contextual_field_name",
        "format_field_suffix",
        "format_context_part",
        "CONTEXT_SEPARATOR",
        "MODELS_CONTEXT_PREFIX",
        "PARAMS_CONTEXT_PREFIX",
    }
)


def _dotted_name(node: ast.expr) -> str | None:
    """The dotted path an expression spells, or None if it is not one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolver_module_aliases(tree: ast.Module) -> set[str]:
    """Local names a module binds to the resolver module itself."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname
                for alias in node.names
                if alias.asname and alias.name.rpartition(".")[2] == "resolver"
            )
        elif isinstance(node, ast.ImportFrom):
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "resolver"
            )
    return aliases


def _reaches_resolver_internals(node: ast.Attribute, aliases: set[str]) -> bool:
    """Whether an attribute read pulls a private name off the resolver module."""
    if not node.attr.startswith("_"):
        return False
    target = _dotted_name(node.value)
    if target is None:
        return False
    if target in aliases:
        return True
    # Only a dotted path names the module on its own; a bare name is the
    # module solely when an import bound it, and is otherwise a local.
    return "." in target and target.rpartition(".")[2] == "resolver"


def _gated_symbol_offenses(source: str, filename: str) -> list[str]:
    """Imports, re-definitions and attribute reads of gated grammar symbols.

    Gated: the legacy v0.3 grammar helper names, wherever they are imported,
    defined or read from; and the resolver's private internals under any
    name, reached either by import or through the module object — those are
    not an API, whatever they are currently called.
    """
    offenses: list[str] = []
    tree = ast.parse(source, filename=filename)
    aliases = _resolver_module_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            offenses.extend(
                f"{filename}:{node.lineno} imports {alias.name}"
                for alias in node.names
                if alias.name in _GATED_SYMBOLS
            )
            from_resolver = (node.module or "").rpartition(".")[2] == "resolver"
            offenses.extend(
                f"{filename}:{node.lineno} imports resolver internal {alias.name}"
                for alias in node.names
                if from_resolver and alias.name.startswith("_")
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _GATED_SYMBOLS:
                offenses.append(f"{filename}:{node.lineno} defines {node.name}")
        elif isinstance(node, ast.Assign):
            offenses.extend(
                f"{filename}:{node.lineno} defines {target.id}"
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in _GATED_SYMBOLS
            )
        elif isinstance(node, ast.Attribute):
            if node.attr in _GATED_SYMBOLS:
                offenses.append(f"{filename}:{node.lineno} reads {node.attr}")
            elif _reaches_resolver_internals(node, aliases):
                offenses.append(
                    f"{filename}:{node.lineno} reads resolver internal {node.attr}"
                )
    return offenses


def _scan_package(root: Path, exempt: Path) -> tuple[list[str], list[Path]]:
    """Gated-symbol offenses across a package tree, with the modules scanned.

    Args:
        root: Package directory walked recursively for ``*.py`` modules.
        exempt: The one module allowed to own the grammar, matched by path.

    Returns:
        The offenses found, and the module paths they were sought in.
    """
    scanned = [path for path in sorted(root.rglob("*.py")) if path != exempt]
    offenses = [
        offense
        for path in scanned
        for offense in _gated_symbol_offenses(
            path.read_text(encoding="utf-8"), str(path)
        )
    ]
    return offenses, scanned


def test_gate_detector_flags_a_planted_violation() -> None:
    planted_import = "from diffract.core.constants import parse_contextual_field_name"
    assert _gated_symbol_offenses(planted_import, "planted.py")

    planted_def = "CONTEXT_SEPARATOR = '@'"
    assert _gated_symbol_offenses(planted_def, "planted.py")


def test_gate_detector_flags_resolver_internal_imports() -> None:
    # Current-name internals are gated too, not only the legacy v0.3 names.
    planted_absolute = "from diffract.session.resolver import _MODELS_PART"
    assert _gated_symbol_offenses(planted_absolute, "planted.py")

    planted_relative = "from .resolver import _SEPARATOR"
    assert _gated_symbol_offenses(planted_relative, "planted.py")

    sanctioned = "from diffract.session.resolver import FieldSelector, resolve"
    assert not _gated_symbol_offenses(sanctioned, "planted.py")


def test_gate_detector_flags_attribute_reach_into_the_grammar() -> None:
    # Importing the module and reading through it needs no gated import name.
    planted_alias = (
        "from diffract.session import resolver\n"
        "def f(a):\n"
        "    return resolver._resolve_string(a)\n"
    )
    assert _gated_symbol_offenses(planted_alias, "planted.py")

    planted_asname = (
        "import diffract.session.resolver as r\n"
        "def f(s):\n"
        "    return r._MODELS_PART.match(s)\n"
    )
    assert _gated_symbol_offenses(planted_asname, "planted.py")

    planted_dotted = (
        "import diffract.session.resolver\nSEP = diffract.session.resolver._SEPARATOR\n"
    )
    assert _gated_symbol_offenses(planted_dotted, "planted.py")

    planted_legacy = (
        "from diffract.core import constants\n"
        "def f(n):\n"
        "    return constants.parse_contextual_field_name(n)\n"
    )
    assert _gated_symbol_offenses(planted_legacy, "planted.py")


def test_gate_detector_leaves_unrelated_private_attributes_alone() -> None:
    # Private attributes on objects that merely resolve something, and the
    # resolver's own public surface, are not grammar reach.
    sanctioned = (
        "from diffract.session import resolver\n"
        "def f(y_resolver, values, address):\n"
        "    y_resolver._normalize(values)\n"
        "    return resolver.resolve(address)\n"
    )
    assert not _gated_symbol_offenses(sanctioned, "sanctioned.py")

    # viz styling binds locals named exactly `resolver` (plots/base/line.py,
    # marker.py, jitter.py) and calls private helpers on them; only an
    # import binds that name to the module.
    styling = (
        "from diffract.viz.styling import ColorResolver\n"
        "def f(palette, values):\n"
        "    resolver = ColorResolver(palette)\n"
        "    return resolver._normalize(values)\n"
    )
    assert not _gated_symbol_offenses(styling, "sanctioned.py")


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_gate_walk_reports_a_planted_module(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    exempt = _write(package / "resolver.py", "CONTEXT_SEPARATOR = '@'\n")
    clean = _write(
        package / "clean.py", "from diffract.session.resolver import resolve\n"
    )
    planted = _write(
        package / "sub" / "consumer.py",
        "from diffract.core.constants import parse_contextual_field_name\n",
    )

    offenses, scanned = _scan_package(package, exempt)

    assert set(scanned) == {clean, planted}
    assert len(offenses) == 1
    assert str(planted) in offenses[0]


def test_gate_walk_exempts_the_resolver_by_path_not_by_filename(
    tmp_path: Path,
) -> None:
    package = tmp_path / "pkg"
    exempt = _write(package / "resolver.py", "CONTEXT_SEPARATOR = '@'\n")
    impostor = _write(package / "other" / "resolver.py", "CONTEXT_SEPARATOR = '@'\n")

    offenses, scanned = _scan_package(package, exempt)

    assert exempt not in scanned
    assert len(offenses) == 1
    assert str(impostor) in offenses[0]


def test_no_module_outside_the_resolver_touches_grammar_symbols() -> None:
    src_root = Path(diffract.__file__).parent
    resolver_path = src_root / "session" / "resolver.py"
    assert resolver_path.is_file()

    offenses, scanned = _scan_package(src_root, resolver_path)

    # The sweep is only as good as its reach: a walk that quietly stops
    # covering the identity zones would pass on an empty offense list.
    covered = set(scanned)
    assert resolver_path not in covered
    for consumer in (
        Path("session") / "namespaces" / "results" / "__init__.py",
        Path("viz") / "data" / "extraction.py",
        Path("core") / "data" / "nn" / "aggregates" / "metadata.py",
        Path("core") / "storage" / "hdf5_manager.py",
    ):
        assert src_root / consumer in covered

    assert not offenses, "grammar symbols leaked outside the resolver:\n" + "\n".join(
        offenses
    )


def test_core_surfaces_no_longer_expose_grammar_symbols() -> None:
    import diffract.core
    import diffract.core.constants

    for name in _GATED_SYMBOLS:
        assert not hasattr(diffract.core, name)
        assert not hasattr(diffract.core.constants, name)
        assert name not in getattr(diffract.core, "__all__", ())


# ---------------- consumer migration evidence ----------------


def _session_with_frob_norm(container) -> Session:
    session = Session(container=container)
    rng = np.random.default_rng(0)
    session.models.add({"w": rng.random((4, 4))}, model_id="model_a")
    session.compute.apply("frob_norm")
    return session


def test_export_validation_extracts_base_names_via_resolver(
    ram_container, monkeypatch: pytest.MonkeyPatch
) -> None:
    from diffract.session.namespaces.results import validation

    session = _session_with_frob_norm(ram_container)

    seen: list[str] = []
    real_resolve = validation.resolve

    def _spy(address):
        seen.append(address)
        return real_resolve(address)

    monkeypatch.setattr(validation, "resolve", _spy)

    # A contextual miss takes the validation path via the resolver.
    session.results.export_metrics("frob_norm@models[model_a]")

    assert "frob_norm@models[model_a]" in seen


def test_aggregate_merge_renders_labels_from_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import diffract.session.namespaces.results as results_module

    rendered: list[str] = []
    real_render = results_module.render_label

    def _spy(selector, **kwargs):
        label = real_render(selector, **kwargs)
        rendered.append(label)
        return label

    monkeypatch.setattr(results_module, "render_label", _spy)

    metrics = {
        "uid1": {
            "metadata": {"name": "w", "model_id": "m1"},
            "fields": {},
        }
    }
    aggregate = {
        "field": "agreement",
        "member_uids": ("uid1",),
        "context_models": ("m1", "m2"),
        "context_params": ("w",),
        "value": 0.5,
    }
    results_module.ResultsNamespace._merge_single_aggregate(metrics, aggregate)

    assert rendered == ["agreement [models=m1, m2, parameters=w]"]
    assert metrics["uid1"]["fields"] == {"agreement [models=m1, m2, parameters=w]": 0.5}


def test_erase_paths_never_interpret_grammar(
    ram_container, monkeypatch: pytest.MonkeyPatch
) -> None:
    import diffract.session.resolver as resolver_module

    session = _session_with_frob_norm(ram_container)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "frob_norm",
                "context_models": ("model_a",),
                "context_params": ("w",),
                "value": 1.0,
            }
        ]
    )

    def _poisoned(address: str) -> FieldSelector:
        raise AssertionError("erase must not interpret identity strings")

    with monkeypatch.context() as patched:
        patched.setattr(resolver_module, "_resolve_string", _poisoned)
        session.results.erase("frob_norm")

    exported = session.results.export("frob_norm", sources="all", export_format="dict")
    for entry in exported.values():
        assert "frob_norm" not in entry["fields"]
