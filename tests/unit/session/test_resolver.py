"""Acceptance tests for the single identity resolver.

Pins three contracts: every address form resolves to the same structured
components, with mint-produced uids round-tripping; the legacy grammar
symbols are neither importable nor redefined outside the resolver module;
and the known consumers (export validation, the aggregate merge, the erase
path) reach entities through resolver selectors.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import diffract
from diffract.core.data.nn.aggregates.metadata import AggregateMetadata
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


# ---------------- round-trip against the uid mint ----------------


def test_minted_uid_round_trips_to_its_context() -> None:
    uid = AggregateMetadata.create_uid_from_context(
        field_name="l_overlap",
        context_models=("model_b", "model_a"),
        context_params=("p2", "p1"),
    )
    selector = resolve(uid)
    assert selector.field == "l_overlap"
    assert selector.models == ("model_a", "model_b")
    assert selector.params == ("p1", "p2")


def test_render_label_matches_the_minted_uid() -> None:
    # Rendered labels must coincide with stored uids while both exist.
    uid = AggregateMetadata.create_uid_from_context(
        field_name="agreement", context_models=("m1", "m2"), context_params=("w",)
    )
    assert render_label(resolve(uid)) == uid


def test_render_label_sorts_context_members() -> None:
    label = render_label(
        FieldSelector(field="k", models=("m2", "m1"), params=("p2", "p1"))
    )
    assert label == "k@models[m1,m2]@params[p1,p2]"


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


def _gated_symbol_offenses(source: str, filename: str) -> list[str]:
    """Imports or re-definitions of gated grammar symbols in a module."""
    offenses: list[str] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            offenses.extend(
                f"{filename}:{node.lineno} imports {alias.name}"
                for alias in node.names
                if alias.name in _GATED_SYMBOLS
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
    return offenses


def test_gate_detector_flags_a_planted_violation() -> None:
    planted_import = "from diffract.core.constants import parse_contextual_field_name"
    assert _gated_symbol_offenses(planted_import, "planted.py")

    planted_def = "CONTEXT_SEPARATOR = '@'"
    assert _gated_symbol_offenses(planted_def, "planted.py")


def test_no_module_outside_the_resolver_touches_grammar_symbols() -> None:
    src_root = Path(diffract.__file__).parent
    resolver_path = src_root / "session" / "resolver.py"

    offenses: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path == resolver_path:
            continue
        offenses.extend(
            _gated_symbol_offenses(path.read_text(encoding="utf-8"), str(path))
        )

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

    def _spy(selector):
        label = real_render(selector)
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
        "context_models": ("m1", "m2"),
        "context_params": ("w",),
        "value": 0.5,
    }
    results_module.ResultsNamespace._merge_single_aggregate(metrics, aggregate)

    assert rendered == ["agreement@models[m1,m2]@params[w]"]
    assert metrics["uid1"]["fields"] == {"agreement@models[m1,m2]@params[w]": 0.5}


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
