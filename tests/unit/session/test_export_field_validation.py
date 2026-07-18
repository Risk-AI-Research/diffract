"""Unit tests for requested-field validation in result exports.

An export request distinguishes three cases, and each has a distinct,
observable outcome: a misspelled or unknown field raises with suggestions
(parity with ``apply``); a known field with no values in the current scope
is reported with an explicit warning naming the exporter that serves it;
ingested fields absent from the kernel registry export cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from diffract.session import KernelNotFoundError, Session

pytestmark = pytest.mark.unit

_VALIDATION_LOGGER = "diffract.session.namespaces.results.validation"


@contextmanager
def _capture_warnings(logger_name: str = _VALIDATION_LOGGER) -> Iterator[list[str]]:
    """Capture WARNING messages from a named diffract logger.

    The package loggers set ``propagate = false`` with their own handler, so
    pytest's ``caplog`` (rooted at the root logger) never sees them; attach a
    handler to the exact module logger instead.
    """
    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _ListHandler(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)


def _session_with_frob_norm(container, model_ids: tuple[str, ...] = ("a",)) -> Session:
    """A session with one 4x4 numpy parameter per model and frob_norm applied."""
    session = Session(container=container)
    rng = np.random.default_rng(0)
    for model_id in model_ids:
        session.models.add({"w": rng.random((4, 4))}, model_id=model_id)
    session.compute.apply("frob_norm")
    return session


# ---------------- misspelled / unknown fields raise ----------------


def test_export_metrics_misspelled_field_raises_with_suggestion(ram_container) -> None:
    """A typo must raise with a suggestion, like ``apply`` does — never
    yield an empty frame indistinguishable from an uncomputed field."""
    session = _session_with_frob_norm(ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*frob_norm"):
        session.results.export_metrics("frob_nrom")


def test_export_metrics_unknown_field_points_to_metric_listing(ram_container) -> None:
    session = _session_with_frob_norm(ram_container)

    with pytest.raises(KernelNotFoundError, match="list_available_metrics"):
        session.results.export_metrics("qqz_definitely_not_a_field")


def test_export_aggregates_misspelled_field_raises(ram_container) -> None:
    session = _session_with_frob_norm(ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Cannot export 'l_overlpa'"):
        session.results.export_aggregates("l_overlpa")


def test_export_all_sources_unknown_field_raises(ram_container) -> None:
    """The union path of sources='all' must still reject a name that neither
    source nor the registry can account for."""
    session = _session_with_frob_norm(ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Cannot export 'frob_nrom'"):
        session.results.export("frob_nrom", sources="all")


def test_misspelled_ingested_field_suggests_stored_name(ram_container) -> None:
    """Suggestions must draw from stored field names, not only the registry:
    a typo of an ingested (registry-unknown) field gets a did-you-mean too."""
    session = _session_with_frob_norm(ram_container)
    uid = next(iter(session.results.export_metrics("frob_norm", export_format="dict")))
    session.results.ingest_metrics({uid: {"my_custom_metric": 42.0}})

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*my_custom_metric"):
        session.results.export_metrics("my_custom_metrik")


# ---------------- known fields with no values warn, never silently ----------------


def test_known_uncomputed_field_warns_and_omits_no_other_columns(ram_container) -> None:
    """A producible-but-uncomputed field is not a typo: the export must go
    through, warn explicitly, and leave the computed columns intact."""
    session = _session_with_frob_norm(ram_container)

    with _capture_warnings() as warnings:
        result = session.results.export_metrics(
            "effective_rank", "frob_norm", export_format="dict"
        )

    assert any("'effective_rank'" in m and "compute.apply" in m for m in warnings), (
        warnings
    )
    for entry in result.values():
        assert "frob_norm" in entry["fields"]
        assert "effective_rank" not in entry["fields"]


def test_computed_field_exports_without_warning(ram_container) -> None:
    """The happy path stays silent: a served request must not warn."""
    session = _session_with_frob_norm(ram_container)

    with _capture_warnings() as warnings:
        result = session.results.export_metrics("frob_norm", export_format="dict")

    assert warnings == []
    assert len(result) == 1


def test_aggregate_level_field_via_metrics_names_the_right_exporter(
    ram_container,
) -> None:
    """l_overlap values live on the aggregate side; export_metrics() cannot
    return them and must say which exporter can (the A.6 level split)."""
    session = _session_with_frob_norm(ram_container, model_ids=("a", "b"))
    session.compute.apply("l_overlap")

    with _capture_warnings() as warnings:
        session.results.export_metrics("l_overlap", export_format="dict")

    assert any("'l_overlap'" in m and "export_aggregates" in m for m in warnings), (
        warnings
    )


def test_uncomputed_aggregate_level_field_via_metrics_mentions_level(
    ram_container,
) -> None:
    """Before any computation the warning must still route the user by the
    kernel's apply level rather than suggest a hopeless retry."""
    session = _session_with_frob_norm(ram_container, model_ids=("a", "b"))

    with _capture_warnings() as warnings:
        session.results.export_metrics("l_overlap", export_format="dict")

    assert any("CROSS_MODEL" in m and "export_aggregates" in m for m in warnings), (
        warnings
    )


def test_parameter_field_via_aggregates_names_the_right_exporter(
    ram_container,
) -> None:
    session = _session_with_frob_norm(ram_container)

    with _capture_warnings() as warnings:
        session.results.export_aggregates("frob_norm", export_format="list")

    assert any("'frob_norm'" in m and "export_metrics" in m for m in warnings), warnings


def test_all_sources_serves_aggregate_field_without_warning(ram_container) -> None:
    """sources='all' searches both sides; a field found on either must not
    warn — a per-leg validator would fire spuriously here."""
    session = _session_with_frob_norm(ram_container, model_ids=("a", "b"))
    session.compute.apply("l_overlap")

    with _capture_warnings() as warnings:
        session.results.export("l_overlap", sources="all", export_format="dict")

    assert warnings == []


# ---------------- ingested fields stay exportable ----------------


def test_ingested_metric_field_exports_without_registry_entry(ram_container) -> None:
    """Registry-only validation would break ingested fields; a stored field
    the registry has never heard of must export cleanly and silently."""
    session = _session_with_frob_norm(ram_container)
    uid = next(iter(session.results.export_metrics("frob_norm", export_format="dict")))
    session.results.ingest_metrics({uid: {"my_custom_metric": 42.0}})

    with _capture_warnings() as warnings:
        result = session.results.export_metrics(
            "my_custom_metric", export_format="dict"
        )

    assert warnings == []
    assert result[uid]["fields"]["my_custom_metric"] == 42.0


def test_ingested_aggregate_field_exports_without_registry_entry(
    ram_container,
) -> None:
    session = Session(container=ram_container)
    session.results.ingest_aggregates(
        [
            {
                "field_name": "my_agg",
                "context_models": ("m1", "m2"),
                "context_params": ("w",),
                "value": 7.0,
            }
        ]
    )

    with _capture_warnings() as warnings:
        result = session.results.export_aggregates("my_agg", export_format="list")

    assert warnings == []
    assert len(result) == 1
    assert result[0]["value"] == 7.0


# ---------------- scope-relative resolution ----------------


def test_scoped_export_of_field_computed_elsewhere_warns(ram_container) -> None:
    """Model b was added after frob_norm ran, so the scoped view has no
    values: the export must say so instead of returning silence."""
    session = _session_with_frob_norm(ram_container)
    session.models.add({"w": np.random.default_rng(1).random((4, 4))}, model_id="late")

    with _capture_warnings() as warnings:
        result = session.filter(model_ids=["late"]).results.export_metrics(
            "frob_norm", export_format="dict"
        )

    assert any("'frob_norm'" in m and "no computed values" in m for m in warnings), (
        warnings
    )
    for entry in result.values():
        assert entry["fields"] == {}


def test_scoped_export_of_out_of_scope_ingested_field_raises(ram_container) -> None:
    """Resolution is scope-relative: an ingested name existing only outside
    the filter is unknown within it and must raise, naming the scope."""
    session = _session_with_frob_norm(ram_container, model_ids=("a", "b"))
    exported = session.filter(model_ids=["a"]).results.export_metrics(
        "frob_norm", export_format="dict"
    )
    uid = next(iter(exported))
    session.results.ingest_metrics({uid: {"only_on_a": 1.0}})

    with pytest.raises(KernelNotFoundError, match="current scope"):
        session.filter(model_ids=["b"]).results.export_metrics("only_on_a")
