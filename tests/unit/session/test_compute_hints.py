"""Unit tests for compute namespace metric listing and error hints."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from diffract.session import (
    KernelNotFoundError,
    ScopeValidationError,
    Session,
    SessionError,
)

pytestmark = pytest.mark.unit

_COMPUTE_LOGGER = "diffract.session.namespaces.compute"


@contextmanager
def _capture_logs(logger_name: str) -> Iterator[list[str]]:
    """Capture INFO messages from a named diffract logger.

    The package loggers set ``propagate = false`` with their own handler, so
    pytest's ``caplog`` (rooted at the root logger) never sees them; attach a
    handler to the exact module logger instead.
    """
    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _ListHandler(level=logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)


def _seed_models(container, model_ids: tuple[str, ...]) -> Session:
    """Seed a session with one single-parameter numpy model per id."""
    session = Session(container=container)
    rng = np.random.default_rng(0)
    for model_id in model_ids:
        session.models.add({"w.weight": rng.random((4, 4))}, model_id=model_id)
    return session


def test_list_available_metrics_returns_fields(ram_container) -> None:
    session = Session(container=ram_container)

    metrics = session.compute.list_available_metrics()

    assert "frob_norm" in metrics
    assert "stable_rank" in metrics


def test_list_available_metrics_verbose_names_producing_kernel(ram_container) -> None:
    session = Session(container=ram_container)

    verbose = session.compute.list_available_metrics(verbose=True)

    assert any(entry.startswith("frob_norm <- ") for entry in verbose)


def test_apply_unknown_field_suggests_close_match(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*frob_norm"):
        session.compute.apply("frobnorm")


def test_apply_unknown_field_points_to_metric_listing(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match="list_available_metrics"):
        session.compute.apply("definitely_not_a_metric")


def test_configure_unknown_kernel_suggests_close_match(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match=r"Did you mean.*hard_rank"):
        session.compute.configure_kernel("hard_rankk", threshold=1e-6)


# ---------------- CROSS_MODEL BINARY scope validation ----------------


def test_apply_binary_cross_model_three_models_raises_scope_error(
    ram_container,
) -> None:
    """l_overlap is a binary cross-model kernel: applying it with three models
    in scope must raise an actionable error, not silently write nothing."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with pytest.raises(ScopeValidationError) as excinfo:
        session.compute.apply("l_overlap")

    message = str(excinfo.value)
    assert "l_overlap" in message
    for model_id in ("a", "b", "c"):
        assert model_id in message
    # The recommendation is filter(model_ids=[...]); an unshipped pairs=
    # parameter must not be advertised.
    assert "filter(model_ids=" in message
    assert "pairs=" not in message
    # filter() returns a SessionContext, so the runnable form goes through
    # the compute namespace, never a bare .apply on the context.
    assert ".compute.apply(" in message
    assert ").apply(" not in message


def test_apply_transitive_binary_cross_model_raises_same_scope_error(
    ram_container,
) -> None:
    """A dependent metric (l_agreement -> l_overlap) must raise the same typed,
    actionable error rather than a raw KeyError from deep in storage."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with pytest.raises(ScopeValidationError) as excinfo:
        session.compute.apply("l_agreement")

    assert isinstance(excinfo.value, SessionError)
    message = str(excinfo.value)
    assert "l_overlap" in message  # names the offending dependency
    assert "l_agreement" in message  # names the requested field
    assert "filter(model_ids=" in message
    assert "pairs=" not in message


def test_apply_binary_cross_model_two_models_computes_end_to_end(
    ram_container,
) -> None:
    """A valid model pair is not over-blocked: the full chain computes and both
    l_overlap and its dependent l_agreement are stored."""
    session = _seed_models(ram_container, ("a", "b"))

    session.compute.apply("l_agreement")

    overlaps = session.results.export_aggregates("l_overlap", export_format="list")
    agreements = session.results.export_aggregates("l_agreement", export_format="list")
    assert len(overlaps) == 1
    assert len(agreements) == 1


def test_apply_does_not_log_success_when_scope_invalid(ram_container) -> None:
    """The false 'Successfully produced fields' log must not fire when the scope
    is rejected."""
    session = _seed_models(ram_container, ("a", "b", "c"))

    with (
        _capture_logs(_COMPUTE_LOGGER) as messages,
        pytest.raises(ScopeValidationError),
    ):
        session.compute.apply("l_overlap")

    assert not any("Successfully produced fields" in m for m in messages)


def test_apply_success_log_reports_only_written_fields(ram_container) -> None:
    """Re-applying an already-computed field writes nothing; the log must say so
    rather than falsely claim success."""
    session = _seed_models(ram_container, ("a", "b"))
    session.compute.apply("frob_norm")

    with _capture_logs(_COMPUTE_LOGGER) as messages:
        session.compute.apply("frob_norm")

    assert not any("Successfully produced fields" in m for m in messages)
    assert any("No new fields produced" in m for m in messages)
