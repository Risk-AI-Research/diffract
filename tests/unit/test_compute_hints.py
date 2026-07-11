"""Unit tests for compute namespace metric listing and error hints."""

from __future__ import annotations

import pytest

from diffract.session import KernelNotFoundError, Session

pytestmark = pytest.mark.unit


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

    with pytest.raises(KernelNotFoundError, match="Did you mean.*frob_norm"):
        session.compute.apply("frobnorm")


def test_apply_unknown_field_points_to_metric_listing(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match="list_available_metrics"):
        session.compute.apply("definitely_not_a_metric")


def test_configure_unknown_kernel_suggests_close_match(ram_container) -> None:
    session = Session(container=ram_container)

    with pytest.raises(KernelNotFoundError, match="Did you mean.*hard_rank"):
        session.compute.configure_kernel("hard_rankk", threshold=1e-6)
