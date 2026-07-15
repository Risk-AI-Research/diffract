"""Tests for the configured default export format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffract.containers import PROFILES, create_main_container
from diffract.session.session import Session

pytestmark = pytest.mark.unit


def _config_with_export_format(path: Path, export_format: str | None) -> Path:
    """Rewrite the [export] default of an existing ram config.

    Passing None drops the [export] section entirely.
    """
    text = path.read_text()
    if export_format is None:
        text = text.replace("[export]\ndefault_export_format = dict\n\n", "")
    else:
        text = text.replace(
            "default_export_format = dict",
            f"default_export_format = {export_format}",
        )
    path.write_text(text)
    return path


def _session_with(path: Path, export_format: str | None) -> Session:
    cfg = _config_with_export_format(path, export_format)
    session = Session(profile=None, config_path=cfg)
    weights = np.random.default_rng(0).standard_normal((4, 3))
    session.models.add({"w": weights}, model_id="m1")
    session.compute.apply("frob_norm")
    return session


def test_export_format_defaults_to_the_configured_value(ram_config_path: Path) -> None:
    """A call that names no format takes the configured one. The formatters
    return distinct container types, so the returned type identifies which
    formatter ran: a configuration asking for "list" that yielded a dict would
    mean the key reached no consumer."""
    session = _session_with(ram_config_path, "list")

    assert isinstance(session.results.export_metrics("frob_norm"), list)


def test_explicit_export_format_overrides_the_configured_default(
    ram_config_path: Path,
) -> None:
    """The per-call argument wins over configuration."""
    session = _session_with(ram_config_path, "list")

    assert isinstance(
        session.results.export_metrics("frob_norm", export_format="dict"), dict
    )


def test_export_format_falls_back_when_the_section_is_absent(
    ram_config_path: Path,
) -> None:
    """A configuration carrying no [export] section still exports, in dict form."""
    session = _session_with(ram_config_path, None)

    assert isinstance(session.results.export_metrics("frob_norm"), dict)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_shipped_profiles_default_to_a_format_needing_no_extra(profile: str) -> None:
    """The pandas and polars formatters require their optional extras, so a
    shipped profile defaulting to one of them would make export raise for anyone
    who installed only the base package."""
    container = create_main_container(profile=profile)

    configured = container.config()["export"]["default_export_format"]
    assert configured in {"dict", "json", "list"}
