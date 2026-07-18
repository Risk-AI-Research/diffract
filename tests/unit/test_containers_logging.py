"""Unit tests for logging configuration of built-in profiles."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from diffract.containers import create_main_container

pytestmark = pytest.mark.unit


def test_ram_profile_writes_no_files_and_no_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        create_main_container(profile="ram")

    assert list(tmp_path.rglob("*.log")) == []
    assert "Failed to configure logging" not in caplog.text


def test_local_profile_precreates_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        create_main_container(profile="local")

    assert (tmp_path / ".diffract").is_dir()
    assert "Failed to configure logging" not in caplog.text


def test_config_without_logging_section_uses_fallback_silently(
    ram_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        create_main_container(ram_config_path)

    assert "Failed to configure logging" not in caplog.text


def test_hybrid_profile_logs_under_diffract_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        create_main_container(profile="hybrid")

    assert not (tmp_path / "diffract.log").exists()
    assert "Failed to configure logging" not in caplog.text


def test_user_logging_survives_session_resource_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    container = create_main_container(profile="ram")

    root = logging.getLogger()
    diffract_logger = logging.getLogger("diffract")
    root_handler = logging.NullHandler()
    diffract_handler = logging.NullHandler()
    original_root_level = root.level

    root.addHandler(root_handler)
    diffract_logger.addHandler(diffract_handler)
    root.setLevel(logging.DEBUG)
    try:
        # A namespace call enters and exits the container context, i.e. one
        # init_resources()/shutdown_resources() pair.
        for _ in range(3):
            container.init_resources()
            container.shutdown_resources()

        assert root_handler in root.handlers
        assert diffract_handler in diffract_logger.handlers
        assert root.level == logging.DEBUG
    finally:
        root.removeHandler(root_handler)
        diffract_logger.removeHandler(diffract_handler)
        root.setLevel(original_root_level)
