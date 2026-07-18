"""Session behavior when a persistent store's schema version is incompatible."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from diffract.session import IncompatibleStoreError, Session, SessionError

pytestmark = pytest.mark.unit


def _write_config(path: Path, metadata_path: Path) -> None:
    path.write_text(
        f"""
[storage]
backend = ram

[metadata]
backend = sqlite

[metadata.sqlite]
path = {metadata_path}

[cache]
backend = simple

[cache.simple]
max_memory_mb = 16
ttl_seconds = 3600
key_prefix = "test:cache:"

[parallel.thread_pool]
max_workers = 1

[parallel.process_pool]
max_workers = 1

[export]
default_export_format = dict

[nn.extractor]
skip_not_implemented_types = true
""".strip()
        + "\n"
    )


def _make_legacy_metadata_store(path: Path) -> None:
    """A metadata database from before schema versioning: tables exist but
    ``user_version`` was never stamped (it reads back as 0)."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE parameters (uid TEXT PRIMARY KEY, name TEXT, "
            "model_id TEXT, ptype TEXT, json_data TEXT)"
        )
        conn.commit()
    finally:
        conn.close()


def test_opening_a_session_on_a_legacy_store_raises_actionable_session_error(
    temp_dir: Path,
) -> None:
    """A persistent store predating schema versioning must not open silently.
    The failure is a typed SessionError -- so ``except SessionError``
    catches it -- and the message is actionable: it names the explicit
    upgrade entry point and recommends a backup first."""
    metadata_path = temp_dir / "metadata.db"
    _make_legacy_metadata_store(metadata_path)
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    with pytest.raises(IncompatibleStoreError) as excinfo:
        Session(config_path=cfg)

    assert isinstance(excinfo.value, SessionError)
    message = str(excinfo.value)
    # The named remediation is the publicly importable path, verbatim.
    assert "diffract.upgrade_metadata_index" in message
    assert "Back up" in message


def test_named_upgrade_entry_point_is_publicly_importable(temp_dir: Path) -> None:
    """The call the refusal message dictates must work as written: the
    top-level export upgrades the store in place and the session opens."""
    import diffract

    metadata_path = temp_dir / "metadata.db"
    _make_legacy_metadata_store(metadata_path)
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    with pytest.raises(IncompatibleStoreError):
        Session(config_path=cfg)

    applied = diffract.upgrade_metadata_index(metadata_path)
    assert applied  # the legacy store needed at least one step

    session = Session(config_path=cfg)
    with session:
        assert list(session.models.list()) == []


def test_upgrade_entry_point_expands_the_user_home(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tilde path works as users expect: docs and shell habits write
    ``~/...``, and pathlib does not expand it on its own."""
    import diffract

    monkeypatch.setenv("HOME", str(temp_dir))
    monkeypatch.setenv("USERPROFILE", str(temp_dir))
    metadata_path = temp_dir / "metadata.db"
    _make_legacy_metadata_store(metadata_path)

    applied = diffract.upgrade_metadata_index("~/metadata.db")

    assert applied


def test_failed_session_open_is_retryable(temp_dir: Path) -> None:
    """The failed open must release what it acquired: a second attempt sees
    the same typed error, not a lingering lock or a half-initialized state.
    This is the documented recovery loop (catch, back up, upgrade, retry)."""
    metadata_path = temp_dir / "metadata.db"
    _make_legacy_metadata_store(metadata_path)
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    for _ in range(2):
        with pytest.raises(IncompatibleStoreError):
            Session(config_path=cfg)


def test_session_opens_normally_on_a_fresh_store(temp_dir: Path) -> None:
    """The version guard is silent on a fresh store: a new persistent index
    is created at the current version and the session initializes."""
    metadata_path = temp_dir / "metadata.db"
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    session = Session(config_path=cfg)
    with session:
        assert list(session.models.list()) == []
