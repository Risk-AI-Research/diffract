"""Session behavior when a persistent store's schema version is incompatible."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from diffract.session import (
    IncompatibleStoreError,
    Session,
    SessionError,
    UnrecognizedStoreError,
)

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
    assert "python -m diffract.migrate" in message
    assert "Back up" in message


def test_metadata_only_upgrade_leaves_legacy_session_store_intact(
    temp_dir: Path,
) -> None:
    from diffract import upgrade_metadata_index
    from diffract.core.data.metadata.migrations import SchemaVersionError

    metadata_path = temp_dir / "metadata.db"
    _make_legacy_metadata_store(metadata_path)
    original = metadata_path.read_bytes()
    with pytest.raises(SchemaVersionError, match="guarded identity migration"):
        upgrade_metadata_index(metadata_path)
    assert metadata_path.read_bytes() == original


def test_upgrade_entry_point_expands_the_user_home(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tilde path works as users expect: docs and shell habits write
    ``~/...``, and pathlib does not expand it on its own."""
    import diffract

    monkeypatch.setenv("HOME", str(temp_dir))
    monkeypatch.setenv("USERPROFILE", str(temp_dir))
    metadata_path = temp_dir / "metadata.db"
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)
    with Session(config_path=cfg):
        pass
    assert diffract.upgrade_metadata_index("~/metadata.db") == []


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


def test_opening_a_session_on_another_applications_database_refuses(
    temp_dir: Path,
) -> None:
    """A mistyped or stale ``[metadata.sqlite] path`` can name a database
    that belongs to something else. The session must refuse it as a typed
    SessionError, leave it untouched, and not hand the user an upgrade
    instruction that would write to it."""
    metadata_path = temp_dir / "app.db"
    conn = sqlite3.connect(str(metadata_path))
    try:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
    finally:
        conn.close()
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    with pytest.raises(UnrecognizedStoreError) as excinfo:
        Session(config_path=cfg)

    assert isinstance(excinfo.value, SessionError)
    message = str(excinfo.value)
    assert str(metadata_path) in message
    assert "diffract.migrate" not in message
    with sqlite3.connect(str(metadata_path)) as check:
        assert int(check.execute("PRAGMA user_version").fetchone()[0]) == 0
        tables = check.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert tables == [("customers",)]


def test_session_reopens_a_store_written_before_the_identity_marker(
    temp_dir: Path,
) -> None:
    """Stores from the release that introduced schema versioning carry no
    application id, so the gate has to recognize them from their tables
    alone. Getting that wrong locks users out of their own stores on
    upgrade, which is worse than the footgun the gate closes."""
    import numpy as np

    metadata_path = temp_dir / "metadata.db"
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    session = Session(config_path=cfg)
    with session:
        session.models.add(
            {"layer.weight": np.ones((2, 2), dtype=np.float32)}, model_id="m1"
        )
    # Clear the marker the current release stamps: what remains is exactly
    # what the previous generation left on disk.
    conn = sqlite3.connect(str(metadata_path))
    try:
        conn.execute("PRAGMA application_id = 0")
    finally:
        conn.close()

    reopened = Session(config_path=cfg)
    with reopened:
        assert list(reopened.models.list()) == ["m1"]


def test_session_opens_normally_on_a_fresh_store(temp_dir: Path) -> None:
    """The version guard is silent on a fresh store: a new persistent index
    is created at the current version and the session initializes."""
    metadata_path = temp_dir / "metadata.db"
    cfg = temp_dir / "cfg.ini"
    _write_config(cfg, metadata_path)

    session = Session(config_path=cfg)
    with session:
        assert list(session.models.list()) == []
