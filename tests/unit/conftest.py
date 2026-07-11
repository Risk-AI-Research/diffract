"""Shared fixtures for unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffract.containers import create_main_container


def write_ram_config(path: Path) -> None:
    """Write a minimal RAM-storage INI config for container-based tests."""
    metadata_path = path.parent / f"{path.stem}_metadata.db"
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


@pytest.fixture(scope="session")
def ram_config_factory(tmp_path_factory: pytest.TempPathFactory):
    """Factory producing fresh RAM-storage INI configs in isolated dirs."""

    def factory(name: str = "ram_cfg") -> Path:
        cfg = tmp_path_factory.mktemp(name) / "cfg.ini"
        write_ram_config(cfg)
        return cfg

    return factory


@pytest.fixture
def ram_config_path(tmp_path: Path) -> Path:
    """Path to a minimal RAM-storage INI config."""
    cfg = tmp_path / "ram_cfg.ini"
    write_ram_config(cfg)
    return cfg


@pytest.fixture
def ram_container(ram_config_path: Path):
    """A fully wired main container backed by RAM storage."""
    return create_main_container(ram_config_path)
