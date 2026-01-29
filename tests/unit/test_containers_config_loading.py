"""Tests for container configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffract.containers import _coerce_ini_value, _parse_ini_config, create_main_container


def test_coerce_ini_value_basic_types() -> None:
    assert _coerce_ini_value("true") is True
    assert _coerce_ini_value("FALSE") is False
    assert _coerce_ini_value("none") is None
    assert _coerce_ini_value("null") is None

    assert _coerce_ini_value("2") == 2
    assert _coerce_ini_value("128_000") == 128000
    assert _coerce_ini_value("3.5") == 3.5

    assert _coerce_ini_value('["a", 1, true]') == ["a", 1, True]
    assert _coerce_ini_value('{"k": 1}') == {"k": 1}
    assert _coerce_ini_value('"/tmp/x"') == "/tmp/x"

    assert _coerce_ini_value("/tmp/x") == "/tmp/x"


def test_parse_ini_config_creates_nested_dict(temp_dir: Path) -> None:
    ini_path = temp_dir / "cfg.ini"
    ini_path.write_text(
        """
[storage]
backend=hdf5

[storage.hdf5]
path=/tmp/a.h5

[compute.executor]
max_workers=2
""".strip()
        + "\n"
    )

    parsed = _parse_ini_config(ini_path)
    assert parsed["storage"]["backend"] == "hdf5"
    assert parsed["storage"]["hdf5"]["path"] == "/tmp/a.h5"
    assert parsed["compute"]["executor"]["max_workers"] == 2


def test_create_main_container_unsupported_config_extension(temp_dir: Path) -> None:
    cfg_path = temp_dir / "cfg.toml"
    cfg_path.write_text("a=1\n")

    with pytest.raises(ValueError, match="Unsupported config file extension"):
        create_main_container(cfg_path)
