"""Identifier-alphabet validation across the session ingest boundaries.

Namespace methods open their own session context, so these tests call them
directly (no ``with session:``), matching the surrounding suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffract.containers import WiringConfiguration, create_main_container
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.schema import ParameterType
from diffract.session import InvalidIdentifierError, Session, SessionError

pytestmark = pytest.mark.unit

GRAMMAR_SEPARATORS = ["@", "[", "]", ",", "/"]


def _matrix() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_invalid_identifier_error_is_a_session_error() -> None:
    assert issubclass(InvalidIdentifierError, SessionError)


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_add_rejects_model_id_with_separator(sep: str) -> None:
    session = Session(profile="ram")
    with pytest.raises(InvalidIdentifierError) as excinfo:
        session.models.add({"layer.weight": _matrix()}, model_id=f"run{sep}0")
    assert repr(sep) in str(excinfo.value)
    assert session.models.list() == []


def test_add_accepts_clean_model_id() -> None:
    session = Session(profile="ram")
    session.models.add({"layer.0.weight": _matrix()}, model_id="run-001.a")
    assert session.models.list() == ["run-001.a"]


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_add_rejects_parameter_name_with_separator(sep: str) -> None:
    session = Session(profile="ram")
    with pytest.raises(InvalidIdentifierError) as excinfo:
        session.models.add({f"layer{sep}weight": _matrix()}, model_id="ok")
    assert repr(sep) in str(excinfo.value)
    assert "parameter name" in str(excinfo.value)


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_rename_rejects_invalid_new_model_id(sep: str) -> None:
    session = Session(profile="ram")
    session.models.add({"layer.weight": _matrix()}, model_id="clean")
    with pytest.raises(InvalidIdentifierError):
        session.models.rename("clean", f"bad{sep}id")
    assert session.models.list() == ["clean"]


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_ingest_aggregates_rejects_context_model_with_separator(sep: str) -> None:
    session = Session(profile="ram")
    with pytest.raises(InvalidIdentifierError):
        session.results.ingest_aggregates(
            [
                {
                    "field_name": "l_overlap",
                    "context_models": (f"m{sep}1", "m2"),
                    "value": 1.0,
                }
            ]
        )


def test_ingest_aggregates_rejects_field_name_with_separator() -> None:
    session = Session(profile="ram")
    with pytest.raises(InvalidIdentifierError, match="aggregate field name"):
        session.results.ingest_aggregates(
            [
                {
                    "field_name": "l@overlap",
                    "context_models": ("m1", "m2"),
                    "value": 1.0,
                }
            ]
        )


def test_ingest_metrics_rejects_field_name_with_slash() -> None:
    session = Session(profile="ram")
    session.models.add({"layer.weight": _matrix()}, model_id="m")
    uid = session.models.parameters.list()[0]["uid"]
    with pytest.raises(InvalidIdentifierError, match="'/'"):
        session.results.ingest_metrics({uid: {"grad/norm": 1.0}})


def test_ingest_metrics_allows_contextual_field_name() -> None:
    session = Session(profile="ram")
    session.models.add({"layer.weight": _matrix()}, model_id="m")
    uid = session.models.parameters.list()[0]["uid"]
    session.results.ingest_metrics({uid: {"stable_rank@models[m]": 0.5}})
    got = session.results.export_metrics("stable_rank@models[m]", export_format="dict")
    assert got[uid]["fields"]["stable_rank@models[m]"] == 0.5


def _write_ram_config(path: Path, metadata_path: Path) -> None:
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

[export]
default_export_format = dict
""".strip()
        + "\n"
    )


@pytest.mark.parametrize("sep", GRAMMAR_SEPARATORS)
def test_merge_rejects_store_with_violating_identifier(
    tmp_path: Path, sep: str
) -> None:
    # '/' cannot reach a store: ParameterMetadata blocks it at construction.
    if sep == "/":
        pytest.skip("'/' is blocked by ParameterMetadata's storage-safety guard")

    cfg_src = tmp_path / f"src_{ord(sep)}.ini"
    cfg_dst = tmp_path / f"dst_{ord(sep)}.ini"
    _write_ram_config(cfg_src, tmp_path / f"src_{ord(sep)}.db")
    _write_ram_config(cfg_dst, tmp_path / f"dst_{ord(sep)}.db")

    container_src = create_main_container(cfg_src)
    container_src.storage.storage_manager()
    container_src.cache.cache_manager()
    repo_src = container_src.nn.parameter_repository()

    meta = ParameterMetadata(
        uid="v1", name="w", ptype=ParameterType.DENSE, model_id=f"run{sep}0"
    )
    ParameterDataProxy.create_and_store(meta=meta, repository=repo_src).set_field(
        "metric", 1.0
    )

    container_dst = create_main_container(cfg_dst)
    container_dst.storage.storage_manager()
    container_dst.cache.cache_manager()
    container_dst.nn.parameter_repository()

    WiringConfiguration.wire(container_src)
    session_src = Session(container=container_src)
    assert session_src.models.list() == [f"run{sep}0"]  # exotic store still reads

    WiringConfiguration.wire(container_dst)
    session_dst = Session(container=container_dst)

    WiringConfiguration.wire(container_dst)
    with pytest.raises(InvalidIdentifierError) as excinfo:
        session_dst.utils.merge_other_session(session_src)
    assert repr(sep) in str(excinfo.value)
    assert session_dst.models.list() == []
