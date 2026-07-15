"""Executable README: run python code blocks from README.md via sybil.

Every python block in README.md is executed as a test unless preceded by
an invisible ``<!-- skip: next -->`` comment. Blocks share one namespace
per document; the models referenced by the examples are prepared here.
The document runs inside a temporary working directory so persistent
profiles do not touch the repository tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser


def _build_readme_models() -> dict[str, Any]:
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(32, 32)
            self.k_proj = torch.nn.Linear(32, 32)

    class Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 32)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()
            self.mlp = Mlp()

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class ToyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()

    torch_model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
    )
    return {
        "torch": torch,
        "torch_model": torch_model,
        "torch_state_dict": torch_model.state_dict(),
        "model": ToyTransformer(),
        "my_model": torch_model,
    }


def _readme_setup(namespace: dict[str, Any]) -> None:
    namespace["__readme_cwd"] = str(Path.cwd())
    workdir = tempfile.mkdtemp(prefix="diffract-readme-")
    os.chdir(workdir)
    namespace.update(_build_readme_models())


def _readme_teardown(namespace: dict[str, Any]) -> None:
    os.chdir(namespace["__readme_cwd"])


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    pattern="README.md",
    setup=_readme_setup,
    teardown=_readme_teardown,
).pytest()
