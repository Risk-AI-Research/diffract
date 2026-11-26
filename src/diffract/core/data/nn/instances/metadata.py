"""Metadata for a persistent model instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstanceMetadata:
    """Named model identity with a creation time in Unix seconds."""

    instance_uid: str
    name: str
    display_label: str | None
    created_at: int
    json_data: dict[str, Any]
