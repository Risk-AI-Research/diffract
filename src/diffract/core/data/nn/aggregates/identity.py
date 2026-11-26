"""Canonical identities and membership rows for aggregates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


PAIR_SIZE = 2


@dataclass(frozen=True, slots=True)
class AggregateIdentity:
    """Content address and ordered (role, parameter UID) memberships."""

    uid: str
    members: tuple[tuple[str, str], ...]


def aggregate_identity(
    field: str, member_uids: Iterable[str], *, directed: bool = False
) -> AggregateIdentity:
    """Hash a field and its parameter memberships with a format discriminator.

    Args:
        field: Base field name.
        member_uids: Parameter identifiers, in source-to-target order for a pair.
        directed: Whether the inputs have distinct source and target roles.

    Returns:
        SHA-256 identity and the canonical membership order for storage.
    """
    if directed:
        source, target = member_uids
        members = (("source", source), ("target", target))
        payload = ["directed", field, [source, target]]
    else:
        ordered = sorted(set(member_uids), key=str.encode)
        members = tuple(("member", uid) for uid in ordered)
        payload = ["set", field, ordered]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return AggregateIdentity(hashlib.sha256(encoded).hexdigest(), members)
