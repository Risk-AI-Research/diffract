"""Plan a transition from name-addressed weights to coordinate identities."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

from diffract.core.data.identity import normalize_parameter_name, validate_identifier
from diffract.core.data.metadata.migrations import (
    conflicting_identity_objects,
    verify_legacy_schema_meta,
)
from diffract.core.data.nn.aggregates.identity import aggregate_identity
from diffract.core.utils.hashing import new_entity_uid
from diffract.session.resolver import resolve

type Keep = Literal["abort", "oldest", "newest"]
type Coordinate = tuple[str, int, str]
_PAIR_MEMBERS = 2
_STEP_SUFFIX = re.compile(r"(.+)_step(\d+)\Z")
_ENTITY_INDEXES = {
    (table, f"idx_{table}_{column}_uid"): (column, "uid")
    for table, columns in (
        ("parameters", ("model_id", "ptype", "name")),
        ("aggregates", ("field_name",)),
    )
    for column in columns
}


@dataclass
class LegacyPlan:
    """Entities, retained owners and diagnostics for the guarded migration."""

    instances: dict[str, str] = field(default_factory=dict)
    parameters: list[dict] = field(default_factory=list)
    aggregates: list[dict] = field(default_factory=list)
    members: list[dict] = field(default_factory=list)
    uid_map: dict[tuple[str, str], str] = field(default_factory=dict)
    retained: dict[tuple[str, str], str] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _metadata_rows(db: sqlite3.Connection, table: str) -> list[dict]:
    if not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? COLLATE NOCASE",
        (table,),
    ).fetchone():
        return []
    if table not in {"parameters", "aggregates"}:
        raise ValueError(f"Unknown entity table {table!r}")
    cursor = db.execute(
        f"SELECT rowid AS original_rowid, * FROM {table} ORDER BY rowid"  # nosec B608
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def metadata_signature(db: sqlite3.Connection) -> bytes:
    """Fingerprint the input rows used to plan entity identity."""
    digest = hashlib.sha256()
    for entry in db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY name"
    ):
        digest.update(json.dumps(entry).encode())
    if db.execute(
        "SELECT 1 FROM sqlite_schema WHERE name='schema_meta' COLLATE NOCASE"
    ).fetchone():
        for record in db.execute("SELECT * FROM schema_meta ORDER BY rowid"):
            digest.update(json.dumps(record).encode())
    for table in ("parameters", "aggregates"):
        digest.update(table.encode())
        for row in _metadata_rows(db, table):
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
    return digest.digest()


def _choose(rows: list[dict], key: Any, keep: Keep, plan: LegacyPlan) -> dict:
    if len(rows) > 1:
        plan.conflicts.append(
            {
                "coordinate": key,
                "rows": [
                    {
                        "old_uid": row["uid"],
                        "rowid": row["original_rowid"],
                        "provenance": json.loads(row["json_data"] or "{}"),
                    }
                    for row in rows
                ],
            }
        )
    return rows[-1] if keep == "newest" else rows[0]


def _dependent_extensions(db: sqlite3.Connection) -> set[str]:
    """Find schema objects whose behavior would change when owners are rekeyed."""
    dependent = {
        name
        for (name,) in db.execute(
            "SELECT m.name FROM sqlite_schema AS m, "
            "pragma_foreign_key_list(m.name) AS fk "
            "WHERE m.type='table' AND lower(fk.[table]) IN ('parameters','aggregates') "
            "UNION SELECT name FROM sqlite_schema WHERE type='trigger' "
            "AND lower(tbl_name) IN ('parameters','aggregates','schema_meta')"
        )
    }
    for table in ("parameters", "aggregates"):
        indexes = db.execute(
            "SELECT name, [unique], partial FROM pragma_index_list(?) "
            "WHERE origin!='pk'",
            (table,),
        )
        for name, unique, partial in indexes:
            columns = [
                col
                for (col,) in db.execute(
                    "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (name,)
                )
            ]
            if (
                unique
                or partial
                or _ENTITY_INDEXES.get((table, name)) != tuple(columns)
            ):
                dependent.add(name)

    def observe(
        _action: int,
        table: str | None,
        _column: str | None,
        _database: str | None,
        source: str | None,
    ) -> int:
        if source and table and table.lower() in {"parameters", "aggregates"}:
            dependent.add(source)
        return sqlite3.SQLITE_OK

    extensions = list(
        db.execute(
            "SELECT type,name,tbl_name FROM sqlite_schema "
            "WHERE type IN ('view','trigger')"
        )
    )
    if not extensions:
        return dependent
    db.set_authorizer(observe)
    try:
        for kind, name, table in extensions:
            if name in dependent:
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            if kind == "view":
                db.execute(f"EXPLAIN SELECT * FROM {quoted}")  # nosec B608 - quoted schema identifiers
                continue
            columns = [
                q
                for (q,) in db.execute(
                    """SELECT printf('"%w"',name) FROM pragma_table_xinfo(?) """
                    "WHERE hidden=0",
                    (table,),
                )
            ]
            assignments = ",".join(f"{c}={c}" for c in columns)
            for operation in (
                f"DELETE FROM {quoted}",  # nosec B608 - quoted schema identifiers
                f"INSERT INTO {quoted} DEFAULT VALUES",  # nosec B608 - quoted schema identifiers
                f"UPDATE {quoted} SET {assignments}",  # nosec B608 - quoted schema identifiers
            ):
                try:
                    db.execute(f"EXPLAIN {operation}")
                except sqlite3.OperationalError as exc:
                    if "because it is a view" not in str(exc):
                        raise
    finally:
        db.set_authorizer(None)
    return dependent


def plan_legacy(
    db: sqlite3.Connection,
    *,
    keep: Keep,
    rename: dict[str, dict[str, str]],
    parse_step_suffix: bool,
) -> LegacyPlan:
    """Resolve legacy rows, reporting conflicts before either store is changed."""
    if keep not in {"abort", "oldest", "newest"}:
        raise ValueError(f"Invalid duplicate policy {keep!r}")
    if (
        not isinstance(rename, dict)
        or set(rename) - {"models", "parameters"}
        or any(
            not isinstance(mapping, dict)
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
            )
            for mapping in rename.values()
        )
    ):
        raise ValueError("Expected a JSON object with models and parameters name maps.")
    plan = LegacyPlan()
    if not verify_legacy_schema_meta(db):
        plan.errors.append("Unsupported schema_meta definition or producer record.")
    for name in conflicting_identity_objects(db):
        plan.errors.append(f"Schema object {name!r} conflicts with the target schema.")
    for table, known in {
        "parameters": {"uid", "name", "model_id", "ptype", "json_data"},
        "aggregates": {
            "uid",
            "field_name",
            "context_models",
            "context_params",
            "json_data",
        },
        "schema_meta": {"id", "created_by"},
    }.items():
        columns = {
            row[0]
            for row in db.execute("SELECT name FROM pragma_table_xinfo(?)", (table,))
        }
        if columns and columns != known:
            plan.errors.append(
                f"Unsupported columns in {table}: {sorted(columns ^ known)}"
            )
    for name in sorted(_dependent_extensions(db)):
        plan.errors.append(
            f"Schema extension {name!r} depends on legacy identities. "
            "Move or adapt this extension before upgrading."
        )
    if plan.errors:
        return plan
    groups: dict[Coordinate, list[dict]] = {}
    original_coordinates: dict[tuple[str, str], str] = {}
    for row in _metadata_rows(db, "parameters"):
        if not isinstance(row["uid"], str) or not row["uid"]:
            plan.errors.append(f"Invalid parameter UID at row {row['original_rowid']}.")
            continue
        old_model, old_name = row["model_id"], row["name"]
        name = rename.get("models", {}).get(old_model, old_model)
        parameter = rename.get("parameters", {}).get(old_name, old_name)
        step = 0
        try:
            if (
                parse_step_suffix
                and isinstance(name, str)
                and (match := _STEP_SUFFIX.fullmatch(name))
            ):
                name, step = match.group(1), int(match.group(2))
            validate_identifier(name, kind="model name")
            parameter = normalize_parameter_name(parameter)
        except ValueError as exc:
            plan.errors.append(f"{row['uid']}: {exc}")
            continue
        if step > 2**63 - 1:
            plan.errors.append(f"{old_model!r}: step exceeds SQLite integer range")
            continue
        if name not in plan.instances:
            plan.instances[name] = new_entity_uid()
        plan.uid_map["instance", old_model] = plan.instances[name]
        groups.setdefault((name, step, parameter), []).append(row)

    for (model, step, name), rows in groups.items():
        selected = _choose(rows, (model, step, name), keep, plan)
        uid = new_entity_uid()
        for row in rows:
            plan.uid_map["parameter", row["uid"]] = uid
            original_coordinates[row["model_id"], row["name"]] = uid
        plan.retained["parameters", selected["uid"]] = uid
        plan.parameters.append(
            {
                "uid": uid,
                "instance_uid": plan.instances[model],
                "step": step,
                "name": name,
                "ptype": "UNKNOWN" if selected["ptype"] is None else selected["ptype"],
                "json_data": selected["json_data"] or "{}",
            }
        )

    by_model: dict[str, dict[str, str]] = {}
    for (old_model, old_name), uid in original_coordinates.items():
        by_model.setdefault(old_model, {})[old_name] = uid
    aggregates: dict[str, list[dict]] = {}
    bridges: dict[str, tuple] = {}
    for row in _metadata_rows(db, "aggregates"):
        if not isinstance(row["uid"], str) or not row["uid"]:
            plan.errors.append(f"Invalid aggregate UID at row {row['original_rowid']}.")
            continue
        address = resolve(row["uid"])
        field_name = row["field_name"]
        models = set(json.loads(row["context_models"] or "[]"))
        names = set(json.loads(row["context_params"] or "[]"))
        # Stored JSON disambiguates old identifiers containing grammar delimiters.
        for recorded, parsed, part in (
            (models, address.models, "model"),
            (names, address.params, "parameter"),
        ):
            if not recorded and parsed:
                recorded.update(parsed)
            elif (
                parsed
                and all(not set(n).intersection("@[],") for n in recorded)
                and recorded != set(parsed)
            ):
                plan.errors.append(
                    f"Conflicting {part} context in aggregate {row['uid']!r}"
                )
        if field_name != address.field:
            plan.errors.append(
                f"Aggregate {row['uid']!r} disagrees with field {field_name!r}"
            )
            continue
        chosen = [
            (model, name, uid)
            for model in (models or (by_model if names else ()))
            for name, uid in by_model.get(model, {}).items()
            if not names or name in names
        ]
        if models - {model for model, _, _ in chosen} or names - {
            name for _, name, _ in chosen
        }:
            plan.errors.append(f"Unresolved members for aggregate {row['uid']!r}")
            continue
        identity = aggregate_identity(field_name, (uid for _, _, uid in chosen))
        row["unknown_pair"] = (
            len(identity.members) == _PAIR_MEMBERS
            and len(chosen) == _PAIR_MEMBERS
            and len({model for model, _, _ in chosen}) == _PAIR_MEMBERS
        )
        plan.uid_map["aggregate", row["uid"]] = identity.uid
        aggregates.setdefault(identity.uid, []).append(row)
        bridges[identity.uid] = identity.members

    for uid, rows in aggregates.items():
        row = _choose(rows, uid, keep, plan)
        name = row["field_name"]
        flags = {"direction_unknown": True} if row["unknown_pair"] else {}
        plan.aggregates.append(
            {
                "uid": uid,
                "field_name": name,
                "json_data": json.dumps(json.loads(row["json_data"] or "{}") | flags),
            }
        )
        plan.members.extend(
            {
                "aggregate_uid": uid,
                "member_uid": member,
                "role": role,
                "json_data": json.dumps(flags),
            }
            for role, member in bridges[uid]
        )
        plan.retained["aggregates", row["uid"]] = uid
    return plan
