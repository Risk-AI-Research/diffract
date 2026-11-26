"""Tests for the SQLite metadata index."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from threading import Thread

import pytest

from diffract.core.data.metadata.errors import (
    CheckConstraintError,
    ConstraintViolationError,
    ForeignKeyViolationError,
    IndexContentionError,
    NotNullViolationError,
    UniqueConstraintError,
)
from diffract.core.data.metadata.interface import JoinSpec
from diffract.core.data.metadata.sqlite_index import (
    IN_MEMORY_DATABASE,
    SQLiteMetadataIndex,
)

pytestmark = pytest.mark.unit


def test_in_memory_sentinel_creates_no_filesystem_entry(temp_dir: Path) -> None:
    """The sentinel selects an in-memory database, so the index must not treat it
    as a file: no directory or file named ':memory:' appears on disk."""
    index = SQLiteMetadataIndex(IN_MEMORY_DATABASE)
    try:
        index.connect()
    finally:
        index.close()

    assert not (Path.cwd() / IN_MEMORY_DATABASE).exists()
    assert not (temp_dir / IN_MEMORY_DATABASE).exists()


def test_file_backed_index_creates_its_parent_directory(temp_dir: Path) -> None:
    """An ordinary path is still a path: the index creates the parent directory
    and the database file."""
    db_path = temp_dir / "nested" / "index.db"

    index = SQLiteMetadataIndex(str(db_path))
    try:
        index.connect()
    finally:
        index.close()

    assert db_path.parent.is_dir()
    assert db_path.exists()


@pytest.fixture
def linked_index(tmp_path: Path) -> Iterator[SQLiteMetadataIndex]:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.define_table("samples", {"name": str, "description": str})
        with closing(sqlite3.connect(path)) as db:
            db.execute(
                "CREATE TABLE members (uid TEXT PRIMARY KEY, param_uid TEXT NOT NULL "
                "REFERENCES samples(uid) ON DELETE CASCADE DEFERRABLE "
                "INITIALLY DEFERRED, json_data TEXT)"
            )
        index.define_table("members", {"param_uid": str})
        yield index


def test_invalid_membership_rolls_back_batch_on_reopened_index(
    linked_index: SQLiteMetadataIndex,
) -> None:
    index = linked_index
    index.close()

    def commit_invalid_batch() -> None:
        with index:
            index.insert("samples", "uncommitted", name="encoder")
            index.insert("members", "dangling", param_uid="missing")
            index.insert("samples", "unrelated")

    with pytest.raises(ForeignKeyViolationError, match="FOREIGN KEY") as error:
        commit_invalid_batch()
    assert error.value.statement_table == "members"
    with index:
        assert index.get("samples", "uncommitted") is None
        assert index.get("members", "dangling") is None
        index.insert("members", "edge", param_uid="valid")
        index.insert("samples", "valid", name="encoder")
    assert index.get("members", "edge")["param_uid"] == "valid"


def test_upsert_replaces_parameter_fields_and_keeps_membership(
    linked_index: SQLiteMetadataIndex,
) -> None:
    index = linked_index
    index.insert("samples", "p", name="old", description="label", run=7)
    index.insert("members", "edge", param_uid="p")
    index.upsert("samples", "p", name="new")
    row = index.get("samples", "p")
    assert row["name"] == "new"
    assert row["description"] is None
    assert "run" not in row
    assert index.get("members", "edge")["param_uid"] == "p"
    index.delete("samples", "p")
    assert index.get("members", "edge") is None


def test_upsert_requires_a_single_table_identifier(
    linked_index: SQLiteMetadataIndex,
) -> None:
    with pytest.raises(ValueError, match="Invalid metadata identifier"):
        linked_index.upsert("main.samples", "p", name="encoder")
    assert linked_index.get("samples", "p") is None


def test_another_thread_can_retry_after_busy_begin(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path), timeout=0)) as index:
        index.define_table("samples", {"name": str})
        with closing(sqlite3.connect(path)) as other_writer:
            other_writer.execute("BEGIN IMMEDIATE")
            with pytest.raises(IndexContentionError) as error, index:
                pytest.fail("a second writer acquired the reserved lock")
            assert error.value.__cause__.sqlite_errorcode == sqlite3.SQLITE_BUSY

        def retry() -> None:
            with index:
                index.insert("samples", "p", name="encoder")

        worker = Thread(target=retry, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "failed BEGIN kept the index locked"
        assert index.get("samples", "p")["name"] == "encoder"


def test_clearing_linked_tables_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.define_table("samples", {"name": str})
        with closing(sqlite3.connect(path)) as db:
            db.execute(
                "CREATE TABLE members (uid TEXT PRIMARY KEY, param_uid TEXT "
                "REFERENCES samples(uid), json_data TEXT)"
            )
        index.define_table("members", {"param_uid": str})
        index.insert("samples", "p", name="encoder")
        index.insert("members", "edge", param_uid="p")

        with pytest.raises(ForeignKeyViolationError, match="FOREIGN KEY") as error:
            index.clear("samples")
        assert error.value.statement_table == "samples"
        assert index.list_uids("samples") == ["p"]
        assert index.list_uids("members") == ["edge"]
        with index:
            index.clear("samples")
            index.clear("members")
        index.insert("samples", "q", name="head")
        index.insert("members", "edge2", param_uid="q")
        index.clear()
        assert index.list_uids("samples") == []
        assert index.list_uids("members") == []


@pytest.mark.parametrize("threshold", [2, 512])
def test_read_comparison_members_with_roles_and_positions(
    tmp_path: Path, threshold: int
) -> None:
    path = tmp_path / "metadata.db"
    with closing(
        SQLiteMetadataIndex(str(path), in_temp_table_threshold=threshold)
    ) as index:
        index.define_table("samples", {"name": str, "step": int})
        index.insert("samples", "p2", name="encoder", step=2, label="after")
        index.insert("samples", "p1", name="encoder", step=0, label="before")
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            db.execute(
                "CREATE TABLE comparison_members (aggregate_uid TEXT, member_uid TEXT "
                "REFERENCES samples(uid), ordinal INTEGER, role TEXT, "
                "PRIMARY KEY(aggregate_uid, member_uid, role)) WITHOUT ROWID"
            )
        fields = ("aggregate_uid", "member_uid", "ordinal", "role")
        members = [
            ("pair", "p1", 1, "target"),
            ("pair", "p2", 0, "source"),
            ("unrelated", "p1", None, "member"),
        ]
        assert index.insert_rows(
            "comparison_members",
            [dict(zip(fields, row, strict=True)) for row in members],
        ) == len(members)
        statements = []
        index._conn.set_trace_callback(statements.append)
        result = index.query_joined_rows(
            "comparison_members",
            JoinSpec("samples", "member_uid", "uid", "param"),
            columns=["base.role", "base.ordinal", "param.step"],
            where={"base.aggregate_uid": "pair"},
            where_in={"param.uid": ["p1", "p2", "p1"]},
            order_by=["base.ordinal"],
        )
        assert result == [
            {"base.role": "source", "base.ordinal": 0, "param.step": 2},
            {"base.role": "target", "base.ordinal": 1, "param.step": 0},
        ]
        assert sum(stmt.startswith("SELECT ") for stmt in statements) == 1
        with pytest.raises(ValueError, match="None"):
            index.query_rows("comparison_members", where={"ordinal": None})
        assert index.query_rows(
            "samples", columns=["name"], where={"name": "encoder"}, distinct=True
        ) == [{"name": "encoder"}]
        assert set(index.query("samples", where_like={"name": "ENC%"})) == {
            "p1",
            "p2",
        }
        (row,) = index.query_rows("samples", where={"uid": "p2"})
        assert row["json_data"] == '{"label": "after"}'
        assert index.query_rows("comparison_members", where_in={"role": []}) == []


def test_point_lookup_and_upsert_preserve_keys_and_transaction_rollback(tmp_path):
    with closing(SQLiteMetadataIndex(str(tmp_path / "index.db"))) as index:
        index.define_table("samples", {"name": str, "weight": int})
        key = {"uid": "p1"}

        def read():
            return index.find_row("samples", columns=("name", "weight"), where=key)

        assert read() is None
        index.upsert_row("samples", key=key, values={"name": "first", "weight": 4})
        index.upsert_row("samples", key=key, values={"name": "second"})
        assert read() == {"name": "second", "weight": 4}

        def aborted_write():
            index.upsert_row("samples", key=key, values={"weight": 9})
            assert read() == {"name": "second", "weight": 9}
            raise RuntimeError("roll back")

        with pytest.raises(RuntimeError, match="roll back"), index:
            aborted_write()
        assert read() == {"name": "second", "weight": 4}
        with pytest.raises(ValueError, match="identifier"):
            index.find_row("samples", columns=("name FROM samples",), where=key)
        with pytest.raises(ValueError, match="identifier"):
            index.upsert_row("samples", key=key, values={"weight=?": 9})


def test_missing_related_values_checks_every_distinct_output(tmp_path):
    with closing(
        SQLiteMetadataIndex(str(tmp_path / "metadata.db"), in_temp_table_threshold=2)
    ) as index:
        index.define_table("owners", {})
        index.define_table("outputs", {"owner_uid": str, "field": str, "active": int})
        for uid in ("full", "partial", "empty"):
            index.insert("owners", uid)
        rows = [
            ("full", "x", 1),
            ("full", "y", 1),
            ("partial", "x", 1),
            ("partial", "x", 1),
            ("partial", "y", 0),
        ]
        for i, (uid, field, active) in enumerate(rows):
            index.insert("outputs", str(i), owner_uid=uid, field=field, active=active)

        def missing(*fields):
            return set(
                index.uids_missing_related_values(
                    "owners",
                    JoinSpec("outputs", "uid", "owner_uid"),
                    column="field",
                    required=fields,
                    where={"active": 1},
                )
            )

        assert missing() == set()
        assert missing("x") == {"empty"}
        assert missing("y", "x", "x") == {"partial", "empty"}


def test_large_filters_return_each_uid_once_and_release_temp_tables(
    tmp_path: Path,
) -> None:
    with closing(
        SQLiteMetadataIndex(str(tmp_path / "metadata.db"), in_temp_table_threshold=2)
    ) as index:
        index.define_table("samples", {})
        index.insert("samples", "1")
        assert index.query("samples", where_in={"uid": ["1", 1]}) == ["1"]
        with pytest.raises(sqlite3.ProgrammingError, match="binding"):
            index.query_rows("samples", where_in={"uid": ["1", object()]})
        assert (
            index._conn.execute("SELECT name FROM sqlite_temp_master").fetchall() == []
        )


def test_bulk_insert_arbitrates_natural_key_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.connect()
        with closing(sqlite3.connect(path)) as db:
            db.execute(
                "CREATE TABLE samples (uid TEXT PRIMARY KEY NOT NULL, "
                "instance_uid TEXT NOT NULL, step INTEGER NOT NULL CHECK(step >= 0), "
                "name TEXT NOT NULL, UNIQUE(instance_uid, step, name))"
            )
        key = ("instance_uid", "step", "name")
        original = {"uid": "p", "instance_uid": "m", "step": 0, "name": "encoder"}
        other = {**original, "uid": "other", "name": "head"}
        duplicate = {**original, "uid": "collision"}
        assert index.insert_rows("samples", [original]) == 1
        with index:
            with pytest.raises(UniqueConstraintError) as error:
                index.insert_rows("samples", [other, duplicate])
            assert error.value.columns == key
            assert error.value.table == error.value.statement_table == "samples"
            assert isinstance(error.value.__cause__, sqlite3.IntegrityError)
            assert index.query_rows("samples") == [original]
            assert (
                index.insert_rows(
                    "samples",
                    [duplicate, other],
                    on_conflict="ignore",
                    conflict_target=key,
                )
                == 1
            )
        with pytest.raises(UniqueConstraintError) as error:
            index.insert_rows(
                "samples",
                [{**original, "name": "different"}],
                on_conflict="ignore",
                conflict_target=key,
            )
        assert error.value.kind == "primary_key"
        assert error.value.columns == ("uid",)
        assert (
            index.update_rows("samples", where={"uid": "other"}, values={"step": 2})
            == 1
        )
        with pytest.raises(CheckConstraintError) as checked:
            index.update_rows("samples", where={"uid": "other"}, values={"step": -1})
        assert checked.value.kind == "check"
        detail = "step >= 0" if sqlite3.sqlite_version_info >= (3, 34) else "samples"
        assert str(checked.value.__cause__) == f"CHECK constraint failed: {detail}"
        with pytest.raises(NotNullViolationError) as error:
            index.insert_rows(
                "samples",
                [{**other, "uid": "null", "name": None}],
                on_conflict="ignore",
                conflict_target=key,
            )
        assert error.value.columns == ("name",)
        assert index.delete_rows("samples", where_in={"uid": []}) == 0
        assert index.delete_rows("samples", where={"step": 2}) == 1
        assert index.query_rows("samples") == [original]


def test_deferred_row_error_keeps_cause_and_allows_retry(
    linked_index: SQLiteMetadataIndex,
) -> None:
    index = linked_index
    row = {"uid": "edge", "param_uid": "unknown"}
    with pytest.raises(ForeignKeyViolationError) as error:
        index.insert_rows("members", [row])
    assert error.value.statement_table == "members"
    assert error.value.__cause__.sqlite_errorname == "SQLITE_CONSTRAINT_FOREIGNKEY"
    assert index.query_rows("members") == []
    with index:
        index.insert_rows("members", [row])
        index.insert_rows("samples", [{"uid": "unknown", "name": "head"}])
    assert index.query_rows("members", columns=["uid", "param_uid"]) == [row]


def test_unrecognized_unique_constraint_preserves_sqlite_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.db"
    with closing(SQLiteMetadataIndex(str(path))) as index:
        index.define_table("samples", {"name": str})
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE UNIQUE INDEX folded_names ON samples(lower(name))")
        index.insert_rows("samples", [{"uid": "p", "name": "Head"}])
        with pytest.raises(UniqueConstraintError) as error:
            index.insert_rows("samples", [{"uid": "q", "name": "HEAD"}])
        assert error.value.kind == "unique"
        assert error.value.table is None
        assert error.value.columns == ()
        assert error.value.__cause__.sqlite_errorname == "SQLITE_CONSTRAINT_UNIQUE"


def test_join_keys_and_destructive_predicate_guards(linked_index):
    index = linked_index
    index.insert("samples", "p", name="a", provenance=1)
    index.insert("members", "edge", param_uid="p", provenance=2)
    join = JoinSpec("samples", "param_uid", "uid")
    assert index.query_joined_rows(
        "members", join, columns=["base.json_data", "j.json_data"]
    ) == [{"base.json_data": '{"provenance": 2}', "j.json_data": '{"provenance": 1}'}]
    for kwargs in (
        {"columns": ["uid"]},
        {"where": {"uid": "p"}},
        {"order_by": ["uid"]},
        {"where_in": {"uid": ["p"]}},
    ):
        with pytest.raises(ValueError, match="qualified"):
            index.query_joined_rows(
                "members", join, **({"columns": ["base.uid"]} | kwargs)
            )
    with pytest.raises(ValueError, match="predicate"):
        index.delete_rows("samples", where={}, where_in={})
    with pytest.raises(ValueError, match="predicate"):
        index.update_rows("samples", where={}, values={"name": "lost"})
    with pytest.raises(ValueError, match="None"):
        index.delete_rows("samples", where_in={"uid": ["p", None]})
    with pytest.raises(ValueError, match="conflict_target"):
        index.insert_rows("samples", [{"uid": "p"}], on_conflict="ignore")
    assert index.get("samples", "p")["name"] == "a"


def test_clear_preserves_uid_audit_and_hash_scheme(linked_index):
    index = linked_index
    row = {"entity_kind": "instance", "old_uid": "old", "new_uid": "new"}
    index.insert_rows("uid_map", [row])
    for table in ("UID_MAP", "schema_meta", None):
        index.clear(table)
        assert index.query_rows("uid_map") == [row]
        assert index.query_rows("schema_meta", columns=["hash_scheme_version"]) == [
            {"hash_scheme_version": 1}
        ]


def test_trigger_constraint_retains_unknown_diagnostic(linked_index):
    index = linked_index
    index._conn.execute(
        "CREATE TRIGGER reject_insert BEFORE INSERT ON samples "
        "BEGIN SELECT RAISE(ABORT, 'rejected'); END"
    )
    with pytest.raises(ConstraintViolationError) as error:
        index.insert_rows("samples", [{"uid": "p"}])
    assert type(error.value) is ConstraintViolationError
    assert error.value.kind == "unknown"
    assert error.value.statement_table == "samples"
    assert error.value.table is None
    assert error.value.columns == ()
    assert error.value.__cause__.sqlite_errorname == "SQLITE_CONSTRAINT_TRIGGER"


def test_multirow_insert_rolls_back_earlier_pages_and_counts_only_new_rows(
    linked_index,
):
    rows = [{"uid": f"p{i}", "name": f"param{i}"} for i in range(300)]
    with linked_index:
        assert linked_index.insert_rows("samples", rows[:2]) == 2
        with pytest.raises(UniqueConstraintError):
            linked_index.insert_rows("samples", [*rows[2:], rows[0]])
        assert linked_index.count("samples") == 2
        assert (
            linked_index.insert_rows(
                "samples", rows, on_conflict="ignore", conflict_target=["uid"]
            )
            == 298
        )
    assert linked_index.count("samples") == 300
