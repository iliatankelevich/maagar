"""The metadata walk for foreign keys that would let one tenant reach another's rows."""

from __future__ import annotations

import pytest
from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint, Integer, MetaData, String, Table
from sqlalchemy.exc import NoReferencedTableError

from maagar import UnanchoredKey, unanchored_foreign_keys


def _parent(md: MetaData, name: str = "members", tenant: str = "tenant_id") -> Table:
    return Table(
        name,
        md,
        Column(tenant, String, primary_key=True),
        Column("id", Integer, primary_key=True),
    )


def test_a_single_column_key_between_tenant_tables_is_reported() -> None:
    md = MetaData()
    _parent(md)
    Table(
        "messages",
        md,
        Column("tenant_id", String),
        Column("member_id", Integer, ForeignKey("members.id")),
    )
    assert unanchored_foreign_keys(md) == [
        UnanchoredKey("messages", None, ("member_id",), "members", ("id",))
    ]


def test_a_composite_key_pairing_the_tenant_passes() -> None:
    md = MetaData()
    _parent(md)
    Table(
        "messages",
        md,
        Column("tenant_id", String),
        Column("member_id", Integer),
        ForeignKeyConstraint(
            ["tenant_id", "member_id"], ["members.tenant_id", "members.id"], name="fk_m"
        ),
    )
    assert unanchored_foreign_keys(md) == []


def test_a_composite_key_pairing_the_tenant_with_another_column_is_reported() -> None:
    md = MetaData()
    _parent(md)
    Table(
        "messages",
        md,
        Column("tenant_id", String),
        Column("member_id", Integer),
        ForeignKeyConstraint(["tenant_id", "member_id"], ["members.id", "members.tenant_id"]),
    )
    assert [k.table for k in unanchored_foreign_keys(md)] == ["messages"]


def test_a_key_into_a_table_without_the_tenant_is_ignored() -> None:
    md = MetaData()
    Table("plans", md, Column("id", Integer, primary_key=True))
    Table(
        "members",
        md,
        Column("tenant_id", String),
        Column("plan_id", Integer, ForeignKey("plans.id")),
    )
    assert unanchored_foreign_keys(md) == []


def test_a_key_from_a_table_without_the_tenant_into_a_tenant_table_is_reported() -> None:
    md = MetaData()
    _parent(md)
    Table("audit", md, Column("member_id", Integer, ForeignKey("members.id")))
    assert [k.table for k in unanchored_foreign_keys(md)] == ["audit"]


def test_a_self_reference_is_held_to_the_same_rule() -> None:
    bad, good = MetaData(), MetaData()
    for md, constraint in (
        (bad, ForeignKeyConstraint(["parent_id"], ["nodes.id"])),
        (
            good,
            ForeignKeyConstraint(["tenant_id", "parent_id"], ["nodes.tenant_id", "nodes.id"]),
        ),
    ):
        Table(
            "nodes",
            md,
            Column("tenant_id", String, primary_key=True),
            Column("id", Integer, primary_key=True),
            Column("parent_id", Integer),
            constraint,
        )
    assert [k.table for k in unanchored_foreign_keys(bad)] == ["nodes"]
    assert unanchored_foreign_keys(good) == []


def test_the_tenant_column_name_is_configurable() -> None:
    md = MetaData()
    _parent(md, tenant="org")
    Table(
        "messages",
        md,
        Column("org", String),
        Column("member_id", Integer, ForeignKey("members.id")),
    )
    assert unanchored_foreign_keys(md) == []
    assert len(unanchored_foreign_keys(md, column="org")) == 1


def test_a_composite_key_on_a_custom_tenant_column_passes() -> None:
    md = MetaData()
    _parent(md, tenant="org")
    Table(
        "messages",
        md,
        Column("org", String),
        Column("member_id", Integer),
        ForeignKeyConstraint(["org", "member_id"], ["members.org", "members.id"]),
    )
    assert unanchored_foreign_keys(md, column="org") == []


def test_a_schema_qualified_tenant_table_is_still_judged() -> None:
    md = MetaData()
    Table(
        "members",
        md,
        Column("tenant_id", String, primary_key=True),
        Column("id", Integer, primary_key=True),
        schema="app",
    )
    Table(
        "messages",
        md,
        Column("tenant_id", String),
        Column("member_id", Integer, ForeignKey("app.members.id")),
        schema="app",
    )
    assert unanchored_foreign_keys(md) == [
        UnanchoredKey("app.messages", None, ("member_id",), "app.members", ("id",))
    ]


def test_one_anchored_key_does_not_excuse_another_on_the_same_table() -> None:
    md = MetaData()
    _parent(md)
    _parent(md, name="documents")
    Table(
        "messages",
        md,
        Column("tenant_id", String),
        Column("member_id", Integer),
        Column("document_id", Integer, ForeignKey("documents.id")),
        ForeignKeyConstraint(["tenant_id", "member_id"], ["members.tenant_id", "members.id"]),
    )
    assert [(k.table, k.referred_table) for k in unanchored_foreign_keys(md)] == [
        ("messages", "documents")
    ]


def test_a_key_to_a_table_outside_the_metadata_refuses_rather_than_passes() -> None:
    md = MetaData()
    _parent(md)
    Table("messages", md, Column("tenant_id", String), Column("x", Integer, ForeignKey("ghost.id")))
    with pytest.raises(NoReferencedTableError):
        unanchored_foreign_keys(md)


def test_the_report_is_sorted_and_names_the_constraint() -> None:
    md = MetaData()
    _parent(md)
    for name in ("zeta", "alpha"):
        Table(
            name,
            md,
            Column("tenant_id", String),
            Column("member_id", Integer),
            ForeignKeyConstraint(["member_id"], ["members.id"], name=f"fk_{name}"),
        )
    found = unanchored_foreign_keys(md)
    assert [(k.table, k.name) for k in found] == [("alpha", "fk_alpha"), ("zeta", "fk_zeta")]
