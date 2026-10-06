"""The metadata walk for foreign keys that would let one tenant reach another's rows."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint, Integer, MetaData, String, Table

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
