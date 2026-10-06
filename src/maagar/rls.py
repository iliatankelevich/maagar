"""Row level security: the policies, and the two ways they silently stop working.

RLS is the **backstop**, never the primary mechanism. The primary mechanism is that every query
carries its ``tenant_id`` predicate, because that is the one an isolation suite exercises directly.
RLS is what makes a *forgotten* predicate harmless instead of catastrophic. Measured overhead on
realistic workloads is a few percent.

⚠️ **Postgres exempts two kinds of connection from every policy in the database:**

* a table's **owner**, unless the table is marked ``FORCE ROW LEVEL SECURITY``; and
* any **superuser** or role holding ``BYPASSRLS`` — and ``FORCE`` does **not** help there.

Both are deployment configuration, not code, which is why they are so easy to ship: every policy is
bypassed, and every test still passes, in exactly the same green way as a correct system. That is
not hypothetical. The first run of a consumer's isolation suite connected as the Postgres image's
``POSTGRES_USER`` — a superuser — and watched all six cross-tenant assertions pass while policies
did precisely nothing.

Hence :class:`~maagar.placement.Placement` carrying two DSNs, and hence ``FORCE`` being emitted here
unconditionally. What this module cannot do is check that the *application* DSN it was handed is
unprivileged; that remains an obligation of the deployment. :func:`assert_unprivileged` gives a
caller a way to assert it at startup rather than discover it after a leak.

Policies are applied in **both** placements, including one-database-per-tenant where they are
strictly redundant. Two reasons: the code path stays identical in both, so the isolation suite
proves the same thing about each; and if a directory ever resolves a tenant to the wrong database,
the policy denies the read that the missing rows would otherwise have merely... not returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import AsyncConnection

#: The session variable the policies read. ``current_setting(..., true)`` returns NULL rather than
#: raising when it is unset, so a connection that never announced a tenant sees **zero rows**
#: rather than erroring — fail closed. A connection that forgot is invisible either way; this way
#: is invisible *and* harmless.
TENANT_SETTING = "app.tenant_id"

POLICY_NAME = "tenant_isolation"


def tenant_scoped_tables(metadata: MetaData, column: str = "tenant_id") -> list[str]:
    """Every table carrying the tenant column — i.e. every table a policy must cover.

    Derived from the metadata rather than hand-listed, so a table added tomorrow is covered the
    moment it exists. A hand-maintained list is the thing that goes stale in silence, and it goes
    stale in the direction of *less* coverage.
    """
    return [name for name, table in metadata.tables.items() if column in table.columns]


class UnanchoredKey(NamedTuple):
    """A foreign key into a tenant table that does not carry the tenant."""

    table: str
    name: str | None
    columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]


def unanchored_foreign_keys(metadata: MetaData, column: str = "tenant_id") -> list[UnanchoredKey]:
    """Every foreign key into a tenant table that does not pair the tenant column with itself.

    Postgres runs referential integrity with row security bypassed, so ``REFERENCES members(id)``
    accepts another tenant's member on insert and reaches another tenant's rows on delete. Only a
    composite key, ``(tenant_id, x) -> (tenant_id, id)``, closes that. A key is anchored when at
    least one of its elements maps the tenant column to the tenant column; a composite key that
    merely *contains* the column, paired with some other one, is not.

    Judged by the table a key points **at**, not the one it lives on: a key from a table without
    the tenant column into a tenant table is reported too, since it has nothing to pair. A
    self-reference is held to the same rule. Tables are identified as :func:`tenant_scoped_tables`
    identifies them — by metadata key, so schema-qualified — and the column by its key likewise.

    Raises SQLAlchemy's ``NoReferencedTableError`` for a key whose target is not in ``metadata``:
    an unknown target might be a tenant table, so skipping it would pass a key nobody judged.
    """
    scoped = set(tenant_scoped_tables(metadata, column))
    return sorted(
        (
            UnanchoredKey(
                table=table.key,
                name=key.name if isinstance(key.name, str) else None,
                columns=tuple(c.name for c in key.columns),
                referred_table=key.referred_table.key,
                referred_columns=tuple(element.column.name for element in key.elements),
            )
            for table in metadata.tables.values()
            for key in table.foreign_key_constraints
            if key.referred_table.key in scoped
            and not any(
                element.parent.key == column and element.column.key == column
                for element in key.elements
            )
        ),
        key=lambda k: (k.table, k.columns, k.referred_table, k.referred_columns, k.name or ""),
    )


def policy_statements(
    tables: Sequence[str],
    *,
    column: str = "tenant_id",
    setting: str = TENANT_SETTING,
) -> list[str]:
    """The DDL that puts the tenant policy on each table. **One source, two callers.**

    Provisioning runs it on an async connection (:func:`apply_policies`); an Alembic migration runs
    it on a sync one. Generating the strings here rather than in either caller is what keeps the
    policy in a migration from drifting away from the policy applied at provisioning time — a drift
    nothing would detect, because both paths would keep succeeding.

    Idempotent: the policy is dropped and recreated, so re-running after a schema change converges
    rather than erroring.
    """
    out: list[str] = []
    for table in tables:
        # Identifiers cannot be bound as parameters in DDL, so the safety argument here is
        # provenance, not quoting: `tables` comes from SQLAlchemy metadata and `column`/`setting`
        # from this package's own configuration. Neither is ever user input.
        quoted = f'"{table}"'
        out += [
            f"ALTER TABLE {quoted} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE {quoted} FORCE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS {POLICY_NAME} ON {quoted}",
            (
                f"CREATE POLICY {POLICY_NAME} ON {quoted} "
                f"USING ({column} = current_setting('{setting}', true)) "
                f"WITH CHECK ({column} = current_setting('{setting}', true))"
            ),
        ]
    return out


async def apply_policies(
    conn: AsyncConnection,
    tables: Sequence[str],
    *,
    column: str = "tenant_id",
    setting: str = TENANT_SETTING,
) -> None:
    """Enable, force and (re)create the tenant policy. Runs on an **admin** connection — DDL."""
    for statement in policy_statements(tables, column=column, setting=setting):
        await conn.execute(text(statement))


async def assert_unprivileged(conn: AsyncConnection) -> None:
    """Raise if this connection would bypass RLS. Call once at startup, on the *application* DSN.

    Catches the superuser/``BYPASSRLS`` case, which is the one ``FORCE`` cannot save you from. It
    does **not** catch table ownership — that is what ``FORCE`` is for, and it is applied above.
    """
    row = (
        await conn.execute(
            text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )
    ).first()
    if row is None:  # pragma: no cover - current_user always has a pg_roles row
        return
    is_super, bypasses = bool(row[0]), bool(row[1])
    if is_super or bypasses:
        flags = " and ".join(
            f for f in ("SUPERUSER" if is_super else "", "BYPASSRLS" if bypasses else "") if f
        )
        raise PermissionError(
            f"the application role is {flags}, so row level security is bypassed on every table "
            "and the isolation backstop is decorative. Connect as a role that is neither a "
            "superuser nor BYPASSRLS, and that does not own the tables."
        )
