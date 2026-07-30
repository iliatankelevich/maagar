"""Placement resolution — and the credential boundary that makes the descriptor rule safe.

The interesting assertions are all about what a control plane is **not** allowed to hand out.
"""

from __future__ import annotations

import pytest

from maagar import (
    CatalogDirectory,
    DatabasePerTenant,
    Isolation,
    SharedDatabase,
    Tenant,
    TenantRecord,
    UnknownTenant,
)

ALPHA = Tenant.attested("fam-alpha")
BETA = Tenant.attested("fam-beta")

APP = "postgresql+asyncpg://kip_app:secret@db-1:5432/postgres"
ADMIN = "postgresql+asyncpg://owner:othersecret@db-1:5432/postgres"


async def test_a_tenant_record_carries_no_credential() -> None:
    """⚠️ **The point of the descriptor rule, asserted on the type itself.**

    If a field were ever added here that could hold a DSN, the control plane would be holding a
    credential to every service's database and the separation would be over — not overruled, just
    quietly meaningless. This test is what makes adding that field a conversation.
    """
    record = TenantRecord(isolation=Isolation.isolated, instance="mem-1", database="kip_alpha")
    values = " ".join(str(getattr(record, f)) for f in record.__slots__)
    assert "://" not in values, "a placement descriptor must never contain a URL"
    assert "secret" not in values
    assert set(record.__slots__) == {"isolation", "instance", "database"}


async def test_the_catalog_composes_a_descriptor_with_a_local_credential() -> None:
    """Neither half is sufficient alone — which is the security property, not a nicety."""

    async def lookup(t: Tenant) -> TenantRecord:
        return TenantRecord(isolation=Isolation.isolated, instance="mem-1", database=f"kip_{t.id}")

    directory = CatalogDirectory(lookup=lookup, instances={"mem-1": (APP, ADMIN)})
    placement = await directory.locate(ALPHA)

    assert placement.isolation is Isolation.isolated
    assert "/kip_fam-alpha" in placement.dsn
    assert "kip_app:secret" in placement.dsn, "the credential comes from local config"
    assert "owner:othersecret" in placement.admin_dsn, "and the two roles stay distinct"


async def test_an_unknown_instance_fails_loudly() -> None:
    """A tenant placed somewhere this service has no credential for is a configuration fault.

    Failing is the only safe answer: the alternative is falling back to *some* database, which is
    exactly the cross-tenant write the whole design exists to make impossible.
    """

    async def lookup(t: Tenant) -> TenantRecord:
        return TenantRecord(isolation=Isolation.isolated, instance="mem-9", database="x")

    directory = CatalogDirectory(lookup=lookup, instances={"mem-1": (APP, ADMIN)})
    with pytest.raises(UnknownTenant, match="mem-9"):
        await directory.locate(ALPHA)


async def test_a_pooled_record_ignores_the_database_name() -> None:
    """Under pooled, the shared database is whatever the instance credential points at."""

    async def lookup(t: Tenant) -> TenantRecord:
        return TenantRecord(isolation=Isolation.pooled, instance="mem-1", database="ignored")

    directory = CatalogDirectory(lookup=lookup, instances={"mem-1": (APP, ADMIN)})
    placement = await directory.locate(ALPHA)
    assert placement.isolation is Isolation.pooled
    assert placement.dsn == APP


async def test_fleet_operations_refuse_to_guess_the_roster() -> None:
    """⚠️ A fleet migration is the last place to infer which tenants exist.

    ``DatabasePerTenant`` can discover a roster from ``pg_database`` by prefix, and that is
    convenient and *wrong* for anything authoritative: it invents tenants from stray databases and
    misses tenants not yet provisioned.
    """

    async def lookup(t: Tenant) -> TenantRecord:
        return TenantRecord(isolation=Isolation.pooled, instance="mem-1")

    directory = CatalogDirectory(lookup=lookup, instances={"mem-1": (APP, ADMIN)})
    with pytest.raises(UnknownTenant, match="roster"):
        await directory.roster()


async def test_database_names_stay_inside_postgres_identifier_limits() -> None:
    """``Tenant.attested`` caps ids at 40 bytes so ``<prefix><id>`` fits in 63.

    Over that limit Postgres **truncates silently**, and two different tenants can land on one
    database — a cross-tenant merge produced by a naming convention.
    """
    directory = DatabasePerTenant(instance_dsn=APP, instance_admin_dsn=ADMIN, prefix="kipchat_")
    longest = Tenant.attested("a" * 40)
    assert len(directory.database_name(longest).encode()) <= 63


async def test_the_shared_directory_sends_every_tenant_to_one_place() -> None:
    directory = SharedDatabase(dsn=APP, admin_dsn=ADMIN, tenants=(ALPHA, BETA))
    assert await directory.locate(ALPHA) == await directory.locate(BETA)
    assert set(await directory.roster()) == {ALPHA, BETA}
