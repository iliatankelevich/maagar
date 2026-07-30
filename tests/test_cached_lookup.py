"""The cache in front of the control plane — and the failure rule that is the whole point.

These are the assertions that distinguish this from a TTL dict. A cache that expires and then
propagates the control plane's error is *worse* than no cache during an outage: it works for exactly
one TTL and then takes every tenant down at once, which is a far harder failure to diagnose than one
that never worked.
"""

from __future__ import annotations

import pytest

from maagar import CachedLookup, Isolation, Tenant, TenantRecord, UnknownTenant

ALPHA = Tenant.attested("fam-alpha")
BETA = Tenant.attested("fam-beta")

POOLED = TenantRecord(isolation=Isolation.pooled, instance="mem-1", database=None)
ISOLATED = TenantRecord(isolation=Isolation.isolated, instance="mem-1", database="kip_alpha")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Control:
    """A stand-in control plane whose availability the test drives."""

    def __init__(self, record: TenantRecord = POOLED) -> None:
        self.record = record
        self.up = True
        self.calls = 0

    async def __call__(self, tenant: Tenant) -> TenantRecord:
        self.calls += 1
        if not self.up:
            raise UnknownTenant("control plane unreachable")
        return self.record


async def test_a_fresh_entry_does_not_call_the_control_plane() -> None:
    control = Control()
    lookup = CachedLookup(control, ttl_seconds=100, clock=FakeClock())

    assert await lookup(ALPHA) == POOLED
    assert await lookup(ALPHA) == POOLED

    assert control.calls == 1, "the second resolve must be served from cache"


async def test_the_entry_is_refreshed_once_the_ttl_passes() -> None:
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, clock=clock)

    await lookup(ALPHA)
    clock.now = 101
    await lookup(ALPHA)

    assert control.calls == 2


async def test_an_outage_serves_a_stale_record_rather_than_failing() -> None:
    """⚠️ **The rule the whole class exists for.**

    A tenant already resolved keeps being served through a control-plane outage of any length.
    Without this, the platform stays up for one TTL after mom dies and then stops serving everyone
    simultaneously — which is the outage, merely postponed and made harder to attribute.
    """
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 10_000  # long past the TTL

    assert await lookup(ALPHA) == POOLED
    assert lookup.served_stale == 1, "serving stale must be counted — it is an alertable condition"


async def test_an_unresolved_tenant_fails_during_an_outage() -> None:
    """The other half of *"no **new** tenants served, never nobody served"*.

    Failing here is the correct behaviour and the alternative is the bug: inventing a placement for
    a tenant nobody can look up means guessing which database to write to, and a guessed placement
    is a cross-tenant write.
    """
    control = Control()
    control.up = False
    lookup = CachedLookup(control, clock=FakeClock())

    with pytest.raises(UnknownTenant):
        await lookup(BETA)


async def test_a_failed_lookup_is_not_remembered() -> None:
    """A tenant registered but not yet placed resolves as soon as it is placed.

    Provisioning is eventually consistent, so "not there yet" is a normal state that fixes itself.
    Caching the miss would turn a signup seconds from working into one that fails for a full TTL.
    """
    control = Control()
    control.up = False
    lookup = CachedLookup(control, ttl_seconds=10_000, clock=FakeClock())

    with pytest.raises(UnknownTenant):
        await lookup(ALPHA)

    control.up = True
    assert await lookup(ALPHA) == POOLED, "the miss must not have been cached"


async def test_invalidate_forces_a_refresh() -> None:
    """The escape hatch for the one case staleness is genuinely wrong: a deliberate migration."""
    control = Control()
    lookup = CachedLookup(control, ttl_seconds=10_000, clock=FakeClock())
    assert await lookup(ALPHA) == POOLED

    control.record = ISOLATED
    assert await lookup(ALPHA) == POOLED, "still cached, correctly"

    lookup.invalidate(ALPHA)
    assert await lookup(ALPHA) == ISOLATED


async def test_a_catalog_directory_can_provision() -> None:
    """⚠️ **The production directory must be able to create databases, and once could not.**

    ``SupportsProvisioning`` was originally a synchronous ``maintenance_dsn`` property plus a
    synchronous ``database_name`` — a shape only a directory that already knows every answer can
    satisfy. :class:`CatalogDirectory` has to *ask* the control plane, so it silently failed the
    ``isinstance`` check and every isolated tenant it routed raised ``ProvisioningError`` at
    ``provision()``. The static directories used in tests implemented it fine, so nothing noticed.

    The DSN returned must be the **instance**, not the tenant's database: ``CREATE DATABASE`` cannot
    run from inside its own target.
    """
    from maagar import CatalogDirectory
    from maagar.store import SupportsProvisioning

    directory = CatalogDirectory(
        lookup=Control(ISOLATED),
        instances={"mem-1": ("postgresql://app@host/shared", "postgresql://owner@host/shared")},
    )

    assert isinstance(directory, SupportsProvisioning)
    dsn, database = await directory.provisioning_target(ALPHA)
    assert database == "kip_alpha"
    assert dsn.endswith("/shared"), "provisioning must connect to the instance, not to the target"


async def test_provisioning_a_pooled_tenant_is_refused_with_a_reason() -> None:
    """A pooled tenant has no database of its own, so asking for one is a caller error rather than
    something to paper over — and papering over it means creating a database nobody routes to."""
    from maagar import CatalogDirectory
    from maagar.errors import ProvisioningError

    directory = CatalogDirectory(
        lookup=Control(POOLED),
        instances={"mem-1": ("postgresql://app@host/shared", "postgresql://owner@host/shared")},
    )

    with pytest.raises(ProvisioningError, match="pooled"):
        await directory.provisioning_target(ALPHA)


async def test_tenants_are_cached_independently() -> None:
    """A cache keyed on anything coarser than the tenant would hand one family another's placement —
    the failure this package exists to make impossible."""
    seen: list[str] = []

    async def by_tenant(tenant: Tenant) -> TenantRecord:
        seen.append(tenant.id)
        return (
            ISOLATED
            if tenant.id == ALPHA.id
            else TenantRecord(isolation=Isolation.pooled, instance="mem-2", database=None)
        )

    lookup = CachedLookup(by_tenant, clock=FakeClock())

    assert (await lookup(ALPHA)).database == "kip_alpha"
    assert (await lookup(BETA)).instance == "mem-2"
    assert (await lookup(ALPHA)).database == "kip_alpha"

    assert seen == [ALPHA.id, BETA.id], "each tenant resolved once, and never on the other's behalf"
