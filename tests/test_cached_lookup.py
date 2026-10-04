"""The cache in front of the control plane — and the failure rule that is the whole point.

These are the assertions that distinguish this from a TTL dict. A cache that expires and then
propagates the control plane's error is *worse* than no cache during an outage: it works for exactly
one TTL and then takes every tenant down at once, which is a far harder failure to diagnose than one
that never worked.
"""

from __future__ import annotations

import asyncio

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


class GatedControl:
    """A control plane whose calls each block on their own future until the test resolves them —
    so two calls for the same tenant can be put in flight together and settled independently, in
    either order and with either outcome, to exercise the races :meth:`CachedLookup` must survive.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.entered: list[asyncio.Event] = []
        self._futures: list[asyncio.Future[TenantRecord]] = []

    async def __call__(self, tenant: Tenant) -> TenantRecord:
        future: asyncio.Future[TenantRecord] = asyncio.get_running_loop().create_future()
        entered = asyncio.Event()
        self.entered.append(entered)
        self._futures.append(future)
        self.calls += 1
        entered.set()
        return await future

    def succeed(self, index: int, record: TenantRecord) -> None:
        self._futures[index].set_result(record)

    def fail(self, index: int, error: Exception) -> None:
        self._futures[index].set_exception(error)


# `ensure_future` only schedules a task; one yield lets it run to its first real await, which is
# always the lookup's own `await future`: there is nothing else to await before it.
async def _wait_until_entered(gated: GatedControl, index: int) -> None:
    await asyncio.sleep(0)
    await gated.entered[index].wait()


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


async def test_within_the_retry_interval_a_stale_record_is_served_without_a_call() -> None:
    """KIP-120: past the TTL, a down control plane must not be re-attempted on every call — only
    once per retry interval, so a deadline on the lookup is not paid by every caller."""
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, retry_interval_seconds=30, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 101  # past the ttl: the first call here pays for the failed attempt
    await lookup(ALPHA)
    calls_after_first_failure = control.calls

    clock.now = 110  # within the retry interval of that failed attempt
    assert await lookup(ALPHA) == POOLED
    assert control.calls == calls_after_first_failure, "backing off: no attempt before the interval"


async def test_past_the_retry_interval_exactly_one_call_re_attempts() -> None:
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, retry_interval_seconds=30, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 101
    await lookup(ALPHA)  # first failed attempt; starts the backoff
    calls_before = control.calls

    clock.now = 101 + 30  # the retry interval has elapsed
    await lookup(ALPHA)
    assert control.calls == calls_before + 1, "one interval elapsed, one attempt"


async def test_recovery_after_a_backoff_stores_the_fresh_record_and_resets_it() -> None:
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, retry_interval_seconds=30, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 101
    await lookup(ALPHA)  # stale, backoff started

    control.up = True
    control.record = ISOLATED
    clock.now = 101 + 30
    assert await lookup(ALPHA) == ISOLATED, "the control plane recovered; the fresh record wins"

    clock.now = 101 + 30 + 1  # inside what would have been the old backoff window
    assert await lookup(ALPHA) == ISOLATED, "freshly cached again, so this call is not backing off"
    assert control.calls == 3, "fetched, failed once, recovered once — the third call is cached"


async def test_invalidate_clears_the_backoff_along_with_the_record() -> None:
    """``invalidate`` forgets the whole entry, backoff included — a stale record that was wrong
    about to be stale should not also leave behind a stale opinion of when to retry."""
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, retry_interval_seconds=30, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 101
    await lookup(ALPHA)  # stale, backoff started
    calls_before_invalidate = control.calls

    lookup.invalidate(ALPHA)
    clock.now = 102  # well within what was the backoff window
    with pytest.raises(UnknownTenant):
        await lookup(ALPHA)
    attempted = control.calls == calls_before_invalidate + 1
    assert attempted, "invalidate forgot the backoff, not just the record"


async def test_invalidate_during_an_in_flight_failing_refresh_is_not_resurrected() -> None:
    """The migration race: :meth:`invalidate` fires while a stale entry's refresh is already in
    flight, and that refresh then fails. The failure's write-back must lose to the invalidate —
    otherwise the pre-migration record it removed comes right back from the failed attempt, which
    is exactly the one case the class's docstring says staleness is wrong rather than merely old.
    """
    clock = FakeClock()
    gated = GatedControl()
    lookup = CachedLookup(gated, ttl_seconds=100, retry_interval_seconds=30, clock=clock)

    seed = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 0)
    gated.succeed(0, POOLED)
    assert await seed == POOLED

    clock.now = 101  # past the ttl
    in_flight = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 1)  # the refresh is awaiting; cache entry still the seed

    lookup.invalidate(ALPHA)  # races the in-flight refresh

    gated.fail(1, UnknownTenant("control plane unreachable"))
    resolved = await in_flight
    assert resolved == POOLED, "this caller still gets the snapshot it read before invalidate"

    next_call = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 2)
    gated.fail(2, UnknownTenant("control plane unreachable"))
    with pytest.raises(UnknownTenant):
        await next_call
    assert gated.calls == 3, "the next call must re-attempt, not be served a resurrected record"


async def test_a_concurrent_success_is_not_overwritten_by_a_slower_failure() -> None:
    """Two refreshes for the same tenant race: one succeeds and stores a fresh record, the other
    fails afterwards. The slower failure must not clobber the fresher, correct success."""
    clock = FakeClock()
    gated = GatedControl()
    lookup = CachedLookup(gated, ttl_seconds=100, retry_interval_seconds=30, clock=clock)

    seed = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 0)
    gated.succeed(0, POOLED)
    assert await seed == POOLED

    clock.now = 101  # past the ttl
    slow_failure = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 1)  # in flight, holding the pre-race stale snapshot

    fast_success = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 2)
    gated.succeed(2, ISOLATED)
    assert await fast_success == ISOLATED

    gated.fail(1, UnknownTenant("control plane unreachable"))
    assert await slow_failure == POOLED, "the losing caller still gets its own stale snapshot"

    # The fresh, successful record must still be what the cache answers with.
    clock.now = 102  # still inside what would have been the failure's backoff window
    assert await lookup(ALPHA) == ISOLATED
    assert gated.calls == 3, "fetched, raced, recovered — the last call must be served from cache"


async def test_the_backoff_runs_from_when_the_attempt_failed_not_when_it_started() -> None:
    """A lookup that hangs past the retry interval before failing must not hand the very next call
    a license to re-attempt immediately — the backoff has to be measured from the failure, not
    from the call that produced it, or a slow failure defeats the cap this class exists to put on
    a down control plane."""
    clock = FakeClock()
    gated = GatedControl()
    lookup = CachedLookup(gated, ttl_seconds=100, retry_interval_seconds=30, clock=clock)

    seed = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 0)
    gated.succeed(0, POOLED)
    assert await seed == POOLED

    clock.now = 101  # past the ttl: this call starts the failing refresh
    in_flight = asyncio.ensure_future(lookup(ALPHA))
    await _wait_until_entered(gated, 1)

    clock.now = 101 + 40  # the lookup hangs longer than the retry interval before it fails
    gated.fail(1, UnknownTenant("control plane unreachable"))
    assert await in_flight == POOLED

    calls_before = gated.calls
    next_call = asyncio.ensure_future(lookup(ALPHA))
    await asyncio.sleep(0)  # let it either finish from cache or reach its own re-attempt
    if not next_call.done():
        gated.succeed(2, POOLED)  # unblock rather than hang the test if the backoff was wrong
    assert await next_call == POOLED
    assert gated.calls == calls_before, "no re-attempt: the backoff runs from when the call failed"


async def test_served_stale_counts_every_call_answered_from_memory_past_the_ttl() -> None:
    """``served_stale`` means *"answered from memory because the TTL had passed"*, whether or not
    this particular call paid for a control-plane attempt. Before the backoff every such call was
    an attempt, so the count's rate during an outage is what it always was."""
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=100, retry_interval_seconds=30, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 101
    await lookup(ALPHA)  # attempted, failed
    assert lookup.served_stale == 1

    clock.now = 110  # backing off: no attempt, still a stale serve
    await lookup(ALPHA)
    assert lookup.served_stale == 2


async def test_a_retry_interval_longer_than_the_ttl_is_clamped() -> None:
    """A retry interval past the TTL would never fire — the TTL check above it already gates every
    call — so it is clamped rather than silently ignored."""
    control, clock = Control(), FakeClock()
    lookup = CachedLookup(control, ttl_seconds=50, retry_interval_seconds=100, clock=clock)
    await lookup(ALPHA)

    control.up = False
    clock.now = 51  # past the ttl, and less than the uncapped retry interval would allow
    assert await lookup(ALPHA) == POOLED
    assert control.calls == 2, "clamped to the ttl, so the retry is due as soon as it goes stale"


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
