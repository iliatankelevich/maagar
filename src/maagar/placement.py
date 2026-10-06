"""Where a tenant's data lives — and the seam that keeps callers from knowing.

This is the whole point of the package. A caller says *"a session for this tenant"*; it never
learns whether that tenant shares a database with a thousand others or has one to itself, which host
it is on, or which credential opened it. Moving a tenant between those worlds is a change to a
:class:`Directory`, not to a single line of business logic.

That matters today, not eventually: isolation is a **plan attribute**, so both models have to
coexist at runtime — a tenant on the cheap plan is pooled while a tenant on the private plan is
isolated, in the same process, at the same time.

Two credentials, deliberately
-----------------------------
Every :class:`Placement` carries **two** DSNs, because they must never be the same role:

``dsn``
    The unprivileged application role. Not a superuser, not the owner of any table. This is the only
    one used to serve a request.

``admin_dsn``
    The owner/DDL role, used for provisioning, schema changes and policy DDL. Never used to answer a
    query on behalf of a tenant.

Splitting them is not tidiness — Postgres exempts **superusers and ``BYPASSRLS`` roles** from row
level security entirely, and exempts a table's **owner** unless the table is ``FORCE``d. Connect as
either and every policy in the database becomes decorative while every test still passes. Keeping
the powerful credential in a separate field makes "which one is serving traffic" answerable by
reading the code instead of by reading the deployment.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from sqlalchemy.engine import make_url

from maagar.errors import ProvisioningError, UnknownTenant
from maagar.tenant import MAX_ID_BYTES, Tenant

_PG_IDENTIFIER_BYTES = 63


class Isolation(StrEnum):
    """How a tenant's rows are kept apart from every other tenant's."""

    #: Shares a database with other tenants. Separation is enforced by the ``tenant_id`` predicate
    #: on every query, backstopped by row level security.
    pooled = "pooled"

    #: Has a database of its own. Reaching another tenant's rows is not forbidden, it is *absent* —
    #: there is no shared table to mis-filter. Offboarding is ``DROP DATABASE``.
    isolated = "isolated"


@dataclass(frozen=True, slots=True)
class Placement:
    """Where one tenant lives. Internal to this package — never returned to a caller."""

    isolation: Isolation
    dsn: str
    admin_dsn: str


@runtime_checkable
class Directory(Protocol):
    """Resolves a tenant to a placement, and enumerates what exists.

    Implement this to plug in a real control plane. The two implementations below cover the ends of
    the spectrum; a mixed fleet is a directory that consults a control plane and returns whichever
    of them applies to that tenant.
    """

    async def locate(self, tenant: Tenant) -> Placement:
        """The placement for one tenant. Raises :class:`UnknownTenant` if there is none."""
        ...

    async def roster(self) -> Sequence[Tenant]:
        """Every tenant this directory knows about. Used for fleet-wide operations."""
        ...


def _swap_database(dsn: str, database: str) -> str:
    """Same server and credentials, different database.

    ``render_as_string(hide_password=False)`` rather than ``str(url)``: SQLAlchemy's ``__str__``
    masks the password as ``***``, which is the right default anywhere a URL might be logged and
    exactly wrong when the string is on its way to a driver. Using ``str()`` here fails
    authentication with a message naming the *role*, which sends you looking in the wrong place.
    """
    return make_url(dsn).set(database=database).render_as_string(hide_password=False)


class SharedDatabase:
    """Every tenant pooled into one database, separated by ``tenant_id`` + RLS.

    The cheapest shape and the one the isolation suite was originally written against. Fleet
    operations collapse to a single target, because there is only one database.
    """

    def __init__(self, *, dsn: str, admin_dsn: str, tenants: Sequence[Tenant] = ()) -> None:
        self._placement = Placement(isolation=Isolation.pooled, dsn=dsn, admin_dsn=admin_dsn)
        self._tenants = tuple(tenants)

    async def locate(self, tenant: Tenant) -> Placement:
        # No lookup: in a shared database every tenant resolves to the same place, and the row-level
        # predicate is what separates them. Refusing unknown tenants is the control plane's job,
        # not the directory's — there is nothing to *fail* to find.
        return self._placement

    async def roster(self) -> Sequence[Tenant]:
        return self._tenants


class DatabasePerTenant:
    """One database per tenant, packed onto a shared instance.

    Note the two halves of that sentence. A database each is what makes cross-tenant access absent
    rather than merely forbidden; packing them onto **one instance** is what keeps it affordable. A
    dedicated *instance* per tenant buys nothing extra — ``DROP DATABASE`` is equally clean either
    way — and costs ruinously past a couple of dozen tenants.

    ⚠️ The roster is discovered from ``pg_database`` by prefix. That is self-contained, which is why
    it is the default, but it is also *inference*: a database someone created by hand matching the
    prefix will be treated as a tenant, and a tenant whose database has not been provisioned yet is
    invisible. Pass an explicit ``roster`` (from the control plane) wherever the answer must be
    authoritative — fleet migrations being the obvious case.
    """

    def __init__(
        self,
        *,
        instance_dsn: str,
        instance_admin_dsn: str,
        prefix: str = "kip_",
        roster: Sequence[Tenant] | None = None,
    ) -> None:
        # Empty, every database on the instance is a tenant's: discovery adopts `postgres` and the
        # templates, and a tenant id `postgres` *is* the maintenance database, which decommission
        # drops. One unset environment variable away (kip-mind reads KIP_DB_PREFIX), so refused.
        if not prefix:
            raise ValueError("DatabasePerTenant needs a non-empty database prefix")
        # The id cap leaves room for the prefix only up to here: past 63 bytes Postgres truncates
        # identifiers silently, and two tenants whose ids share a stem would get one database.
        if len(prefix.encode()) + MAX_ID_BYTES > _PG_IDENTIFIER_BYTES:
            raise ValueError(
                f"database prefix {prefix!r} leaves no room for a {MAX_ID_BYTES}-byte tenant id "
                f"inside Postgres's {_PG_IDENTIFIER_BYTES}-byte identifier limit"
            )
        self._dsn = instance_dsn
        self._admin_dsn = instance_admin_dsn
        self._prefix = prefix
        self._roster = tuple(roster) if roster is not None else None

    def database_name(self, tenant: Tenant) -> str:
        """The database a tenant lives in. Public: provisioning and operators both need it."""
        return f"{self._prefix}{tenant.id}"

    @property
    def maintenance_dsn(self) -> str:
        """An admin connection to the *instance*, not to any tenant's database.

        ``CREATE DATABASE`` and ``DROP DATABASE`` cannot run from inside the database they target,
        so provisioning needs a connection that is deliberately pointed somewhere else.
        """
        return self._admin_dsn

    async def provisioning_target(self, tenant: Tenant) -> tuple[str, str]:
        """:class:`~maagar.store.SupportsProvisioning`. Local knowledge, so nothing is awaited —
        the coroutine exists for the directories that must go and ask."""
        return self._admin_dsn, self.database_name(tenant)

    async def locate(self, tenant: Tenant) -> Placement:
        database = self.database_name(tenant)
        return Placement(
            isolation=Isolation.isolated,
            dsn=_swap_database(self._dsn, database),
            admin_dsn=_swap_database(self._admin_dsn, database),
        )

    async def roster(self) -> Sequence[Tenant]:
        if self._roster is not None:
            return self._roster
        return await self._discover()

    async def _discover(self) -> Sequence[Tenant]:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(self._admin_dsn)
        try:
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text(
                        "SELECT datname FROM pg_database "
                        "WHERE starts_with(datname, :prefix) ORDER BY datname"
                    ),
                    {"prefix": self._prefix},
                )
                names = [row[0] for row in rows]
        finally:
            await engine.dispose()

        found: list[Tenant] = []
        for name in names:
            # starts_with() is already literal; this keeps a foreign database out even if the query
            # is ever loosened, because removeprefix() would hand back the whole foreign name.
            if not name.startswith(self._prefix):
                continue
            try:
                found.append(Tenant.attested(name.removeprefix(self._prefix)))
            except Exception:  # noqa: BLE001 - a database that merely shares the prefix
                continue
        return tuple(found)


class StaticDirectory:
    """An explicit tenant→placement map. The mixed fleet, spelled out.

    Useful in tests and as the shape a control-plane-backed directory ends up having: two tenants on
    the same instance, one pooled and one isolated, both served by the same process.
    """

    def __init__(self, placements: dict[Tenant, Placement]) -> None:
        self._placements = dict(placements)

    async def locate(self, tenant: Tenant) -> Placement:
        try:
            return self._placements[tenant]
        except KeyError:
            raise UnknownTenant(f"no placement registered for {tenant.id!r}") from None

    async def roster(self) -> Sequence[Tenant]:
        return tuple(self._placements)


@dataclass(frozen=True, slots=True)
class TenantRecord:
    """What a **control plane** knows about a tenant's storage — and deliberately no more.

    ⚠️ **There is no DSN here, and its absence is the design.** A control plane that handed out
    ``postgresql://user:password@host/kip_alpha`` would be holding a credential to every service's
    database, and "only the mind holds a memory-database credential" would be over — not overruled,
    just quietly meaningless.

    So the registry answers *"tenant alpha is isolated, database ``kip_alpha``, instance
    ``mem-1``"*, and each service combines that with **its own** configured credential for
    ``mem-1``. Placement is routing information. Credentials are not routing information.
    """

    isolation: Isolation
    #: A logical instance name — ``mem-1``, ``chat-2``. Never a host, never a port. Which physical
    #: server that resolves to is a deployment fact each service is configured with separately.
    instance: str
    #: The database within that instance. Meaningless under ``pooled``, where the shared database is
    #: whatever the instance's credential points at.
    database: str | None = None


class CatalogDirectory:
    """Placement from a control plane, composed with locally-held credentials.

    Two halves that never meet in one place:

    * a ``lookup`` coroutine — asks the control plane and gets a :class:`TenantRecord` back; and
    * an ``instances`` map — ``{"mem-1": (app_dsn, admin_dsn)}``, from this service's own config.

    Neither half is sufficient alone, which is the point: compromising the control plane yields
    routing information and no way to connect, and compromising a service's config yields
    credentials and no way to know which tenant lives where.

    ⚠️ **Cache the lookup.** Placement changes approximately never for a given tenant, and this sits
    in front of the data path's cold start — a service that cannot reach the control plane cannot
    open a database. Caching is what keeps a control-plane outage degrading to "no *new* tenants
    served" rather than "nobody served". The cache belongs in the ``lookup`` callable, not here,
    because its invalidation rule is a control-plane concern — see :class:`CachedLookup`, which is
    that wrapper and not a change to this class.
    """

    def __init__(
        self,
        *,
        lookup: Callable[[Tenant], Awaitable[TenantRecord]],
        instances: Mapping[str, tuple[str, str]],
        roster: Callable[[], Awaitable[Sequence[Tenant]]] | None = None,
    ) -> None:
        self._lookup = lookup
        self._instances = dict(instances)
        self._roster = roster

    async def locate(self, tenant: Tenant) -> Placement:
        record = await self._lookup(tenant)
        try:
            app_dsn, admin_dsn = self._instances[record.instance]
        except KeyError:
            raise UnknownTenant(
                f"tenant {tenant.id!r} is placed on instance {record.instance!r}, which this "
                "service has no credential for. Either the control plane knows about an instance "
                "this deployment was not configured with, or the config is stale."
            ) from None

        if record.isolation is Isolation.pooled or not record.database:
            return Placement(isolation=Isolation.pooled, dsn=app_dsn, admin_dsn=admin_dsn)
        return Placement(
            isolation=Isolation.isolated,
            dsn=_swap_database(app_dsn, record.database),
            admin_dsn=_swap_database(admin_dsn, record.database),
        )

    async def roster(self) -> Sequence[Tenant]:
        if self._roster is None:
            raise UnknownTenant(
                "this directory has no roster source; fleet operations need one from the control "
                "plane. Discovering tenants from pg_database would be inference, and a fleet "
                "migration is the last place to guess."
            )
        return await self._roster()

    async def provisioning_target(self, tenant: Tenant) -> tuple[str, str]:
        """:class:`~maagar.store.SupportsProvisioning` — and the reason that protocol is async.

        Which instance a tenant is on, and what its database is called, are the control plane's
        answers, not this object's. The **instance-level** admin DSN is returned deliberately
        un-swapped: :meth:`locate` points ``admin_dsn`` *at* the tenant's database, which is right
        for DDL inside it and useless for creating it, since ``CREATE DATABASE`` cannot run from
        inside its own target.
        """
        record = await self._lookup(tenant)
        try:
            _, instance_admin_dsn = self._instances[record.instance]
        except KeyError:
            raise UnknownTenant(
                f"tenant {tenant.id!r} is placed on instance {record.instance!r}, which this "
                "service has no credential for."
            ) from None

        if record.isolation is not Isolation.isolated or not record.database:
            raise ProvisioningError(
                f"tenant {tenant.id!r} is pooled: there is no database of its own to create or "
                "drop. Provisioning a pooled tenant is schema convergence, and removing one is a "
                "schema-aware purge that belongs beside the models."
            )
        return instance_admin_dsn, record.database


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    record: TenantRecord
    fetched_at: float
    last_attempt_at: float


class CachedLookup:
    """Wraps a :class:`CatalogDirectory` lookup with a TTL and a **serve-stale-on-error** rule.

    A control plane in front of the data path's cold start has to be cached, but the obvious
    implementation gets the failure mode backwards. A plain TTL cache that expires an entry and
    then propagates the control plane's error takes **every** tenant down the moment the TTL rolls
    past an outage — it delays the outage rather than containing it.

    The rule that actually contains it:

    ===================  ==========================================================
    cached and fresh     return it; the control plane is not called
    stale, retry due     refresh; **if the lookup fails, return the stale record**
    stale, backing off   return the stale record; the control plane is not called
    not cached           call; if the lookup fails, **raise**
    ===================  ==========================================================

    which is exactly *"an outage degrades to no **new** tenants served, never to nobody served"*.
    Placement changes approximately never, so a stale record is very nearly always the right answer,
    and the one case where staleness is unsafe — a tenant migrated between instances — is a
    deliberate operation that can :meth:`invalidate`.

    ⚠️ **A failed refresh backs off instead of retrying on every call.** Without it, once an entry is
    past its TTL, *every* call pays for a control-plane attempt — fine for a quick failure, costly
    once the lookup carries a deadline (a caller waiting out a 10-second timeout on a call that was
    always going to be served from the cache anyway). ``fetched_at`` stays the time of the last
    *successful* lookup — it is the record's true age, and the only thing the TTL check reads —
    while a failure instead advances a separate ``retry_interval_seconds`` backoff, so a down
    control plane is attempted at most once per interval per tenant and a recovered one is noticed
    within that interval rather than after a full TTL. The backoff is measured from when the
    attempt *failed*, not from when it started: timing it from the start would let a lookup that
    hangs past the retry interval (a deadline above the default 30s, or a shorter interval
    configured to match a tighter one) license the very next call to re-attempt immediately,
    which is the cap this exists to put on a down control plane, undone. ``served_stale`` counts
    every call answered from memory past its TTL, attempted or backed off alike: it means *"this
    tenant is running on a cached record because the control plane wasn't asked, or was asked and
    failed"*.

    ⚠️ **A failed refresh writes its backoff back only if nothing else changed the entry while it
    was in flight.** Two races matter: :meth:`invalidate` racing a refresh that is already underway
    — the failure must not resurrect the pre-migration record :meth:`invalidate` just removed — and
    two concurrent refreshes, where a slower failure must not overwrite a faster success. Either
    way the caller that lost the race still returns its own snapshot of the stale record; only the
    shared cache entry is left alone.

    ⚠️ **A miss is never cached.** A tenant the control plane has registered but not yet placed must
    not be remembered as unknown for the length of the TTL: with reconciliation-based provisioning
    that is a normal, self-resolving state, and caching it would turn a signup seconds away from
    working into one that fails for an hour.

    ⚠️ **Deliberately not stampede-protected.** Concurrent misses for one tenant each call through.
    The alternative is a per-tenant lock in front of the data path — a second synchronisation
    primitive bought with no measurement. Revisit on evidence.
    """

    #: One hour. Long on purpose: every second of TTL is a second of control-plane outage the caller
    #: never notices.
    DEFAULT_TTL_SECONDS = 3600.0

    #: 30 seconds. Short on purpose: it bounds how long a recovered control plane goes unnoticed,
    #: while still keeping a down one from being attempted on every single call.
    DEFAULT_RETRY_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        lookup: Callable[[Tenant], Awaitable[TenantRecord]],
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        retry_interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lookup = lookup
        self._ttl = ttl_seconds
        # Clamped: a retry interval longer than the TTL would never fire, since the TTL check above
        # it already gates every call.
        self._retry_interval = min(retry_interval_seconds, ttl_seconds)
        self._clock = clock
        self._entries: dict[str, _CacheEntry] = {}
        #: Count of calls answered from memory past their TTL — attempted-and-failed or backed-off
        #: alike — because that is what "this process is running on a cached placement" means to an
        #: alert. Counted rather than logged so it can be a metric; a log line here is one nobody
        #: reads until afterwards.
        self.served_stale = 0

    async def __call__(self, tenant: Tenant) -> TenantRecord:
        now = self._clock()
        cached = self._entries.get(tenant.id)
        if cached is not None:
            if now - cached.fetched_at < self._ttl:
                return cached.record
            if now - cached.last_attempt_at < self._retry_interval:
                self.served_stale += 1
                return cached.record

        try:
            record = await self._lookup(tenant)
        except Exception:
            if cached is None:
                # Never resolved. Failing is correct: the alternative is inventing a placement, and
                # a guessed placement is a cross-tenant write.
                raise
            # Only record the attempt if this tenant's entry is still the one read above. While
            # this failing lookup was in flight, invalidate() may have removed it (a migration —
            # the one case staleness is wrong, and resurrecting the pre-migration record here would
            # defeat it) or a concurrent refresh may have replaced it with a fresh success (which a
            # slower failure must not clobber). Either way, this caller still got a true read of
            # the cache at the time it asked, so it still returns that snapshot.
            if self._entries.get(tenant.id) is cached:
                # Read after the await, not the `now` from before it: the backoff must run from
                # when the attempt failed, not from when it started. A lookup that hangs longer
                # than the retry interval would otherwise leave the next call free to re-attempt
                # immediately, defeating the cap this class exists to put on a down control plane.
                self._entries[tenant.id] = replace(cached, last_attempt_at=self._clock())
            self.served_stale += 1
            return cached.record

        fetched_at = self._clock()
        self._entries[tenant.id] = _CacheEntry(
            record=record, fetched_at=fetched_at, last_attempt_at=fetched_at
        )
        return record

    def invalidate(self, tenant: Tenant) -> None:
        """Forget one tenant — after a deliberate migration between instances, the one case where a
        stale record is wrong rather than merely old."""
        self._entries.pop(tenant.id, None)
