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

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from sqlalchemy.engine import make_url

from maagar.errors import UnknownTenant
from maagar.tenant import Tenant


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
    the spectrum; a mixed fleet is a directory that consults ``kip-mom`` and returns whichever of
    them applies to that tenant.
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
                        "SELECT datname FROM pg_database WHERE datname LIKE :pat ORDER BY datname"
                    ),
                    {"pat": f"{self._prefix}%"},
                )
                names = [row[0] for row in rows]
        finally:
            await engine.dispose()

        found: list[Tenant] = []
        for name in names:
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

    This is the shape L54 asks for. Two halves that never meet in one place:

    * a ``lookup`` coroutine — asks the control plane and gets a :class:`TenantRecord` back; and
    * an ``instances`` map — ``{"mem-1": (app_dsn, admin_dsn)}``, from this service's own config.

    Neither half is sufficient alone, which is the point: compromising the control plane yields
    routing information and no way to connect, and compromising a service's config yields
    credentials and no way to know which tenant lives where.

    ⚠️ **Cache the lookup.** Placement changes approximately never for a given tenant, and this sits
    in front of the data path's cold start — a service that cannot reach the control plane cannot
    open a database. Caching is what keeps a control-plane outage degrading to "no *new* tenants
    served" rather than "nobody served". The cache belongs in the ``lookup`` callable, not here,
    because its invalidation rule is a control-plane concern (see Q83).
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
