"""The public surface: ask for a tenant's session, get one. Everything else is hidden.

``Maagar`` is initialised with two things and nothing else:

* **the entities** — a SQLAlchemy ``MetaData``, i.e. the caller's own models; and
* **the placement** — a :class:`~maagar.placement.Directory` that says where tenants live.

From there a caller only ever writes::

    async with store.session(tenant) as db:
        ...

and never learns which database that was, which host it is on, which credential opened it, or
whether the tenant shares it with a thousand others. Changing any of those is a change to the
directory. That is the seam, and it is the reason this is a package rather than a module.

What deliberately stays visible
-------------------------------
The session handed back is a real ``AsyncSession``, and the models are real SQLAlchemy models. This
package hides **placement and connection topology**; it does not hide the ORM, and pretending
otherwise would mean reimplementing query capability behind a smaller, worse interface — which would
leak at the first ``select()`` anyway, because the entities passed in *are* SQLAlchemy entities.
Naming the line is more useful than blurring it.

The rule this cannot enforce, and why it must be written down
--------------------------------------------------------------
⚠️ A shared data-access package makes a *credential-separation* rule harder to keep, not easier. If
several services import this, then any one of them **could** be configured with any other's DSN —
and the separation that used to be a property of the code becomes a property of the deployment. That
is the same shape as the RLS-superuser trap in :mod:`maagar.rls`: nothing fails, nothing goes red,
and the guarantee is quietly gone. Where a service must be the only holder of a credential, that has
to be asserted where the configuration is assembled. This package cannot see far enough to do it.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import MetaData, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from maagar.engines import EnginePool, OnConnect, PoolStats, attach_on_connect
from maagar.errors import ProvisioningError
from maagar.placement import Directory, Isolation, Placement
from maagar.rls import TENANT_SETTING, apply_policies, assert_unprivileged, tenant_scoped_tables
from maagar.tenant import Tenant

#: A migration step. Sync callables (alembic's ``command.upgrade``) are run in a worker thread, so a
#: slow one cannot stall the event loop while the rest of the fleet waits.
Upgrade = Callable[[str], Awaitable[None]] | Callable[[str], None]


@runtime_checkable
class SupportsProvisioning(Protocol):
    """A directory that can create and drop databases — i.e. one that isolates tenants.

    ⚠️ **Per-tenant and asynchronous, and both of those are load-bearing.** An earlier shape exposed
    a single ``maintenance_dsn`` property and a synchronous ``database_name``, which works only when
    the directory already knows every answer locally. A directory backed by a control plane does
    not: which instance a tenant is on, and what its database is called, are facts it has to go and
    ask for. A synchronous property cannot ask, so that shape silently excluded the one directory
    that matters in production from ever provisioning anything.

    Both values are returned together because they come from one lookup and are only meaningful as a
    pair — a database name is worthless without the instance to create it on.
    """

    async def provisioning_target(self, tenant: Tenant) -> tuple[str, str]:
        """``(maintenance_dsn, database_name)`` for this tenant.

        The DSN is admin access to the **instance**, deliberately pointed at some *other* database:
        ``CREATE DATABASE`` and ``DROP DATABASE`` cannot run from inside the database they target.
        """
        ...


async def _invoke(upgrade: Upgrade, dsn: str) -> None:
    """Call a migration step, whichever flavour it is.

    Alembic's ``command.upgrade`` is synchronous and does real I/O, so calling it inline would stall
    the event loop and serialise a fan-out that is supposed to be concurrent. A thread per target,
    bounded by the caller's semaphore, is what makes ``concurrency`` mean anything.
    """
    if inspect.iscoroutinefunction(upgrade):
        await upgrade(dsn)
        return
    result = await asyncio.to_thread(upgrade, dsn)
    if inspect.isawaitable(result):
        await result


def _redact(dsn: str) -> str:
    """A DSN safe to put in a report or a log line.

    ``str(URL)`` masks the password as ``***``. That is the right behaviour here and the wrong
    behaviour when handing a string to a driver — the two uses look identical and are not.
    """
    return str(make_url(dsn))


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """What happened to one database during a fleet operation."""

    target: str
    ok: bool
    seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FleetReport:
    """The result of a fleet-wide operation.

    Carries wall-clock per target because migration fan-out time across a per-tenant fleet is the
    number that decides whether the model scales, and it has never been measured. A report that only
    said "ok" would leave the question open forever.
    """

    outcomes: tuple[TargetOutcome, ...]
    seconds: float

    @property
    def failed(self) -> tuple[TargetOutcome, ...]:
        """⚠️ A non-empty value here is a **partially migrated fleet** — some tenants on the new
        schema, some on the old, with one build of the application serving both. It is an
        operational state, not merely a failed command, and it needs resolving before deploy."""
        return tuple(o for o in self.outcomes if not o.ok)

    @property
    def slowest(self) -> TargetOutcome | None:
        return max(self.outcomes, key=lambda o: o.seconds, default=None)

    def __bool__(self) -> bool:
        return not self.failed


class Maagar:
    """Multi-tenant data access with the placement hidden behind it.

    ``on_connect`` runs on every connection the store opens: serving, :meth:`admin`, and the
    maintenance connection provisioning uses for ``CREATE DATABASE``. That last one is not a
    tenant's database, so a hook must tolerate a database without what it looks for, as
    ``maagar.vectors.register_binary_vectors`` does. Engines a caller creates itself get it through
    :func:`maagar.attach_on_connect`.
    """

    def __init__(
        self,
        *,
        metadata: MetaData,
        directory: Directory,
        max_engines: int | None = 64,
        engine_options: Mapping[str, Any] | None = None,
        tenant_column: str = "tenant_id",
        tenant_setting: str = TENANT_SETTING,
        app_role: str | None = None,
        extensions: Sequence[str] = (),
        schema_step: Upgrade | None = None,
        on_connect: OnConnect | None = None,
    ) -> None:
        self._metadata = metadata
        self._directory = directory
        self._on_connect = on_connect
        self._pool = EnginePool(
            max_engines=max_engines, engine_options=engine_options, on_connect=on_connect
        )
        self._column = tenant_column
        self._setting = tenant_setting
        self._app_role = app_role
        self._extensions = tuple(extensions)
        self._schema_step = schema_step

    # Every engine the store makes outside the serving pool comes through here, so `on_connect`
    # cannot be forgotten on one of them.
    def _engine(self, dsn: str, **options: Any) -> AsyncEngine:
        engine = create_async_engine(dsn, **options)
        if self._on_connect is not None:
            attach_on_connect(engine, self._on_connect)
        return engine

    # ------------------------------------------------------------------ serving

    @asynccontextmanager
    async def session(self, tenant: Tenant) -> AsyncIterator[AsyncSession]:
        """A session inside a transaction pinned to one tenant.

        The tenant announcement uses ``set_config(..., is_local => true)``, which scopes the setting
        to the surrounding **transaction**. That third argument is the whole thing: without it the
        setting outlives the transaction and rides along to the next checkout of that pooled
        connection, which is how connection pooling turns into a cross-tenant read.

        ⚠️ ``set_config()`` rather than ``SET LOCAL``, for a specific reason: **Postgres does not
        accept bind parameters in a ``SET`` statement** — ``SET LOCAL app.tenant_id = $1`` is a
        syntax error. The alternatives are interpolating the tenant id into the SQL string, which
        turns the one value that must never be attacker-influenced into an injection sink, or
        hand-quoting it. ``set_config`` is an ordinary function call, so the value binds normally.
        """
        placement = await self._directory.locate(tenant)
        async with self._pool.lease(placement.dsn) as engine:
            maker = async_sessionmaker(engine, expire_on_commit=False)
            async with maker() as session, session.begin():
                await session.execute(
                    text("SELECT set_config(:setting, :tenant, true)"),
                    {"setting": self._setting, "tenant": tenant.id},
                )
                yield session

    async def verify_posture(self, tenant: Tenant, *, exclusive_schema: bool = True) -> None:
        """Assert this store is correctly wired. Call once at startup, before serving anything.

        Two checks, both cheap, both converting a misconfiguration from "silently green forever"
        into "refuses to start":

        **The credential does not bypass RLS.** A superuser or ``BYPASSRLS`` role turns every policy
        in the database into decoration while every test stays green.

        **The database contains only the entities this store declares.** This is the startup
        detector for the credential-separation hazard in this module's docstring. If a service is
        handed another service's DSN — the failure that a shared data-access package makes
        possible — it opens a database full of tables it never declared, and says so. It is not a
        proof of separation, since two services *could* be pointed at one database deliberately;
        it catches the accident, which is the case that actually happens. Pass
        ``exclusive_schema=False`` where sharing is intended.
        """
        placement = await self._directory.locate(tenant)
        async with self._pool.lease(placement.dsn) as engine, engine.connect() as conn:
            await assert_unprivileged(conn)
            if exclusive_schema:
                await self._assert_exclusive_schema(conn)

    async def _assert_exclusive_schema(self, conn: AsyncConnection) -> None:
        rows = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
        )
        # alembic_version is the migration tool's bookkeeping, not an entity, and it is present in
        # every real database.
        found = {row[0] for row in rows} - {"alembic_version"}
        foreign = found - set(self._metadata.tables)
        if foreign:
            raise PermissionError(
                f"this database contains {sorted(foreign)}, which this store never declared. "
                "That usually means the service was configured with another service's DSN — the "
                "failure mode a shared data-access package introduces, where credential separation "
                "stops being a property of the code and becomes a property of the deployment."
            )

    # ------------------------------------------------------------------- schema

    @asynccontextmanager
    async def admin(self, tenant: Tenant) -> AsyncIterator[AsyncConnection]:
        """An owner-level connection to a tenant's database. **DDL and seeding only.**

        Deliberately a separate method with a blunt name rather than a flag on :meth:`session`. This
        connection bypasses row level security by owning the tables, so every use of it is a place
        where the backstop is off — and that should be greppable in one search.
        """
        placement = await self._directory.locate(tenant)
        engine = self._engine(placement.admin_dsn)
        try:
            async with engine.begin() as conn:
                yield conn
        finally:
            await engine.dispose()

    async def ensure_schema(self, tenant: Tenant) -> None:
        """Bring a tenant's database to the current schema, idempotently.

        ⚠️ **Pass ``schema_step`` in production.** Without it this falls back to ``create_all`` plus
        policy DDL, which produces the right *shape* and leaves the database **unstamped** — no
        ``alembic_version`` row. The next fleet migration then tries to run the baseline against a
        database that already has every table, and fails on the first ``CREATE TABLE``. A tenant
        provisioned that way is a tenant that can never be migrated again, and nothing notices until
        the first schema change.

        The fallback stays because it is genuinely right for tests and for a service that has no
        migration tool yet — but it is a fallback, not a default worth relying on.
        """
        if self._schema_step is not None:
            placement = await self._directory.locate(tenant)
            await _invoke(self._schema_step, placement.admin_dsn)
        else:
            async with self.admin(tenant) as conn:
                for extension in self._extensions:
                    await conn.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{extension}"'))
                await conn.run_sync(self._metadata.create_all)
                await apply_policies(
                    conn,
                    tenant_scoped_tables(self._metadata, self._column),
                    column=self._column,
                    setting=self._setting,
                )

        # Grants run either way and always last: they must cover whatever the schema step created,
        # and a role granted before the tables exist is granted nothing.
        if self._app_role:
            async with self.admin(tenant) as conn:
                await self._grant(conn, self._app_role)

    async def _grant(self, conn: AsyncConnection, role: str) -> None:
        """Give the unprivileged role the rights it needs, and nothing more.

        Roles are cluster-wide in Postgres but **grants are per database**, so every newly created
        tenant database starts with the application role able to connect and able to touch nothing.
        Forgetting this makes provisioning look successful and the first real request fail.
        """
        quoted = f'"{role}"'
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {quoted}"))
        await conn.execute(
            text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {quoted}")
        )
        await conn.execute(
            text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {quoted}")
        )
        # Without this, tables created by a *later* migration are invisible to the app role and the
        # failure surfaces long after the deploy that caused it.
        await conn.execute(
            text(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {quoted}"
            )
        )

    # ------------------------------------------------------------- provisioning

    async def provision(self, tenant: Tenant) -> None:
        """Make a tenant's storage exist and be ready to serve.

        Pooled: there is no database to create, so this is schema convergence only. Isolated:
        ``CREATE DATABASE`` first, then the same schema work.
        """
        placement = await self._directory.locate(tenant)
        if placement.isolation is Isolation.isolated:
            await self._create_database(tenant)
        await self.ensure_schema(tenant)

    async def decommission(self, tenant: Tenant) -> None:
        """Remove a tenant's data.

        Isolated: ``DROP DATABASE``, which is why the model was chosen — offboarding is one
        statement with no way to miss a table. Pooled: deliberately **not implemented**, because a
        correct pooled purge must delete from every tenant-scoped table in dependency order, and a
        wrong one leaves rows behind that no test would notice. That belongs in the service that
        owns the schema, next to the test that walks the metadata and proves nothing survived.
        """
        placement = await self._directory.locate(tenant)
        if placement.isolation is not Isolation.isolated:
            raise NotImplementedError(
                "decommissioning a pooled tenant is a schema-aware purge, not a generic one — "
                "implement it beside the models, with a metadata walk asserting no rows remain."
            )
        await self._drop_database(tenant)

    def _provisioner(self) -> SupportsProvisioning:
        directory = self._directory
        if not isinstance(directory, SupportsProvisioning):
            raise ProvisioningError(
                f"{type(directory).__name__} cannot create or drop databases; it must expose "
                "provisioning_target to isolate tenants."
            )
        return directory

    @asynccontextmanager
    async def _maintenance(self, dsn: str) -> AsyncIterator[AsyncConnection]:
        """A connection to the instance, outside any tenant database, in AUTOCOMMIT.

        ``CREATE DATABASE`` and ``DROP DATABASE`` cannot run inside a transaction block, and
        SQLAlchemy opens one by default — so this is not a stylistic choice.
        """
        engine = self._engine(dsn, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                yield conn
        finally:
            await engine.dispose()

    async def _create_database(self, tenant: Tenant) -> None:
        maintenance_dsn, name = await self._provisioner().provisioning_target(tenant)
        async with self._maintenance(maintenance_dsn) as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            )
            if exists:
                return
            # The identifier cannot be bound. Tenant.attested() already refused anything outside
            # [a-z0-9_-] and anything over 40 bytes, which is what makes this quoting sufficient
            # rather than hopeful — and the length check is what stops two tenants colliding on one
            # silently truncated 63-byte name.
            await conn.execute(text(f'CREATE DATABASE "{name}"'))

    async def _drop_database(self, tenant: Tenant) -> None:
        maintenance_dsn, name = await self._provisioner().provisioning_target(tenant)
        await self._pool.dispose_all()
        async with self._maintenance(maintenance_dsn) as conn:
            # WITH (FORCE) terminates other sessions; without it the drop fails whenever anything is
            # still connected, which during offboarding is normal rather than exceptional.
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))

    # ------------------------------------------------------------ fleet actions

    async def targets(self) -> tuple[tuple[str, str], ...]:
        """The distinct databases behind the roster, as ``(label, admin_dsn)``.

        Pooled fleets collapse to one entry; isolated fleets produce one per tenant. Callers that
        operate on *databases* rather than on tenants should go through this, so the same code is
        correct in both models.
        """
        seen: dict[str, str] = {}
        for tenant in await self._directory.roster():
            placement: Placement = await self._directory.locate(tenant)
            seen.setdefault(placement.admin_dsn, _redact(placement.admin_dsn))
        return tuple((label, dsn) for dsn, label in seen.items())

    async def migrate_fleet(self, upgrade: Upgrade, *, concurrency: int = 4) -> FleetReport:
        """Run ``upgrade(dsn)`` against every database in the fleet.

        The migration tool is injected rather than imported, so this package stays free of alembic
        and a caller can migrate however it already does.

        ⚠️ **One failure does not abort the rest.** Stopping at the first error leaves a fleet split
        at an arbitrary point with no record of where; running everything and reporting leaves it
        split at a *known* point. Neither is good, but only one is diagnosable.
        """
        targets = await self.targets()
        gate = asyncio.Semaphore(concurrency)

        async def run(label: str, dsn: str) -> TargetOutcome:
            async with gate:
                started = time.monotonic()
                try:
                    await _invoke(upgrade, dsn)
                    return TargetOutcome(label, ok=True, seconds=time.monotonic() - started)
                except Exception as exc:  # noqa: BLE001 - recorded per target, not swallowed
                    return TargetOutcome(
                        label, ok=False, seconds=time.monotonic() - started, error=repr(exc)
                    )

        started = time.monotonic()
        outcomes = await asyncio.gather(*(run(label, dsn) for label, dsn in targets))
        return FleetReport(tuple(outcomes), seconds=time.monotonic() - started)

    # ----------------------------------------------------------------- lifecycle

    def stats(self) -> PoolStats:
        return self._pool.stats()

    async def dispose(self) -> None:
        await self._pool.dispose_all()
