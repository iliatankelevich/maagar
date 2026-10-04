"""``on_connect`` reaches every engine the store creates, and only through the hook it was given.

No database: the connect event is dispatched by hand, which is all SQLAlchemy does on a real
connection, and the store's paths are driven against a port nothing listens on, so each one gets as
far as creating its engine and then fails to connect.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from maagar import (
    DatabasePerTenant,
    EnginePool,
    Maagar,
    SharedDatabase,
    Tenant,
    attach_on_connect,
)

NOWHERE = "postgresql+asyncpg://u:p@127.0.0.1:1/nowhere"


async def hook(conn: Any) -> None:
    return None


class FakeDBAPIConnection:
    """What SQLAlchemy's connect event receives: an adapted connection with ``run_async``."""

    def __init__(self) -> None:
        self.ran: list[Any] = []

    def run_async(self, fn: Any) -> None:
        self.ran.append(fn)


# Fires the connect listeners maagar registered, and only those: the asyncpg dialect registers its
# own, which need a live server, and what is under test is maagar's wiring, not SQLAlchemy's. The
# engine's connect event is its pool's, which is where SQLAlchemy keeps the listeners.
def _connect(engine: AsyncEngine) -> list[Any]:
    fake = FakeDBAPIConnection()
    for listener in engine.sync_engine.pool.dispatch.connect:
        if getattr(listener, "__module__", None) == "maagar.engines":
            listener(fake, None)
    return fake.ran


async def test_an_attached_engine_runs_the_hook_on_each_connection() -> None:
    engine = create_async_engine(NOWHERE)
    attach_on_connect(engine, hook)
    assert _connect(engine) == [hook]
    assert _connect(engine) == [hook]
    await engine.dispose()


async def test_an_engine_nobody_attached_runs_nothing() -> None:
    engine = create_async_engine(NOWHERE)
    assert _connect(engine) == []
    await engine.dispose()


async def test_the_pool_attaches_the_hook_to_the_engines_it_makes() -> None:
    pool = EnginePool(on_connect=hook)
    async with pool.lease(NOWHERE) as engine:
        assert _connect(engine) == [hook]
    await pool.dispose_all()


async def test_a_callers_factory_does_not_opt_its_engines_out_of_the_hook() -> None:
    pool = EnginePool(factory=create_async_engine, on_connect=hook)
    async with pool.lease(NOWHERE) as engine:
        assert _connect(engine) == [hook]
    await pool.dispose_all()


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[AsyncEngine]:
    engines: list[AsyncEngine] = []

    def recording(dsn: str, **options: Any) -> AsyncEngine:
        engine = create_async_engine(dsn, **options)
        engines.append(engine)
        return engine

    monkeypatch.setattr("maagar.store.create_async_engine", recording)
    monkeypatch.setattr("maagar.engines.create_async_engine", recording)
    return engines


async def test_every_engine_the_store_creates_runs_the_hook(made: list[AsyncEngine]) -> None:
    tenant = Tenant.attested("alpha")
    pooled = Maagar(
        metadata=MetaData(),
        directory=SharedDatabase(dsn=NOWHERE, admin_dsn=NOWHERE),
        on_connect=hook,
    )
    isolated = Maagar(
        metadata=MetaData(),
        directory=DatabasePerTenant(instance_dsn=NOWHERE, instance_admin_dsn=NOWHERE),
        on_connect=hook,
    )

    with pytest.raises(OSError):
        async with pooled.session(tenant):
            pass
    with pytest.raises(OSError):
        async with pooled.admin(tenant):
            pass
    # Provisioning an isolated tenant opens the maintenance connection first, for CREATE DATABASE.
    with pytest.raises(OSError):
        await isolated.provision(tenant)

    assert len(made) == 3, "serving, admin and maintenance should each have made one engine"
    for engine in made:
        assert _connect(engine) == [hook]
    await pooled.dispose()
    await isolated.dispose()


async def test_a_store_without_a_hook_attaches_nothing(made: list[AsyncEngine]) -> None:
    store = Maagar(metadata=MetaData(), directory=SharedDatabase(dsn=NOWHERE, admin_dsn=NOWHERE))
    with pytest.raises(OSError):
        async with store.admin(Tenant.attested("alpha")):
            pass
    assert [_connect(engine) for engine in made] == [[]]
    await store.dispose()
