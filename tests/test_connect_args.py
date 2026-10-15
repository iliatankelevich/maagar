"""``connect_args`` reach every engine the store opens, and directory discovery's too.

Behind a transaction-pooling PgBouncer asyncpg's statement cache must be off on every connection,
or the next transaction lands on a server connection that never prepared the statement. A store
that took the setting only in ``engine_options`` turned it off for its serving pool and left it on
for ``admin()``, provisioning and decommissioning: a consumer's suite through such a pooler failed
two runs in three, at ``ensure_schema``.

No database, as in ``test_on_connect.py``: each path is driven against a port nothing listens on,
so it gets as far as creating its engine, and the engine's arguments are what is checked.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from maagar import DatabasePerTenant, EnginePool, Maagar, SharedDatabase, Tenant
from test_on_connect import NOWHERE, _connect, hook

CACHE_OFF = {"statement_cache_size": 0}
#: The serving pool's own options, as a consumer sets them: none of them means anything to a
#: one-shot admin engine.
POOL_ONLY = {"pool_size": 3, "max_overflow": 2, "pool_timeout": 5.0, "pool_pre_ping": True}


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AsyncEngine, dict[str, Any]]]:
    engines: list[tuple[AsyncEngine, dict[str, Any]]] = []

    def recording(dsn: str, **options: Any) -> AsyncEngine:
        engine = create_async_engine(dsn, **options)
        engines.append((engine, options))
        return engine

    monkeypatch.setattr("maagar.store.create_async_engine", recording)
    monkeypatch.setattr("maagar.engines.create_async_engine", recording)
    # Discovery imports it when it runs, from SQLAlchemy itself.
    monkeypatch.setattr("sqlalchemy.ext.asyncio.create_async_engine", recording)
    return engines


async def _open_every_path(**store_options: Any) -> None:
    tenant = Tenant.attested("alpha")
    pooled = Maagar(
        metadata=MetaData(),
        directory=SharedDatabase(dsn=NOWHERE, admin_dsn=NOWHERE),
        **store_options,
    )
    isolated = Maagar(
        metadata=MetaData(),
        directory=DatabasePerTenant(instance_dsn=NOWHERE, instance_admin_dsn=NOWHERE),
        **store_options,
    )
    with pytest.raises(OSError):
        async with pooled.session(tenant):
            pass
    with pytest.raises(OSError):
        async with pooled.admin(tenant):
            pass
    with pytest.raises(OSError):
        await isolated.provision(tenant)
    with pytest.raises(OSError):
        await isolated.decommission(tenant)
    await pooled.dispose()
    await isolated.dispose()


async def test_connect_args_reach_every_engine_the_store_opens(
    made: list[tuple[AsyncEngine, dict[str, Any]]],
) -> None:
    await _open_every_path(engine_options=POOL_ONLY, connect_args=CACHE_OFF)

    serving, *others = (options for _, options in made)
    assert serving == {**POOL_ONLY, "connect_args": CACHE_OFF}
    # admin(), provisioning's maintenance connection and decommissioning's.
    assert len(others) == 3
    for options in others:
        assert options["connect_args"] == CACHE_OFF
        assert not set(options) & set(POOL_ONLY), "a serving-pool option reached an admin engine"


async def test_without_connect_args_every_engine_is_as_before(
    made: list[tuple[AsyncEngine, dict[str, Any]]],
) -> None:
    await _open_every_path()

    serving, *others = (options for _, options in made)
    assert serving == {"pool_pre_ping": True}
    assert others == [{}, {"isolation_level": "AUTOCOMMIT"}, {"isolation_level": "AUTOCOMMIT"}]


async def test_the_serving_pools_own_connect_args_are_kept_beside_the_stores(
    made: list[tuple[AsyncEngine, dict[str, Any]]],
) -> None:
    own = {"server_settings": {"application_name": "serving"}}
    store = Maagar(
        metadata=MetaData(),
        directory=SharedDatabase(dsn=NOWHERE, admin_dsn=NOWHERE),
        engine_options={"connect_args": own},
        connect_args=CACHE_OFF,
    )
    tenant = Tenant.attested("alpha")
    with pytest.raises(OSError):
        async with store.session(tenant):
            pass
    with pytest.raises(OSError):
        async with store.admin(tenant):
            pass
    (_, serving), (_, admin) = made
    assert serving["connect_args"] == {**own, **CACHE_OFF}
    assert admin["connect_args"] == CACHE_OFF
    await store.dispose()


def test_a_driver_argument_set_in_both_places_is_refused() -> None:
    with pytest.raises(ValueError, match="statement_cache_size"):
        Maagar(
            metadata=MetaData(),
            directory=SharedDatabase(dsn=NOWHERE, admin_dsn=NOWHERE),
            engine_options={"connect_args": {"statement_cache_size": 100}},
            connect_args=CACHE_OFF,
        )


def test_a_pool_with_a_factory_refuses_connect_args_it_could_not_apply() -> None:
    with pytest.raises(ValueError, match="factory"):
        EnginePool(factory=create_async_engine, connect_args=CACHE_OFF)


async def test_discovery_takes_the_connect_args_and_the_hook(
    made: list[tuple[AsyncEngine, dict[str, Any]]],
) -> None:
    directory = DatabasePerTenant(
        instance_dsn=NOWHERE, instance_admin_dsn=NOWHERE, connect_args=CACHE_OFF, on_connect=hook
    )
    with pytest.raises(OSError):
        await directory.roster()
    [(engine, options)] = made
    assert options == {"connect_args": CACHE_OFF}
    assert _connect(engine) == [hook]


async def test_discovery_without_them_is_as_before(
    made: list[tuple[AsyncEngine, dict[str, Any]]],
) -> None:
    with pytest.raises(OSError):
        await DatabasePerTenant(instance_dsn=NOWHERE, instance_admin_dsn=NOWHERE).roster()
    [(engine, options)] = made
    assert options == {}
    assert _connect(engine) == []
