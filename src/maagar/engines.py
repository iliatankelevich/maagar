"""A bounded cache of engines, and the eviction rule that makes it safe.

One database per tenant means one connection pool per tenant, which is the cost that model actually
imposes: a thousand tenants is a thousand pools, each holding its minimum idle connections against a
server with a fixed ``max_connections``. Caching engines forever eventually saturates the server;
not caching them at all means a TCP handshake and an authentication round trip on every request.

So: a cache with a cap and a least-recently-used eviction order.

⚠️ **The eviction rule is the part that is easy to get wrong.** A plain LRU disposes whatever is
oldest — including an engine that is, right now, mid-transaction. ``dispose()`` closes
the pool underneath it and the caller gets a severed connection in the middle of a write. So:

    Only **idle** engines are ever evicted. If the cache is over its cap and every engine is leased,
    the cap is exceeded rather than enforced.

Exceeding a soft cap is a bounded, observable problem — :meth:`EnginePool.stats` reports it. Killing
a live transaction to honour a number is not. The number is a budget, not an invariant.

**Setting ``max_engines=None`` disables eviction entirely.** That is right when a process
serves exactly one tenant: there is one engine, it is always in use, and an LRU over a set of size
one is pure overhead. The cache is a property of the shared-process model, and it disappears on its
own when the process stops being shared.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from asyncpg import Connection
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

#: Runs once on every new database connection, handed the raw asyncpg connection: the place to
#: register a type codec or set a session parameter. ``maagar.vectors.register_binary_vectors`` is
#: one.
OnConnect = Callable[[Connection], Awaitable[object]]


def attach_on_connect(engine: AsyncEngine, hook: OnConnect) -> None:
    """Run ``hook`` on every connection ``engine`` opens from now on.

    The store does this for every engine it creates. This is for the ones it does not: an Alembic
    ``env.py``, a migration's ``upgrade(dsn)``, a script. ⚠️ An engine that writes the same tables
    must run the same hook — a type codec registered on some connections and not others fails
    only on the ones without it, which is the hardest version of the bug to find.
    """

    # SQLAlchemy's connect event is synchronous; `run_async` awaits the hook on the driver's own
    # connection inside the greenlet the async engine is already running in.
    @event.listens_for(engine.sync_engine, "connect")
    def _run(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.run_async(hook)


@dataclass(slots=True)
class _Entry:
    engine: AsyncEngine
    leases: int = 0
    #: A monotonic counter, not a clock. Wall-clock timestamps can tie or go backwards; a counter
    #: gives a total order over "most recently handed out" for free.
    touched: int = 0


@dataclass(frozen=True, slots=True)
class PoolStats:
    """What the cache is doing. Worth exporting as metrics before the fleet is large."""

    #: Engines currently held.
    size: int
    #: Engines with at least one live lease.
    leased: int
    #: The configured cap, or ``None`` for unbounded.
    capacity: int | None
    #: How many times eviction ran with the cache over cap and found nothing idle to evict. A
    #: non-zero and rising value means the cap is not being honoured and the real ceiling is the
    #: server's ``max_connections``. This is the number to alert on, not ``size``.
    overflows: int


class EnginePool:
    """Engines keyed by DSN, capped, with idle-only LRU eviction."""

    def __init__(
        self,
        *,
        max_engines: int | None = 64,
        engine_options: Mapping[str, Any] | None = None,
        factory: Callable[[str], AsyncEngine] | None = None,
        on_connect: OnConnect | None = None,
    ) -> None:
        if max_engines is not None and max_engines < 1:
            raise ValueError("max_engines must be >= 1, or None for unbounded")
        self._max = max_engines
        self._options = dict(engine_options or {"pool_pre_ping": True})
        self._factory = factory or self._default_factory
        self._on_connect = on_connect
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        self._clock = 0
        self._overflows = 0

    def _default_factory(self, dsn: str) -> AsyncEngine:
        return create_async_engine(dsn, **self._options)

    @asynccontextmanager
    async def lease(self, dsn: str) -> AsyncIterator[AsyncEngine]:
        """Borrow the engine for ``dsn``, guaranteed not to be disposed while held."""
        async with self._lock:
            entry = self._entries.get(dsn)
            if entry is None:
                engine = self._factory(dsn)
                # On every engine, a caller's factory included: a factory is a choice of engine,
                # not an opt-out of what every connection needs.
                if self._on_connect is not None:
                    attach_on_connect(engine, self._on_connect)
                entry = _Entry(engine=engine)
                self._entries[dsn] = entry
            entry.leases += 1
            self._clock += 1
            entry.touched = self._clock
            engine = entry.engine
        try:
            yield engine
        finally:
            async with self._lock:
                entry.leases -= 1
            # Eviction runs on *release*, not on acquire: releasing is the only moment an engine can
            # become evictable, and doing it here keeps the acquire path free of disposal latency.
            await self._evict_over_cap()

    async def _evict_over_cap(self) -> None:
        if self._max is None:
            return
        while True:
            async with self._lock:
                if len(self._entries) <= self._max:
                    return
                idle = [(e.touched, dsn) for dsn, e in self._entries.items() if e.leases == 0]
                if not idle:
                    self._overflows += 1
                    return
                _, victim = min(idle)
                entry = self._entries.pop(victim)
            # Disposal is I/O and can block; doing it outside the lock keeps every other tenant's
            # lease path moving while this pool drains.
            await entry.engine.dispose()

    def stats(self) -> PoolStats:
        return PoolStats(
            size=len(self._entries),
            leased=sum(1 for e in self._entries.values() if e.leases),
            capacity=self._max,
            overflows=self._overflows,
        )

    async def dispose_all(self) -> None:
        """Dispose every engine. Call on shutdown; leases must already be released."""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            await entry.engine.dispose()
