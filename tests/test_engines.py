"""The eviction rule, which is the only interesting thing about the cache.

A plain LRU is trivially correct and quietly catastrophic here: disposing an engine closes its pool,
and closing a pool underneath a live transaction severs it mid-write. So these tests are about
what the cache **refuses** to do.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, cast

import pytest

from maagar import EnginePool


class FakeEngine:
    """Stands in for an ``AsyncEngine``: the pool only ever calls ``dispose``."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


def _pool(max_engines: int | None) -> tuple[EnginePool, dict[str, FakeEngine]]:
    made: dict[str, FakeEngine] = {}

    def factory(dsn: str) -> Any:
        made[dsn] = FakeEngine(dsn)
        return made[dsn]

    return EnginePool(max_engines=max_engines, factory=cast(Any, factory)), made


async def test_an_engine_is_reused_for_the_same_dsn() -> None:
    pool, made = _pool(4)
    async with pool.lease("a") as first, pool.lease("a") as second:
        assert first is second
    assert len(made) == 1


async def test_the_least_recently_used_idle_engine_is_evicted() -> None:
    pool, made = _pool(2)
    for dsn in ("a", "b", "c"):
        async with pool.lease(dsn):
            pass
    assert made["a"].disposed, "'a' was the oldest idle engine"
    assert not made["b"].disposed
    assert not made["c"].disposed
    assert pool.stats().size == 2


async def test_a_leased_engine_is_never_evicted() -> None:
    """⚠️ The one that matters.

    'a' is the least recently used by a wide margin, and it is also in the middle of a transaction.
    A textbook LRU disposes it and the caller holding it gets a severed connection. Here eviction
    walks past it and takes the oldest *idle* engine instead.
    """
    pool, made = _pool(2)
    async with pool.lease("a"):
        for dsn in ("b", "c", "d"):
            async with pool.lease(dsn):
                pass
        assert not made["a"].disposed
    assert made["b"].disposed


async def test_the_cap_is_exceeded_rather_than_enforced_when_nothing_is_idle() -> None:
    """A soft cap is the right failure. Exceeding it is bounded and reported; honouring it by
    disposing a busy engine is not."""
    pool, _ = _pool(1)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(pool.lease("a"))
        await stack.enter_async_context(pool.lease("a"))  # a second lease on the same engine
        await stack.enter_async_context(pool.lease("b"))
        # Releasing one of the two 'a' leases runs eviction while 'a' is still leased and 'b' is
        # still leased — nothing is evictable.
        async with pool.lease("c"):
            pass

    assert pool.stats().overflows > 0, "the overflow must be counted, not silently absorbed"


async def test_an_unbounded_pool_never_evicts() -> None:
    """The instance-per-tenant setting: one engine per process, so an LRU over it is pure overhead
    and the eviction it might do is pure risk."""
    pool, made = _pool(None)
    for dsn in ("a", "b", "c", "d", "e"):
        async with pool.lease(dsn):
            pass
    assert not any(e.disposed for e in made.values())
    assert pool.stats().size == 5
    assert pool.stats().capacity is None


async def test_dispose_all_clears_the_cache() -> None:
    pool, made = _pool(4)
    async with pool.lease("a"):
        pass
    await pool.dispose_all()
    assert made["a"].disposed
    assert pool.stats().size == 0


def test_a_zero_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="max_engines"):
        EnginePool(max_engines=0)
