"""Text vs binary pgvector, through maagar's own sessions. The README's table comes from this.

    MAAGAR_BENCH_DSN=postgresql+asyncpg://user:pass@localhost:5432/postgres \\
        uv run --extra vectors python benchmarks/binary_vectors.py

The DSN is an admin connection to a Postgres with pgvector available; the script creates a scratch
database beside it and drops it at the end. Two stores share that database, one with the ordinary
``pgvector.sqlalchemy.Vector`` column and one with ``BinaryVector`` and ``register_binary_vectors``.
Both are warmed first, every case alternates which variant runs first, and each number is the
median of ``RUNS`` timings.
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import time
from typing import Any

from pgvector import Vector as PgVector
from pgvector.sqlalchemy import Vector
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from maagar import Maagar, SharedDatabase, Tenant
from maagar.vectors import BinaryVector, register_binary_vectors

ADMIN_DSN = os.environ.get(
    "MAAGAR_BENCH_DSN", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)
DATABASE = "maagar_bench_vectors"
DIM = 1024
RUNS = int(os.environ.get("MAAGAR_BENCH_RUNS", "30"))
TENANT = Tenant.attested("bench")
WRITES = [("Write 1 vector", 1), ("Write 20", 20), ("Write 150", 150), ("Write 500", 500)]
READS = (20, 150)


class TextBase(DeclarativeBase):
    pass


class BinaryBase(DeclarativeBase):
    pass


class TextRow(TextBase):
    __tablename__ = "bench_text"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str]
    batch: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(Vector(DIM))


class BinaryRow(BinaryBase):
    __tablename__ = "bench_binary"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str]
    batch: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(BinaryVector(DIM))


def vector() -> list[float]:
    return [random.random() for _ in range(DIM)]


async def main() -> None:
    admin = create_async_engine(ADMIN_DSN, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {DATABASE}"))
        await conn.execute(text(f"CREATE DATABASE {DATABASE}"))
    dsn = make_url(ADMIN_DSN).set(database=DATABASE).render_as_string(hide_password=False)
    directory = SharedDatabase(dsn=dsn, admin_dsn=dsn)
    stores: dict[str, tuple[Maagar, Any]] = {
        "text": (
            Maagar(metadata=TextBase.metadata, directory=directory, extensions=("vector",)),
            TextRow,
        ),
        "binary": (
            Maagar(
                metadata=BinaryBase.metadata,
                directory=directory,
                extensions=("vector",),
                on_connect=register_binary_vectors,
            ),
            BinaryRow,
        ),
    }
    try:
        for store, model in stores.values():
            await store.ensure_schema(TENANT)
            async with store.session(TENANT) as db:
                await db.execute(select(model).limit(1))
        timings, check = await measure(stores)
    finally:
        for store, _ in stores.values():
            await store.dispose()
        async with admin.connect() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS {DATABASE}"))
        await admin.dispose()
    report(timings, check)


async def measure(
    stores: dict[str, tuple[Maagar, Any]],
) -> tuple[dict[tuple[str, str], list[float]], dict[str, bool]]:
    random.seed(7)
    timings: dict[tuple[str, str], list[float]] = {}
    ids = iter(range(1, 10**9))
    batch = 0

    async def timed(case: str, name: str, work: Any) -> None:
        start = time.perf_counter()
        await work
        timings.setdefault((case, name), []).append(time.perf_counter() - start)

    async def write(name: str, vectors: list[list[float]], batch: int) -> None:
        store, model = stores[name]
        async with store.session(TENANT) as db:
            for v in vectors:
                db.add(model(id=next(ids), tenant_id=TENANT.id, batch=batch, embedding=v))

    async def read(name: str, batch: int) -> None:
        store, model = stores[name]
        async with store.session(TENANT) as db:
            rows = (await db.scalars(select(model).where(model.batch == batch))).all()
            assert rows and all(len(row.embedding) == DIM for row in rows)

    async def search(name: str, query: list[float]) -> None:
        store, model = stores[name]
        async with store.session(TENANT) as db:
            await db.execute(
                select(model.id).order_by(model.embedding.cosine_distance(query)).limit(10)
            )

    for run in range(RUNS):
        order = ("text", "binary") if run % 2 == 0 else ("binary", "text")
        for case, n in WRITES:
            vectors = [vector() for _ in range(n)]
            batch += 1
            for name in order:
                await timed(case, name, write(name, vectors, batch))
            if n in READS:
                for name in order:
                    await timed(f"Read {n}", name, read(name, batch))
        query = vector()
        for name in order:
            await timed("Search, top 10", name, search(name, query))

    # The same vector through both paths: Postgres must hold the same value, and Python must read
    # back the same float32s, whichever way it crossed the wire.
    sample = vector()
    for name in stores:
        await write(name, [sample], -1)
    store, _ = stores["text"]
    async with store.session(TENANT) as db:
        same_in_postgres = await db.scalar(
            text(
                "SELECT t.embedding = b.embedding FROM bench_text t, bench_binary b "
                "WHERE t.batch = -1 AND b.batch = -1"
            )
        )
    read_back: dict[str, list[float]] = {}
    for name, (store, model) in stores.items():
        async with store.session(TENANT) as db:
            value = await db.scalar(select(model.embedding).where(model.batch == -1))
            assert value is not None
            read_back[name] = value
    as_float32 = [PgVector(values).to_binary() for values in read_back.values()]
    return timings, {
        "same value in Postgres": bool(same_in_postgres),
        "same float32s read back": as_float32[0] == as_float32[1],
    }


def report(timings: dict[tuple[str, str], list[float]], check: dict[str, bool]) -> None:
    # The README's order, so a fresh run pastes over its table row for row.
    cases = [case for case, _ in WRITES] + [f"Read {n}" for n in READS] + ["Search, top 10"]
    print(f"| Median of {RUNS} | Text | Binary | Speedup |")
    print("|---|---:|---:|---:|")
    for case in cases:
        t = statistics.median(timings[(case, "text")]) * 1000
        b = statistics.median(timings[(case, "binary")]) * 1000
        print(f"| {case} | {t:.1f} ms | {b:.1f} ms | {t / b:.1f}× |")
    sample = vector()
    as_text = len(PgVector._to_db(sample).encode())  # type: ignore[attr-defined]
    as_binary = len(PgVector(sample).to_binary())
    print(
        f"| Bytes per {DIM}-d vector | {as_text:,} | {as_binary:,} | {as_text / as_binary:.1f}× |"
    )
    print()
    for name, ok in check.items():
        print(f"{name}: {'yes' if ok else 'NO'}")


if __name__ == "__main__":
    asyncio.run(main())
