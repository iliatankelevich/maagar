# maagar

> **מאגר** — a reservoir; *מאגר נתונים* is a database. Multi-tenant Postgres access with the
> **placement hidden behind the interface**.

Sibling to [`maslul`](https://github.com/iliatankelevich/maslul) (*מסלול*, a route), which does the
same trick for LLM providers: hide the choice behind a stable interface so no caller branches on it.

```python
from maagar import DatabasePerTenant, Maagar, Tenant

store = Maagar(
    metadata=Base.metadata,  # the entities
    directory=DatabasePerTenant(...),  # where tenants live
    app_role="kip_app",
    extensions=("vector",),
)

async with store.session(tenant) as db:
    ...
```

Everything below those two arguments is this package's problem: which database, which host, which
credential, whether it is shared, how many engines to keep, and how a schema change reaches a fleet
of them.

## Why it exists

`isolation = pooled | isolated` is a **plan attribute**, so both models have to coexist *at runtime*:
a tenant on the cheap plan shares a database while a tenant on the private plan has one of its own,
in the same process, at the same time. A seam that hides which one a tenant is on is load-bearing
today, not preparation for a hypothetical migration.

The second reason is that a second service needs the same machinery against a **different set of
entities**, and there is no version of "copy the db module across" that stays correct.

## What it hides, and what it deliberately does not

**Hidden:** placement, DSNs, credentials, engine lifecycle, pool caps and eviction, RLS policy
management, provisioning, migration fan-out, and what runs on each new connection.

**Not hidden:** SQLAlchemy. The session handed back is a real `AsyncSession` and the entities are
real models. Hiding the ORM would mean reimplementing query capability behind a smaller, worse
interface — and it would leak at the first `select()` anyway, because the entities passed in *are*
SQLAlchemy entities. Naming that line is more useful than blurring it.

## The parts that are easy to get wrong

**⚠️ RLS is bypassed by superusers, `BYPASSRLS` roles, and (without `FORCE`) table owners.** All are
configuration, not code — so every policy is bypassed and every test still passes, in exactly the
same green way as a correct system. Hence `Placement` carrying two DSNs, `FORCE` emitted
unconditionally, and `verify_posture()` refusing to start on a privileged credential.

**⚠️ A leased engine must never be evicted.** A textbook LRU disposes whatever is oldest, including
an engine in the middle of a transaction — `dispose()` closes the pool underneath it. Here only
*idle* engines are evicted; if the cache is over cap and everything is leased, the cap is exceeded
and counted (`stats().overflows`). A soft cap that is exceeded is bounded and observable; a hard cap
that severs a live write is not. `max_engines=None` disables eviction entirely, which is the right
setting when a process serves exactly one tenant.

**⚠️ A tenant id is attested, never claimed.** `Tenant(...)` raises; `Tenant.attested(...)` is the
only way in, so a raw `str` from a request body does not type-check and the trust boundary is one
greppable name. It also validates the charset and length, because tenant ids reach `CREATE DATABASE`
where they cannot be bound as parameters.

**⚠️ `set_config(..., true)`, not `SET LOCAL`.** Postgres does not accept bind parameters in a `SET`
statement, so `SET LOCAL app.tenant_id = $1` is a syntax error. The alternatives are interpolating
the tenant id into SQL — turning the one value that must never be attacker-influenced into an
injection sink — or hand-quoting it.

**⚠️ A shared data-access package makes credential separation a deployment property.** If several
services import this, any one of them *could* be handed another's DSN. `verify_posture()` includes a
startup check that the database contains only the entities this store declared, which catches the
accident; it cannot prove the rule, because two services could be pointed at one database
deliberately.

## Binary vectors with pgvector

pgvector's SQLAlchemy column sends every vector to Postgres as decimal text: about 19.7 KB for a
1,024-dimension embedding, formatted by Python and parsed again by the server. Postgres also accepts
vectors in binary, 4 bytes a dimension, but pgvector's two halves do not combine on their own.
`maagar.vectors` makes them combine:

```bash
pip install "maagar[vectors]"
```

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from maagar import Maagar, SharedDatabase, Tenant
from maagar.vectors import BinaryVector, register_binary_vectors


class Base(DeclarativeBase):
    pass


class Passage(Base):
    __tablename__ = "passages"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(BinaryVector(1024))


store = Maagar(
    metadata=Base.metadata,
    directory=SharedDatabase(dsn=..., admin_dsn=...),
    extensions=("vector",),
    on_connect=register_binary_vectors,
)

async with store.session(Tenant.attested("alpha")) as db:
    db.add(Passage(id=1, tenant_id="alpha", embedding=embedding))
```

Nothing else changes. Queries, `cosine_distance` and the rest work as before, and reading a row
back still gives a `list[float]`.

### What it buys

| Median of 30 | Text | Binary | Speedup |
|---|---:|---:|---:|
| Write 1 vector | 3.4 ms | 2.9 ms | 1.2× |
| Write 20 | 15.8 ms | 3.9 ms | 4.1× |
| Write 150 | 91.5 ms | 11.4 ms | 8.0× |
| Write 500 | 309.0 ms | 32.2 ms | 9.6× |
| Read 20 | 6.3 ms | 3.5 ms | 1.8× |
| Read 150 | 28.7 ms | 9.7 ms | 2.9× |
| Search, top 10 | 24.3 ms | 22.4 ms | 1.1× |
| Bytes per 1,024-d vector | 19,687 | 4,100 | 4.8× |

**How it was measured.** Every operation went through `store.session()`, on warm engines, with
1,024-dimension vectors. The two variants alternated which ran first, and each figure is the median
of 30 timings. Three runs agreed within about 10%.

**The environment.**
- Apple M2.
- PostgreSQL 16.14 with pgvector 0.8.2, in Docker on the same machine.
- Python 3.13, SQLAlchemy 2.0, asyncpg 0.31, pgvector-python 0.5.0.

To reproduce it:

```bash
MAAGAR_BENCH_DSN=postgresql+asyncpg://user:pass@localhost:5432/postgres \
    uv run --extra vectors python benchmarks/binary_vectors.py
```

**How to read it.**
- **The gain grows with the batch.** One vector barely moves. 20 vectors write about 4× faster,
  150 about 8×, and 500 about 9–10×. Text costs Python time to format each float and server time to
  parse it, and binary copies them.
- **Search does not change, beyond noise** (1.0–1.1× across runs). The distance is computed inside
  Postgres, and only the query vector crosses the wire.
- **There is no network in this measurement.** Database and client shared one machine, so the
  4.8× fewer bytes per vector saved CPU here, not bandwidth.
- **Values are unchanged.** The benchmark checks that Postgres holds the identical vector either
  way. Python reads back the same float32 values, in a more exact spelling: binary gives
  `0.13436424732208252` where text gave `0.13436425`.

### The rules

- **Both halves, always.** pgvector's codec under the ordinary `pgvector.sqlalchemy.Vector` column
  fails every write with `expected list or ndarray`, because that column has already turned the list
  into text. `BinaryVector` without the codec fails with `expected str, got list`. Both fail loudly,
  on the first write.
- **Every engine that writes these tables runs the hook.** The store's own engines do. For one you
  create yourself, such as an Alembic `env.py`, a migration's `upgrade(dsn)` or a script:

  ```python
  from sqlalchemy.ext.asyncio import create_async_engine

  from maagar import attach_on_connect
  from maagar.vectors import register_binary_vectors

  engine = create_async_engine(dsn)
  attach_on_connect(engine, register_binary_vectors)
  ```

- **An extension in another schema** is named with `functools.partial`:
  `on_connect=partial(register_binary_vectors, schema="extensions")`.
- **A database without the extension yet is skipped, not refused.** That covers the maintenance
  database provisioning connects to, and a tenant before its first migration. A connection opened
  there keeps no codec for its lifetime. So create the extension before a database is served, as
  `ensure_schema()` and `provision()` do, through `admin()` or your `schema_step`.
- **Switching an existing column is not a schema change.** The column is `VECTOR(n)` either way, and
  so is every row already in it.

## A hook on every connection: `on_connect`

`on_connect` is the general mechanism underneath. It is an async function, handed the raw asyncpg
connection, run once on every new connection the store opens:
- serving connections;
- `admin()`;
- the maintenance connection provisioning uses for `CREATE DATABASE`.

That last one is not a tenant's database, so a hook must not assume what it will find there.

```python
import asyncpg

from maagar.vectors import register_binary_vectors


async def on_connect(conn: asyncpg.Connection) -> None:
    await register_binary_vectors(conn)
    await conn.execute("SET application_name = 'billing-worker'")


store = Maagar(..., on_connect=on_connect)
```

It reaches every engine the store creates, and a pool built with a custom `factory` too.
`attach_on_connect(engine, hook)` is the same thing for engines the store does not create.

## Tests

```bash
make check     # ruff + pyright + pytest. No database needed.
```

`tests/test_on_connect.py` drives the store's serving, admin and maintenance paths against a port
nothing listens on, and checks that every engine they create runs the hook. The round trip through a
real Postgres is `benchmarks/binary_vectors.py`, which checks the values as well as timing them.

The interesting tests are about what the cache **refuses** to do —
`test_a_leased_engine_is_never_evicted` and
`test_the_cap_is_exceeded_rather_than_enforced_when_nothing_is_idle`.

**Integration coverage lives in the consumer, deliberately.** The consuming service's isolation
suite is parametrised over both placements: every cross-tenant assertion runs once pooled and once
with a real `CREATE DATABASE` per tenant. `test_the_two_placements_are_actually_different` is what
stops that parametrisation from being vacuous — without it, an isolated fixture that quietly
resolved both tenants to one database would leave every assertion passing and the isolated path
untested.

That split is not laziness. This package has no entities of its own, and an isolation test needs
real tables with real foreign keys to mean anything. Synthetic ones here would prove the test
fixture works.

**Sabotage-verified in the consumer.** With the policy gutted to `USING (true)`, the pooled run goes
five tests red and the isolated run goes two — only `WITH CHECK` and the fail-closed case, because
the other tenant's rows are in a different database entirely. That is the argument for one database
per tenant, measured rather than asserted.

## Status

Extracted from its first consumer on 2026-07-28, the day a second service needed the same
machinery. It was developed inside that first service as a workspace member, so it was generalised
*from* a real use rather than *for* an imagined one — the same order `maslul` earned its own repo in.

Not on PyPI, and not general yet: two consumers is enough to justify a repo, not enough to claim an
API. **Expect the interface to move.**

⚠️ **Consumers pin a commit, not a branch.** A floating `main` would silently change every service on
its next build, with no diff anywhere to notice it in.

## License

MIT — see [LICENSE](LICENSE).
