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
management, provisioning, migration fan-out.

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

## Tests

```bash
make check     # ruff + pyright + pytest. No database needed.
```

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
