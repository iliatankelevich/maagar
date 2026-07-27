"""maagar — multi-tenant Postgres access with the placement hidden behind the interface.

    from maagar import DatabasePerTenant, Maagar, Tenant

    store = Maagar(
        metadata=Base.metadata,                    # the entities
        directory=DatabasePerTenant(...),          # where tenants live
        app_role="kip_app",
        extensions=("vector",),
    )

    async with store.session(Tenant.attested(claim)) as db:
        ...

Everything below the first two arguments is this package's problem: which database, which host,
which credential, whether it is shared, how many engines to keep, and how a schema change reaches a
fleet of them. Callers write business logic against a session and stay unable to express a
cross-tenant query even by accident.

The name is Hebrew — *מאגר*, a reservoir; *מאגר נתונים* is a database. It is a sibling to `maslul`
(*מסלול*, a route), which does the same trick for LLM providers: hide the choice behind a stable
interface so no caller branches on it.
"""

from maagar.engines import EnginePool, PoolStats
from maagar.errors import (
    InvalidTenantId,
    MaagarError,
    ProvisioningError,
    UnknownTenant,
)
from maagar.placement import (
    DatabasePerTenant,
    Directory,
    Isolation,
    Placement,
    SharedDatabase,
    StaticDirectory,
)
from maagar.rls import (
    TENANT_SETTING,
    apply_policies,
    assert_unprivileged,
    policy_statements,
    tenant_scoped_tables,
)
from maagar.store import FleetReport, Maagar, SupportsProvisioning, TargetOutcome
from maagar.tenant import Tenant

__all__ = [
    "TENANT_SETTING",
    "DatabasePerTenant",
    "Directory",
    "EnginePool",
    "FleetReport",
    "InvalidTenantId",
    "Isolation",
    "Maagar",
    "MaagarError",
    "Placement",
    "PoolStats",
    "ProvisioningError",
    "SharedDatabase",
    "StaticDirectory",
    "SupportsProvisioning",
    "TargetOutcome",
    "Tenant",
    "UnknownTenant",
    "apply_policies",
    "assert_unprivileged",
    "policy_statements",
    "tenant_scoped_tables",
]

__version__ = "0.1.0"
