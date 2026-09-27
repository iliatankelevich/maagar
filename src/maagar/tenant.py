"""The tenant identity type.

The type exists to make one rule mechanical instead of cultural: **a tenant identity is *attested*,
never claimed.** It is resolved once, at the trust boundary, from a verified credential — and it is
never read from a request body, a query string, or a model's output.

A bare ``str`` cannot express that difference, so every function here takes a :class:`Tenant`, and
the only way to make one is :meth:`Tenant.attested`. That stops no determined caller — nothing
in Python could — but it turns the rule into two things it was not before:

  * a **type error** when a raw ``str`` is passed where a tenant is expected, and
  * a **single greppable name**. ``grep -rn 'Tenant.attested' src/`` should return the handful of
    lines where a credential is verified, and nothing else. In a service that is one call site, at
    the edge. A second one in a request handler is a review finding you can actually see.
"""

from __future__ import annotations

import re
from typing import Final, final

from maagar.errors import InvalidTenantId

_ATTESTATION: Final = object()

#: Tenant ids become database identifiers (``kip_<id>``), so the charset is the intersection of what
#: is safe unquoted-ish and what stays readable in ``\l``. The 40-byte cap leaves room for the
#: prefix inside Postgres's 63-byte identifier limit — over which names *truncate silently* and two
#: different tenants can collide on one database.
#: ⚠️ `\Z`, not `$`: Python's `re` lets `$` match immediately before a trailing newline, so an id
#: with one appended would validate as if it were not there — becoming a second spelling of the
#: same identifier for any caller that compares the two strings directly.
_VALID_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}\Z")


@final
class Tenant:
    """An attested tenant identity."""

    __slots__ = ("id",)

    id: str

    def __init__(self, id: str, _token: object = None) -> None:  # noqa: A002
        if _token is not _ATTESTATION:
            raise TypeError(
                "Tenant(...) is not a public constructor. A tenant identity must be attested — "
                "derived from a verified credential at the trust boundary — never claimed by a "
                "request. Use Tenant.attested(...) there, and pass the Tenant inward."
            )
        object.__setattr__(self, "id", id)

    @classmethod
    def attested(cls, tenant_id: str) -> Tenant:
        """Mint a tenant identity from a **verified** credential.

        Call this where the credential is checked, and nowhere else.
        """
        if not _VALID_ID.match(tenant_id):
            raise InvalidTenantId(
                f"{tenant_id!r} is not a valid tenant id: expected 1-40 chars matching "
                r"^[a-z0-9][a-z0-9_-]*\Z (it becomes a database identifier)."
            )
        return cls(tenant_id, _ATTESTATION)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Tenant is immutable")

    def __repr__(self) -> str:
        return f"Tenant({self.id!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tenant) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("maagar.Tenant", self.id))
