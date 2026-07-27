"""The attestation guard and the id rules.

Both exist for the same reason: a tenant id is the one value in the system that must never be
influenced by a request, and it is also the value that reaches ``CREATE DATABASE``.
"""

from __future__ import annotations

import pytest

from maagar import InvalidTenantId, Tenant


def test_the_bare_constructor_is_refused() -> None:
    """P3 made mechanical. ``Tenant("whatever")`` is what a handler reaching into a request body
    would write, so it is the call that has to fail."""
    with pytest.raises(TypeError, match="attested"):
        Tenant("fam-alpha")


def test_attested_is_the_way_in() -> None:
    assert Tenant.attested("fam-alpha").id == "fam-alpha"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Fam-Alpha",  # uppercase: Postgres would fold it and the name would stop round-tripping
        "fam alpha",
        "fam';DROP DATABASE x;--",  # why validation happens before quoting, not instead of it
        "-leading-hyphen",
        "a" * 41,  # over the cap that keeps `kip_<id>` inside Postgres's 63-byte identifier limit
    ],
)
def test_ids_that_cannot_safely_become_database_names_are_refused(bad: str) -> None:
    with pytest.raises(InvalidTenantId):
        Tenant.attested(bad)


def test_a_forty_character_id_is_allowed() -> None:
    """The boundary itself, so a future tweak to the cap cannot silently move it."""
    assert Tenant.attested("a" * 40).id == "a" * 40


def test_tenants_are_immutable_and_hashable() -> None:
    tenant = Tenant.attested("fam-alpha")
    with pytest.raises(AttributeError):
        tenant.id = "fam-beta"  # type: ignore[misc]
    assert {tenant, Tenant.attested("fam-alpha")} == {tenant}
    assert tenant != Tenant.attested("fam-beta")


def test_a_tenant_is_not_equal_to_its_own_id() -> None:
    """Otherwise ``tenant == "fam-alpha"`` would quietly work, and every place that should have been
    forced to say ``tenant.id`` would keep passing a ``Tenant`` into a string column."""
    assert Tenant.attested("fam-alpha") != "fam-alpha"
