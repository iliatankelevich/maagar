"""Every error this package raises, so a caller can catch them without importing SQLAlchemy."""

from __future__ import annotations


class MaagarError(Exception):
    """Base class for everything raised here."""


class UnknownTenant(MaagarError):
    """No placement is known for this tenant.

    Deliberately an error rather than a fallback to some default database: a tenant that cannot be
    located must fail loudly, because the alternative — silently landing in a shared database — is
    exactly the cross-tenant write this package exists to make impossible.
    """


class InvalidTenantId(MaagarError):
    """The tenant id is not usable as a database identifier.

    Tenant ids reach ``CREATE DATABASE`` / ``DROP DATABASE``, where they cannot be bound as
    parameters. Validating the charset up front is what keeps that from being an injection sink;
    quoting alone is not enough, because a 64-byte id silently truncates instead of failing.
    """


class ProvisioningError(MaagarError):
    """Creating or removing a tenant's database failed."""
