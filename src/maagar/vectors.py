"""Binary pgvector: vectors cross the wire as 4 bytes a dimension instead of as decimal text.

Optional, ``pip install "maagar[vectors]"``. Two halves, and both are required::

    from maagar import Maagar
    from maagar.vectors import BinaryVector, register_binary_vectors

    class Passage(Base):
        embedding: Mapped[list[float]] = mapped_column(BinaryVector(1024))

    store = Maagar(..., on_connect=register_binary_vectors)

⚠️ **pgvector's own two pieces do not compose.** ``pgvector.sqlalchemy.Vector`` renders every
value to text before the driver sees it, and ``pgvector.asyncpg.register_vector`` makes the driver
expect a list, so registering the codec under the ordinary column type fails every write with
``expected list or ndarray``. :class:`BinaryVector` hands the list through untouched. The reverse,
:class:`BinaryVector` on a connection without the codec, fails as loudly (``expected str, got
list``), which is why every engine that writes must run the hook: the store's own engines do, and
:func:`maagar.attach_on_connect` covers the ones a caller creates.

What it buys is measured in the README, under *Binary vectors*.
"""

from __future__ import annotations

from typing import Any

from asyncpg import Connection
from pgvector.asyncpg import register_vector
from pgvector.sqlalchemy import Vector

__all__ = ["BinaryVector", "register_binary_vectors"]


class BinaryVector(Vector):
    """A pgvector column whose values reach asyncpg as lists, for the binary codec to encode.

    Reading back is unchanged: a ``list[float]``, from the codec's exact float32 values.
    """

    cache_ok = True

    def bind_processor(self, dialect: Any) -> None:
        return None


async def register_binary_vectors(conn: Connection, /, *, schema: str = "public") -> None:
    """The ``on_connect`` hook for :class:`BinaryVector`: pgvector's binary codecs on ``conn``.

    ``schema`` is where the ``vector`` extension is installed; pass it with ``functools.partial``.

    A database where the type does not exist yet is skipped, not refused: the maintenance database
    provisioning connects to, and a tenant's database before its first migration. That is asked of
    the catalog first, so every error ``register_vector`` raises is a real one. ⚠️ A connection
    opened there keeps no codec for its lifetime, so a :class:`BinaryVector` write on it fails.
    Create the extension before a database is served, as :meth:`maagar.Maagar.ensure_schema` does,
    through :meth:`~maagar.Maagar.admin` or the caller's ``schema_step``.
    """
    if await conn.fetchval(_VECTOR_TYPE_EXISTS, schema):
        await register_vector(conn, schema=schema)


_VECTOR_TYPE_EXISTS = (
    "SELECT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace"
    " WHERE t.typname = 'vector' AND n.nspname = $1)"
)
