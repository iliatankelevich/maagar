"""``maagar.vectors``: why the column type exists, and what the hook tolerates.

The round trip through a real Postgres is ``benchmarks/binary_vectors.py``'s, and the consumer's
suite; what is pinned here is everything that needs no database.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import pytest
from pgvector import Vector

from maagar.vectors import BinaryVector, register_binary_vectors


class FakeConnection:
    """asyncpg's catalog lookup and ``set_type_codec``. ``has_vector`` is whether the catalog finds
    the type; ``fail`` is raised by every codec registration."""

    def __init__(self, has_vector: bool = True, fail: Exception | None = None) -> None:
        self.codecs: dict[str, dict[str, Any]] = {}
        self.asked: list[tuple[str, ...]] = []
        self._has_vector = has_vector
        self._fail = fail

    async def fetchval(self, query: str, *args: str) -> bool:
        self.asked.append(args)
        return self._has_vector

    async def set_type_codec(self, name: str, **kwargs: Any) -> None:
        if self._fail is not None:
            raise self._fail
        self.codecs[name] = kwargs


def test_the_column_hands_the_list_to_the_driver_untouched() -> None:
    assert BinaryVector(3).bind_processor(cast(Any, None)) is None
    assert BinaryVector(3).get_col_spec() == "VECTOR(3)"


def test_the_column_still_reads_back_a_plain_list() -> None:
    process = BinaryVector(3).result_processor(cast(Any, None), None)
    assert process(Vector([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


async def test_the_binary_codec_cannot_take_what_the_ordinary_column_sends() -> None:
    # The reason BinaryVector exists. If a pgvector release ever makes the codec accept text, this
    # fails, and the column type may no longer be needed.
    conn = FakeConnection()
    await register_binary_vectors(cast(Any, conn))
    encode = conn.codecs["vector"]["encoder"]
    assert isinstance(encode([1.0, 2.0, 3.0]), bytes)
    with pytest.raises(ValueError, match="expected list or ndarray"):
        encode("[1,2,3]")


async def test_the_hook_registers_binary_codecs() -> None:
    conn = FakeConnection()
    await register_binary_vectors(cast(Any, conn))
    assert conn.codecs["vector"]["format"] == "binary"
    assert conn.codecs["vector"]["schema"] == "public"


async def test_the_schema_can_be_named() -> None:
    conn = FakeConnection()
    await partial(register_binary_vectors, schema="extensions")(cast(Any, conn))
    assert conn.asked == [("extensions",)]
    assert conn.codecs["vector"]["schema"] == "extensions"


async def test_a_database_without_the_extension_is_skipped() -> None:
    conn = FakeConnection(has_vector=False)
    await register_binary_vectors(cast(Any, conn))
    assert conn.codecs == {}


async def test_where_the_type_exists_every_failure_is_raised() -> None:
    conn = FakeConnection(fail=ValueError("unknown type: public.vector"))
    with pytest.raises(ValueError, match="unknown type"):
        await register_binary_vectors(cast(Any, conn))
