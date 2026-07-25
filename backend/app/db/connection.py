"""Postgres connectivity: a shared connection pool and schema migration.

Raw SQL via psycopg 3 rather than an ORM. Two reasons: the vector and
full-text queries here are the interesting part of the project and an ORM would
obscure them behind generated SQL, and pgvector's operators (`<=>`) plus
`ts_rank` have no clean ORM expression anyway.
"""
from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from app.config import settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    """Teach each pooled connection how to adapt Python lists to `vector`."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    """Return the process-wide connection pool, opening it on first use."""
    global _pool
    if _pool is None:
        logger.info("opening connection pool")
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure,
            open=True,
        )
    return _pool


def close_pool() -> None:
    """Close the pool. Called on FastAPI shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("connection pool closed")


def run_migrations(embedding_dim: int, recreate: bool = False) -> None:
    """Apply `schema.sql`, sizing the vector column for the active provider.

    Args:
        embedding_dim: Width of the `chunks.embedding` column.
        recreate: Drop existing tables first. Needed when switching embedding
            providers, since the vector column width would otherwise conflict.
    """
    ddl = SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "{embedding_dim}", str(embedding_dim)
    )

    with get_pool().connection() as conn:
        if recreate:
            logger.warning("dropping existing tables")
            conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
            conn.execute("DROP TABLE IF EXISTS recipes CASCADE")

        conn.execute(ddl)

        # Guard against a stale schema: if the table already existed with a
        # different width, every insert would fail with a confusing error.
        actual = conn.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """
        ).fetchone()
        if actual and actual[0] != embedding_dim:
            raise RuntimeError(
                f"chunks.embedding is vector({actual[0]}) but the configured "
                f"embedding model produces {embedding_dim} dimensions. "
                f"Re-run with --recreate to rebuild the schema."
            )

    logger.info("schema ready (embedding dimension %d)", embedding_dim)


def healthcheck() -> bool:
    """Return True if the database answers a trivial query."""
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.error("database healthcheck failed: %s", exc)
        return False
