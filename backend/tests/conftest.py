"""Shared fixtures for tests that touch the database.

Tests run against a dedicated `semantic_search_test` database rather than the
dev database configured in `.env` -- otherwise every test run would drop and
re-seed the actual indexed corpus with toy fixture data. The env var override
below happens before `app.config` is ever imported, since pydantic-settings
reads `DATABASE_URL` from the environment with higher priority than `.env`.

These tests require a live Postgres (e.g. `docker compose up -d postgres`) and
are skipped automatically if one isn't reachable, so `pytest` stays usable in
environments without Docker.
"""
import os

import psycopg
import pytest

TEST_DB_NAME = "semantic_search_test"
_ADMIN_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
os.environ["DATABASE_URL"] = f"postgresql://postgres:postgres@localhost:5432/{TEST_DB_NAME}"


def _ensure_test_database_exists() -> bool:
    try:
        with psycopg.connect(_ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,)
            ).fetchone()
            if not exists:
                conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")

        # register_vector() (used by the connection pool below) queries the
        # `vector` type by name on connect, so the extension must exist
        # *before* the pool ever opens a connection -- schema.sql's own
        # CREATE EXTENSION runs too late to help with that first connection.
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=3) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return True
    except psycopg.OperationalError:
        return False


from app.db.connection import close_pool, healthcheck, run_migrations  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    if not _ensure_test_database_exists() or not healthcheck():
        pytest.skip("no database reachable; skipping DB-backed tests", allow_module_level=True)
    # A small, fixed dimension keeps these tests independent of which
    # embedding provider is configured.
    run_migrations(embedding_dim=8, recreate=True)
    yield
    close_pool()


@pytest.fixture(autouse=True)
def _clean_tables(_database):
    from app.db.connection import get_pool

    yield
    with get_pool().connection() as conn:
        conn.execute("TRUNCATE recipes CASCADE")
