"""Shared fixtures for tests that touch the database.

These tests require a live Postgres (e.g. `docker compose up -d postgres`) and
are skipped automatically if one isn't reachable, so `pytest` stays usable in
environments without Docker.
"""
import pytest

from app.db.connection import close_pool, healthcheck, run_migrations


@pytest.fixture(scope="session", autouse=True)
def _database():
    if not healthcheck():
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
