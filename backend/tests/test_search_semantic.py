"""Tests for the semantic search module and its API endpoint.

Uses the real database and the real (locally cached) embedding model rather
than mocks -- the whole point of this project is the retrieval quality, and a
mocked embedding call would just test that Python can call a function.
"""
import pytest
from fastapi.testclient import TestClient

from app.db.repository import upsert_chunks, upsert_recipes
from app.ingestion.embed import get_embedding_provider
from app.main import app
from app.models.recipe import Chunk, Ingredient, Recipe
from app.search.semantic import semantic_search


@pytest.fixture(autouse=True)
def _seed_recipes(_database):
    """Seed a handful of recipes with real embeddings before each test.

    `_database` recreates the schema with a dummy 8-dim vector column (see
    conftest.py), which doesn't match the real embedding model's 384
    dimensions, so this fixture re-runs migrations at the real width before
    seeding -- otherwise every insert here would fail on a dimension mismatch.

    Function-scoped (not module-scoped) because conftest's `_clean_tables`
    truncates all tables after every test; re-seeding here keeps each test
    self-contained despite that.
    """
    from app.db.connection import run_migrations

    provider = get_embedding_provider()
    run_migrations(embedding_dim=provider.dimensions, recreate=True)

    recipes = [
        Recipe(
            id="1", title="Spicy Chicken Curry", category="Chicken", cuisine="Indian",
            tags=["Spicy", "Curry"],
            ingredients=[Ingredient(name="Chicken", measure="1 kg")],
            instructions="Simmer chicken in a spiced tomato and onion gravy for 30 minutes.",
        ),
        Recipe(
            id="2", title="Classic Beef Burger", category="Beef", cuisine="American",
            tags=["Grill"],
            ingredients=[Ingredient(name="Beef mince", measure="500 g")],
            instructions="Grill the beef patties and serve in a toasted bun with cheese.",
        ),
        Recipe(
            id="3", title="Vegetable Minestrone Soup", category="Soup", cuisine="Italian",
            tags=["Warm", "Vegetarian"],
            ingredients=[Ingredient(name="Tomatoes", measure="4")],
            instructions="A warm, hearty soup simmered slowly with vegetables and beans, perfect comfort food.",
        ),
    ]
    chunks = [Chunk(recipe_id=r.id, chunk_index=0, text=f"{r.title}\n{r.instructions}") for r in recipes]
    embeddings = provider.embed_documents([c.text for c in chunks])

    upsert_recipes(recipes)
    upsert_chunks(chunks, embeddings)


def test_semantic_search_ranks_relevant_recipe_first():
    results, took_ms = semantic_search("hot spicy Indian chicken dish", top_k=3)

    assert results[0].recipe_id == "1"
    assert took_ms >= 0


def test_semantic_search_matches_by_meaning_not_keywords():
    # No literal overlap with "Vegetable Minestrone Soup" or its instructions.
    results, _ = semantic_search("warm comfort food for a rainy day", top_k=3)

    assert results[0].recipe_id == "3"


def test_semantic_search_respects_top_k():
    results, _ = semantic_search("dinner", top_k=2)

    assert len(results) == 2


def test_semantic_search_returns_distinct_recipes():
    results, _ = semantic_search("food", top_k=10)

    ids = [r.recipe_id for r in results]
    assert len(ids) == len(set(ids))


def test_semantic_search_scores_are_descending():
    results, _ = semantic_search("chicken curry", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_endpoint_returns_expected_shape(client):
    response = client.get("/search/semantic", params={"q": "spicy chicken", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "semantic"
    assert body["query"] == "spicy chicken"
    assert len(body["results"]) == 2
    assert "score" in body["results"][0]


def test_endpoint_rejects_empty_query(client):
    assert client.get("/search/semantic", params={"q": ""}).status_code == 422


def test_endpoint_rejects_missing_query(client):
    assert client.get("/search/semantic").status_code == 422


def test_endpoint_rejects_top_k_out_of_range(client):
    assert client.get("/search/semantic", params={"q": "curry", "top_k": 500}).status_code == 422
    assert client.get("/search/semantic", params={"q": "curry", "top_k": 0}).status_code == 422
