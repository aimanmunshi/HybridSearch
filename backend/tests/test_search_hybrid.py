"""Tests for hybrid (semantic + keyword) ranking."""
import pytest
from fastapi.testclient import TestClient

from app.db.repository import upsert_chunks, upsert_recipes
from app.ingestion.embed import get_embedding_provider
from app.main import app
from app.models.recipe import Chunk, Ingredient, Recipe
from app.search.hybrid import _normalize, hybrid_search


@pytest.fixture(autouse=True)
def _seed_recipes(_database):
    """Real embeddings + real recipes, matching test_search_semantic.py's pattern."""
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
            id="3", title="Harissa Roasted Vegetables", category="Side", cuisine="Moroccan",
            tags=["Vegetarian"],
            ingredients=[Ingredient(name="Harissa paste", measure="2 tbs")],
            instructions="Toss root vegetables in harissa paste and olive oil, then roast until tender.",
        ),
        Recipe(
            id="4", title="Warm Vegetable Minestrone Soup", category="Soup", cuisine="Italian",
            tags=["Warm", "Vegetarian"],
            ingredients=[Ingredient(name="Tomatoes", measure="4")],
            instructions="A warm, hearty soup simmered slowly with vegetables and beans, perfect comfort food.",
        ),
    ]
    chunks = [Chunk(recipe_id=r.id, chunk_index=0, text=f"{r.title}\n{r.instructions}") for r in recipes]
    embeddings = provider.embed_documents([c.text for c in chunks])

    upsert_recipes(recipes)
    upsert_chunks(chunks, embeddings)


def test_normalize_scales_to_unit_range():
    normalized = _normalize({"a": 1.0, "b": 3.0, "c": 5.0})

    assert normalized["a"] == 0.0
    assert normalized["b"] == 0.5
    assert normalized["c"] == 1.0


def test_normalize_handles_empty_input():
    assert _normalize({}) == {}


def test_normalize_handles_tied_scores():
    assert _normalize({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_hybrid_search_finds_exact_rare_term_via_keyword_branch():
    # "Harissa" barely resembles the other recipes semantically, so this
    # mainly exercises whether the keyword branch is actually contributing.
    results, _ = hybrid_search("harissa", top_k=3)

    assert results[0].recipe_id == "3"


def test_hybrid_search_alpha_one_matches_pure_semantic_ranking():
    from app.search.semantic import semantic_search

    semantic_results, _ = semantic_search("warm comfort food", top_k=4)
    hybrid_results, _ = hybrid_search("warm comfort food", top_k=4, alpha=1.0)

    assert [r.recipe_id for r in hybrid_results] == [r.recipe_id for r in semantic_results]


def test_hybrid_search_alpha_zero_ranks_keyword_matches_first():
    # Hybrid's candidate pool is a superset of keyword_search's (it also
    # includes semantic-only matches, scored 0 here), so the two result lists
    # aren't identical -- but every keyword match should still outrank every
    # recipe keyword_search didn't find at all.
    from app.search.keyword import keyword_search

    keyword_results, _ = keyword_search("beef burger", top_k=4)
    hybrid_results, _ = hybrid_search("beef burger", top_k=4, alpha=0.0)

    matched_ids = {r.recipe_id for r in keyword_results}
    hybrid_ids = [r.recipe_id for r in hybrid_results]
    matched_rank = [i for i, rid in enumerate(hybrid_ids) if rid in matched_ids]
    unmatched_rank = [i for i, rid in enumerate(hybrid_ids) if rid not in matched_ids]

    assert hybrid_results[0].recipe_id == keyword_results[0].recipe_id
    assert max(matched_rank, default=-1) < min(unmatched_rank, default=len(hybrid_ids))


def test_hybrid_search_respects_top_k():
    results, _ = hybrid_search("food", top_k=2)

    assert len(results) == 2


def test_hybrid_search_scores_are_descending():
    results, _ = hybrid_search("chicken curry", top_k=4)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_includes_recipes_matched_by_only_one_branch():
    # "Harissa" has no semantic neighbours among the other three recipes but
    # should still surface via the keyword branch rather than being dropped.
    results, _ = hybrid_search("harissa", top_k=4)

    assert "3" in {r.recipe_id for r in results}


def test_endpoint_returns_expected_shape():
    with TestClient(app) as client:
        response = client.get("/search/hybrid", params={"q": "harissa", "alpha": 0.5})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["results"][0]["recipe_id"] == "3"


def test_endpoint_rejects_alpha_out_of_range():
    with TestClient(app) as client:
        response = client.get("/search/hybrid", params={"q": "harissa", "alpha": 1.5})

    assert response.status_code == 422


def test_endpoint_rerank_toggle_is_wired_up(monkeypatch):
    """Verify the route actually calls reranking, without loading the real model."""
    from app.search import rerank as rerank_module

    monkeypatch.setattr(
        rerank_module,
        "rerank",
        lambda query, candidates, top_k: candidates[:top_k],
    )
    # routes.py imported `rerank` by name, so the route's own reference must
    # be patched too -- patching the source module alone wouldn't reach it.
    monkeypatch.setattr("app.routes.rerank_candidates", rerank_module.rerank)

    with TestClient(app) as client:
        response = client.get("/search/hybrid", params={"q": "harissa", "rerank": True})

    assert response.status_code == 200
    assert response.json()["mode"] == "hybrid+rerank"
