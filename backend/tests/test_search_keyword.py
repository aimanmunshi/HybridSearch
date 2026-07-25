"""Tests for Postgres full-text keyword search."""
import pytest
from fastapi.testclient import TestClient

from app.db.repository import upsert_recipes
from app.main import app
from app.models.recipe import Ingredient, Recipe
from app.search.keyword import keyword_search


@pytest.fixture(autouse=True)
def _seed_recipes(_database):
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
    ]
    upsert_recipes(recipes)


def test_keyword_search_finds_exact_title_match():
    results, _ = keyword_search("chicken curry", top_k=5)

    assert results[0].recipe_id == "1"


def test_keyword_search_finds_rare_ingredient_by_exact_term():
    # "harissa" is a specific enough term that lexical matching should nail it.
    results, _ = keyword_search("harissa", top_k=5)

    assert len(results) == 1
    assert results[0].recipe_id == "3"


def test_keyword_search_respects_top_k():
    results, _ = keyword_search("beef OR chicken OR harissa", top_k=2)

    assert len(results) == 2


def test_keyword_search_returns_nothing_for_unmatched_terms():
    results, _ = keyword_search("submarine spacecraft", top_k=5)

    assert results == []


def test_keyword_search_handles_stopword_only_query_without_error():
    # websearch_to_tsquery should degrade gracefully rather than raising.
    results, _ = keyword_search("the a of", top_k=5)

    assert results == []


def test_keyword_search_scores_are_descending():
    results, _ = keyword_search("chicken OR beef OR harissa", top_k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_endpoint_returns_expected_shape():
    with TestClient(app) as client:
        response = client.get("/search/keyword", params={"q": "harissa"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "keyword"
    assert body["results"][0]["recipe_id"] == "3"
