"""Tests for the Postgres write path (upsert idempotency, orphan cleanup)."""
from app.db.connection import get_pool
from app.db.repository import (
    corpus_stats,
    delete_orphaned_chunks,
    upsert_chunks,
    upsert_recipes,
)
from app.models.recipe import Chunk, Ingredient, Recipe

DIM = 8


def make_recipe(recipe_id: str, title: str = "Test Curry") -> Recipe:
    return Recipe(
        id=recipe_id,
        title=title,
        category="Chicken",
        cuisine="Thai",
        tags=["Spicy"],
        ingredients=[Ingredient(name="Chicken", measure="500 g")],
        instructions="Heat the oil. Add the paste.",
    )


def make_chunk(recipe_id: str, index: int, text: str = "chunk text") -> Chunk:
    return Chunk(recipe_id=recipe_id, chunk_index=index, text=text)


def vector(seed: float) -> list[float]:
    return [seed] * DIM


def test_upsert_recipes_is_idempotent():
    recipe = make_recipe("1")

    assert upsert_recipes([recipe]) == 1
    assert upsert_recipes([recipe]) == 1  # second run must not duplicate

    stats = corpus_stats()
    assert stats["recipes"] == 1


def test_upsert_recipes_updates_in_place():
    upsert_recipes([make_recipe("1", title="Original Title")])
    upsert_recipes([make_recipe("1", title="Updated Title")])

    with get_pool().connection() as conn:
        title = conn.execute("SELECT title FROM recipes WHERE id = '1'").fetchone()[0]
    assert title == "Updated Title"
    assert corpus_stats()["recipes"] == 1


def test_upsert_chunks_is_idempotent():
    upsert_recipes([make_recipe("1")])
    chunk = make_chunk("1", 0)

    assert upsert_chunks([chunk], [vector(0.1)]) == 1
    assert upsert_chunks([chunk], [vector(0.2)]) == 1

    stats = corpus_stats()
    assert stats["chunks"] == 1


def test_upsert_chunks_rejects_length_mismatch():
    upsert_recipes([make_recipe("1")])
    import pytest

    with pytest.raises(ValueError):
        upsert_chunks([make_chunk("1", 0), make_chunk("1", 1)], [vector(0.1)])


def test_delete_orphaned_chunks_removes_stale_trailing_indices():
    upsert_recipes([make_recipe("1")])
    # Simulate a recipe that used to have 3 chunks.
    upsert_chunks(
        [make_chunk("1", 0), make_chunk("1", 1), make_chunk("1", 2)],
        [vector(0.1), vector(0.2), vector(0.3)],
    )

    # Re-ingest with only 1 chunk now (recipe got shorter).
    upsert_chunks([make_chunk("1", 0)], [vector(0.1)])
    removed = delete_orphaned_chunks([make_chunk("1", 0)])

    assert removed == 2
    assert corpus_stats()["chunks"] == 1


def test_deleting_a_recipe_cascades_to_its_chunks():
    upsert_recipes([make_recipe("1")])
    upsert_chunks([make_chunk("1", 0)], [vector(0.1)])

    with get_pool().connection() as conn:
        conn.execute("DELETE FROM recipes WHERE id = '1'")

    assert corpus_stats() == {"recipes": 0, "chunks": 0}


def test_corpus_stats_reflects_multiple_recipes():
    upsert_recipes([make_recipe("1"), make_recipe("2")])
    upsert_chunks(
        [make_chunk("1", 0), make_chunk("2", 0), make_chunk("2", 1)],
        [vector(0.1), vector(0.2), vector(0.3)],
    )

    assert corpus_stats() == {"recipes": 2, "chunks": 3}
