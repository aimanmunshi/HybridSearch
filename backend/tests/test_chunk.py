"""Tests for the chunking stage."""
from app.ingestion.chunk import MIN_CHUNK_WORDS, TARGET_WORDS, chunk_recipe
from app.models.recipe import Ingredient, Recipe


def make_recipe(instructions: str, **overrides) -> Recipe:
    fields = {
        "id": "1001",
        "title": "Test Curry",
        "category": "Chicken",
        "cuisine": "Thai",
        "tags": ["Spicy"],
        "ingredients": [Ingredient(name="Chicken", measure="500 g")],
        "instructions": instructions,
    }
    fields.update(overrides)
    return Recipe(**fields)


def test_short_recipe_yields_single_chunk():
    chunks = chunk_recipe(make_recipe("Heat the oil. Add the paste."))

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].recipe_id == "1001"


def test_every_chunk_carries_the_header():
    long_instructions = " ".join(f"Step number {n} happens now." for n in range(200))
    chunks = chunk_recipe(make_recipe(long_instructions))

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Test Curry | Thai cuisine | Chicken | Spicy")


def test_chunks_stay_within_the_word_budget():
    long_instructions = " ".join(f"Step number {n} happens now." for n in range(200))
    chunks = chunk_recipe(make_recipe(long_instructions))

    # Header adds a handful of words on top of the body budget.
    for chunk in chunks:
        assert len(chunk.text.split()) <= TARGET_WORDS + 40


def test_chunk_indices_are_sequential():
    long_instructions = " ".join(f"Step number {n} happens now." for n in range(200))
    chunks = chunk_recipe(make_recipe(long_instructions))

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    sentences = [f"Unique marker {n} appears here." for n in range(60)]
    chunks = chunk_recipe(make_recipe(" ".join(sentences)))

    assert len(chunks) > 1
    # The tail of one chunk should reappear at the head of the next.
    first_body = chunks[0].text.split("\n", 1)[1].split()
    second_body = chunks[1].text.split("\n", 1)[1].split()
    assert set(first_body[-10:]) & set(second_body[:20])


def test_no_stubby_trailing_chunk():
    # Length tuned to leave a small remainder after the first window.
    sentences = [f"Sentence {n} is here now." for n in range(30)]
    chunks = chunk_recipe(make_recipe(" ".join(sentences)))

    bodies = [c.text.split("\n", 1)[1] for c in chunks]
    assert all(len(b.split()) >= MIN_CHUNK_WORDS for b in bodies)


def test_recipe_without_body_still_produces_a_chunk():
    recipe = make_recipe("x", ingredients=[])
    recipe.instructions = ""
    chunks = chunk_recipe(recipe)

    assert len(chunks) == 1
    assert "Test Curry" in chunks[0].text


def test_oversized_single_sentence_does_not_loop():
    giant = " ".join(["word"] * (TARGET_WORDS * 3)) + "."
    chunks = chunk_recipe(make_recipe(giant))

    assert 0 < len(chunks) < 5
