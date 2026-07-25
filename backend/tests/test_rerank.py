"""Tests for cross-encoder reranking.

Like test_embed.py, the real CrossEncoder model is swapped for a fake: what
matters here is the reordering/truncation contract, not the model's judgment
of relevance (that's measured end-to-end in the Phase 8 eval harness).
"""
import pytest

from app.models.search import SearchResult
from app.search import rerank as rerank_module


class FakeCrossEncoder:
    """Scores a (query, text) pair by how many query words appear in it."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def make_result(recipe_id: str, snippet: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        recipe_id=recipe_id, title=f"Recipe {recipe_id}", snippet=snippet, score=score
    )


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    monkeypatch.setattr(rerank_module, "_model", None)
    monkeypatch.setattr(rerank_module, "_get_cross_encoder", lambda: FakeCrossEncoder())
    yield
    monkeypatch.setattr(rerank_module, "_model", None)


def test_rerank_reorders_by_cross_encoder_score():
    candidates = [
        make_result("1", "a bland unrelated snippet about nothing much"),
        make_result("2", "spicy chicken curry with rice and coconut milk"),
    ]

    results = rerank_module.rerank("spicy chicken curry", candidates, top_k=2)

    assert results[0].recipe_id == "2"


def test_rerank_truncates_to_top_k():
    candidates = [make_result(str(i), "chicken curry rice") for i in range(5)]

    results = rerank_module.rerank("chicken curry", candidates, top_k=2)

    assert len(results) == 2


def test_rerank_replaces_score_with_cross_encoder_output():
    candidates = [make_result("1", "chicken curry", score=0.123)]

    results = rerank_module.rerank("chicken curry", candidates, top_k=1)

    assert results[0].score == 2.0  # both words of the query matched


def test_rerank_empty_candidates_returns_empty_list():
    assert rerank_module.rerank("anything", [], top_k=5) == []


def test_rerank_preserves_other_result_fields():
    candidates = [make_result("1", "chicken curry")]
    candidates[0] = candidates[0].model_copy(update={"cuisine": "Indian", "tags": ["Spicy"]})

    results = rerank_module.rerank("chicken curry", candidates, top_k=1)

    assert results[0].cuisine == "Indian"
    assert results[0].tags == ["Spicy"]
    assert results[0].title == "Recipe 1"
