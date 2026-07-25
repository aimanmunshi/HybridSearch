"""Tests for the embedding provider abstraction.

The real SentenceTransformer model is deliberately not exercised here: loading
it is slow (seconds) and would make the unit suite network- and
hardware-dependent. What's worth testing at this layer is the *contract*
(normalization, batching, singleton caching), not the model's output quality --
that belongs in the eval harness (Phase 8), which measures it end-to-end.
"""
import numpy as np
import pytest

from app.ingestion import embed


class FakeSentenceTransformer:
    """Stands in for `SentenceTransformer`: deterministic, no I/O."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        vectors = np.array([[hash((t, i)) % 97 for i in range(self.dim)] for t in texts], dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return vectors


@pytest.fixture
def provider(monkeypatch):
    fake = FakeSentenceTransformer()
    p = embed.SentenceTransformerProvider(model_name="fake-model")
    monkeypatch.setattr(type(p), "model", property(lambda self: fake))
    return p


def test_embed_documents_returns_unit_vectors(provider):
    vectors = provider.embed_documents(["chicken curry", "beef stew"])

    assert len(vectors) == 2
    for v in vectors:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_embed_documents_empty_input_returns_empty_list(provider):
    assert provider.embed_documents([]) == []


def test_embed_query_returns_single_vector(provider):
    vector = provider.embed_query("chicken curry")

    assert isinstance(vector, list)
    assert len(vector) == provider.dimensions


def test_dimensions_reflects_the_underlying_model(provider):
    assert provider.dimensions == 4


def test_get_embedding_provider_is_a_singleton(monkeypatch):
    monkeypatch.setattr(embed, "_provider", None)
    monkeypatch.setattr(embed.settings, "use_openai_embeddings", False)

    first = embed.get_embedding_provider()
    second = embed.get_embedding_provider()

    assert first is second
    monkeypatch.setattr(embed, "_provider", None)  # avoid leaking into other tests


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(embed.settings, "openai_api_key", None)

    with pytest.raises(ValueError):
        embed.OpenAIProvider()
