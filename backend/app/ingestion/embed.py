"""Embedding providers.

Two interchangeable backends sit behind one `EmbeddingProvider` protocol:

- `SentenceTransformerProvider` (default) runs `all-MiniLM-L6-v2` locally. It
  needs no API key, costs nothing, and keeps the project runnable offline --
  which is why it is the default rather than a fallback.
- `OpenAIProvider` uses `text-embedding-3-small`. Higher quality, but it adds a
  paid dependency and a network round-trip to every query, so it is opt-in via
  `USE_OPENAI_EMBEDDINGS=true`.

The two produce different dimensionalities (384 vs 1536), so the `chunks.embedding`
column is sized from the active provider at migration time. Switching providers
therefore requires a re-index, which `scripts/index_corpus.py --recreate` does.

Models are loaded lazily. Importing this module must stay cheap so that FastAPI
startup, tests, and CLI tools that never embed anything do not pay for it.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.config import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into vectors for storage and querying."""

    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of corpus documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class SentenceTransformerProvider:
    """Local sentence-transformers backend (the default)."""

    # Dimensionality is a property of the checkpoint, so it is read off the
    # loaded model rather than hard-coded.
    def __init__(self, model_name: str | None = None, batch_size: int = 64):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Load the model on first use, then reuse it."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # normalize_embeddings=True makes every vector unit length, so cosine
        # distance reduces to a dot product -- cheaper for pgvector, and it
        # keeps stored vectors on a common scale.
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # all-MiniLM-L6-v2 is a symmetric model: queries and documents go
        # through the same encoder with no instruction prefix. (Asymmetric
        # models like E5 or BGE would need "query: " / "passage: " here.)
        return self.embed_documents([text])[0]


class OpenAIProvider:
    """OpenAI embeddings backend (opt-in)."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        if not settings.openai_api_key:
            raise ValueError(
                "USE_OPENAI_EMBEDDINGS is true but OPENAI_API_KEY is not set."
            )
        from openai import OpenAI

        self.model_name = model_name
        self.dimensions = 1536
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        # The endpoint caps inputs per request, so send in batches.
        for start in range(0, len(texts), 256):
            batch = texts[start : start + 256]
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured provider as a process-wide singleton.

    Caching matters: `all-MiniLM-L6-v2` takes a couple of seconds to load, and
    re-loading it per request would dominate search latency.
    """
    global _provider
    if _provider is None:
        if settings.use_openai_embeddings:
            logger.info("using OpenAI embeddings")
            _provider = OpenAIProvider()
        else:
            _provider = SentenceTransformerProvider()
    return _provider
