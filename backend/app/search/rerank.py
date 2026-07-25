"""Cross-encoder reranking of an already-retrieved candidate set.

A bi-encoder (what `embed.py` uses) embeds the query and each document
*independently*, so similarity is a single dot product -- cheap enough to run
against the whole corpus, but the model never sees query and document
together. A cross-encoder feeds the (query, document) pair through the model
jointly, so it can model interactions between them (e.g. word-for-word
alignment) that two separate embeddings throw away. That makes it
meaningfully more accurate, but it costs one forward pass *per candidate* --
there is no way to precompute a document's representation ahead of time, so it
cannot run over an entire corpus the way retrieval does.

The standard resolution, used here: retrieve a modest candidate set cheaply
with hybrid search, then rerank only that shortlist with the cross-encoder.
This is why reranking is a second pass bolted onto Phase 5's results rather
than a replacement for them.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.models.search import SearchResult

logger = logging.getLogger(__name__)

_model = None


def _get_cross_encoder():
    """Load the cross-encoder on first use, then reuse it (mirrors embed.py)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("loading cross-encoder %s", settings.cross_encoder_model)
        _model = CrossEncoder(settings.cross_encoder_model)
    return _model


def rerank(query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Rerank `candidates` by cross-encoder relevance to `query`.

    Each candidate's `score` is replaced with the cross-encoder's raw logit
    (unbounded, not comparable to the hybrid score it replaces) so the
    response is honest about which stage actually produced the final order.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    # The snippet (not the title alone) is what the cross-encoder judges
    # relevance against -- it's the same passage the user sees, so the score
    # they're shown corresponds to text they can actually read.
    pairs = [(query, c.snippet) for c in candidates]
    scores = model.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [
        candidate.model_copy(update={"score": float(score)})
        for candidate, score in reranked[:top_k]
    ]
