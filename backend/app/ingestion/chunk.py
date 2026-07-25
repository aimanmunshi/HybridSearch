"""Split recipes into embeddable chunks.

Why chunk at all? `all-MiniLM-L6-v2` truncates input at 256 word-pieces
(~190 English words). A typical recipe here runs well past that once
ingredients and instructions are concatenated, so embedding the whole recipe
would silently discard most of the method -- queries like "simmer for two
hours" would never match, because that text never reached the model.

Chunking strategy:

1. Split the body into sentences, then greedily pack sentences into windows of
   ~`TARGET_WORDS` words. Packing on sentence boundaries keeps each chunk a
   coherent unit rather than cutting mid-step.
2. Overlap consecutive windows by ~`OVERLAP_WORDS` words so an instruction that
   straddles a boundary still appears intact in one of them.
3. Prepend the recipe header (title, cuisine, category, tags) to every chunk.
   Without it, chunk 3 of a recipe is an anonymous list of steps; with it, the
   embedding still knows this is a Thai curry.

Retrieval happens at chunk level, but results are aggregated back to one entry
per recipe (best-scoring chunk wins) -- the user is looking for a recipe, not a
paragraph.
"""
from __future__ import annotations

import logging
import re

from app.models.recipe import Chunk, Recipe

logger = logging.getLogger(__name__)

# Budget in words, chosen to stay under the model's 256 word-piece limit.
# English averages ~1.3 word-pieces per word, and the header costs ~15 words,
# so ~140 body words leaves comfortable headroom.
TARGET_WORDS = 140
OVERLAP_WORDS = 30
# A trailing fragment shorter than this is merged into the previous chunk
# instead of becoming a near-empty chunk of its own.
MIN_CHUNK_WORDS = 25

# Split after ., !, ? or a newline. Avoids splitting on decimals ("1.5 cups")
# by requiring the following character to be whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Split body text into non-empty, whitespace-trimmed sentences."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _pack_windows(sentences: list[str]) -> list[str]:
    """Greedily pack sentences into overlapping word-budgeted windows."""
    if not sentences:
        return []

    windows: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # Flush the window once adding this sentence would exceed the budget.
        # `current` is checked so a single over-long sentence still lands in a
        # window of its own rather than looping forever.
        if current and current_words + sentence_words > TARGET_WORDS:
            windows.append(" ".join(current))
            # Carry the tail of the window forward as overlap.
            overlap: list[str] = []
            overlap_words = 0
            for prev in reversed(current):
                prev_words = len(prev.split())
                if overlap_words + prev_words > OVERLAP_WORDS:
                    break
                overlap.insert(0, prev)
                overlap_words += prev_words
            current = overlap
            current_words = overlap_words

        current.append(sentence)
        current_words += sentence_words

    if current:
        tail = " ".join(current)
        # Fold a stubby final window back into its predecessor.
        if windows and len(tail.split()) < MIN_CHUNK_WORDS:
            windows[-1] = f"{windows[-1]} {tail}"
        else:
            windows.append(tail)

    return windows


def chunk_recipe(recipe: Recipe) -> list[Chunk]:
    """Split one recipe into header-prefixed, overlapping chunks."""
    header = recipe.header_text()
    windows = _pack_windows(_split_sentences(recipe.body_text()))

    # A recipe with no usable body still deserves one chunk so it stays
    # findable by title.
    if not windows:
        windows = [""]

    return [
        Chunk(
            recipe_id=recipe.id,
            chunk_index=index,
            text=f"{header}\n{window}".strip(),
        )
        for index, window in enumerate(windows)
    ]


def chunk_recipes(recipes: list[Recipe]) -> list[Chunk]:
    """Chunk a batch of recipes, logging the resulting fan-out."""
    chunks = [chunk for recipe in recipes for chunk in chunk_recipe(recipe)]
    if recipes:
        logger.info(
            "chunked %d recipes into %d chunks (%.2f per recipe)",
            len(recipes),
            len(chunks),
            len(chunks) / len(recipes),
        )
    return chunks
