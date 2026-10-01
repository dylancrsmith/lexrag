"""Retrieval over chunks. Search scores chunks, but results are sections.

A long section may be split into several chunks; a fixed-size window may span several sections.
Either way, top-k means k *distinct sections*, each represented by its best-scoring chunk, because
gold labels, citations and metrics are all per section.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import snowballstemmer
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from lexrag.config import EmbeddingSettings
from lexrag.models import Chunk, Retrieved

Analyzer = Callable[[str], list[str]]

# Lower-cased words and numbers; decimals stay whole so "12.07" and "11.10" can match.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_POSSESSIVE = re.compile(r"['’]s\b")
# Lucene's default English stop set, which is also Elasticsearch's `_english_`. Deliberately the
# standard list, not one picked to suit this test set.
# fmt: off
ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
})
# fmt: on
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Raw tokens: lower-cased words and numbers, nothing removed."""
    return _TOKEN.findall(text.lower())


def analyze(text: str) -> list[str]:
    """The standard English analyzer: drop possessive 's, stopwords, then Snowball-stem."""
    tokens = tokenize(_POSSESSIVE.sub("", text))
    return list(_stemmer.stemWords([t for t in tokens if t not in ENGLISH_STOPWORDS]))


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Retrieved]: ...


def top_sections(
    chunks: Sequence[Chunk], scores: NDArray[np.float64], k: int, *, positive_only: bool = True
) -> list[Retrieved]:
    """The k best sections, each scored by its best chunk.

    With `positive_only` (keyword search), chunks scoring <= 0 matched nothing and never count.
    Similarity scores (dense search) can be legitimately small or negative, so pass False.
    """
    results: dict[str, Retrieved] = {}
    for i in np.argsort(-scores, kind="stable"):
        if len(results) >= k or (positive_only and scores[i] <= 0):
            break
        for key in chunks[i].section_keys:
            if key not in results and len(results) < k:
                results[key] = Retrieved(section_key=key, score=float(scores[i]), chunk=chunks[i])
    return list(results.values())


class BM25Retriever:
    """Okapi BM25 keyword search over each chunk's `embed_text` (header + text)."""

    def __init__(self, chunks: Sequence[Chunk], analyzer: Analyzer = analyze) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = list(chunks)
        self.analyzer = analyzer
        self._bm25 = BM25Okapi([analyzer(c.embed_text) for c in self.chunks])

    def scores(self, query: str) -> NDArray[np.float64]:
        return np.asarray(self._bm25.get_scores(self.analyzer(query)), dtype=np.float64)

    def search(self, query: str, k: int) -> list[Retrieved]:
        return top_sections(self.chunks, self.scores(query), k)


# --- dense ---------------------------------------------------------------------------------------


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One L2-normalised row per text."""
        ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model. Needs the optional `ml` extra (torch)."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self.settings = settings
        self.model = SentenceTransformer(settings.model, device=settings.device)

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseRetriever:
    """Cosine-similarity search: chunk and query vectors are normalised, so a dot product."""

    def __init__(
        self, chunks: Sequence[Chunk], embedder: Embedder, query_instruction: str = ""
    ) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = list(chunks)
        self.embedder = embedder
        self.query_instruction = query_instruction
        self.vectors = embedder.embed([c.embed_text for c in self.chunks])

    def scores(self, query: str) -> NDArray[np.float64]:
        q = self.embedder.embed([self.query_instruction + query])[0]
        return np.asarray(self.vectors @ q, dtype=np.float64)

    def search(self, query: str, k: int) -> list[Retrieved]:
        return top_sections(self.chunks, self.scores(query), k, positive_only=False)


# --- construction --------------------------------------------------------------------------------

MODES = ("bm25", "bm25-raw", "dense")


def build_retriever(
    mode: str,
    chunks: Sequence[Chunk],
    embedding: EmbeddingSettings,
    embedder: Embedder | None = None,
) -> Retriever:
    if mode == "bm25":
        return BM25Retriever(chunks, analyzer=analyze)
    if mode == "bm25-raw":
        return BM25Retriever(chunks, analyzer=tokenize)
    if mode == "dense":
        embedder = embedder or SentenceTransformerEmbedder(embedding)
        return DenseRetriever(chunks, embedder, embedding.query_instruction)
    raise ValueError(f"unknown mode {mode!r}; available: {', '.join(MODES)}")
