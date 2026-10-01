"""Retrieval over chunks. Search scores chunks, but results are sections.

A long section may be split into several chunks; a fixed-size window may span several sections.
Either way, top-k means k *distinct sections*, each represented by its best-scoring chunk, because
gold labels, citations and metrics are all per section.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import snowballstemmer
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

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


def top_sections(chunks: Sequence[Chunk], scores: NDArray[np.float64], k: int) -> list[Retrieved]:
    """The k best sections, each scored by its best chunk. Chunks scoring <= 0 never count."""
    results: dict[str, Retrieved] = {}
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= 0 or len(results) >= k:
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
