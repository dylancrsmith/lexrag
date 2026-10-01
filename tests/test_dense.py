from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from lexrag.config import EmbeddingSettings
from lexrag.index import BM25Retriever, DenseRetriever, Embedder, build_retriever, tokenize
from lexrag.models import Chunk

DIM = 64


class FakeEmbedder:
    """Bag-of-words hashed into a small normalised vector. Records every text it embeds."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.seen.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, hash(token) % DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1, norms)).astype(np.float32)


def chunk(key: str, text: str) -> Chunk:
    return Chunk(chunk_id=f"{key}/0", section_keys=(key,), text=text, embed_text=f"HEADER {text}")


CORPUS = [
    chunk("wtr#reg-12", "rest break twenty minutes"),
    chunk("era#s-86", "notice terminate employment"),
    chunk("era#s-8", "itemised pay statement"),
]


def test_chunks_embedded_once_with_header_and_queries_get_the_instruction() -> None:
    fake = FakeEmbedder()
    retriever = DenseRetriever(CORPUS, fake, query_instruction="Query: ")
    assert fake.seen == [c.embed_text for c in CORPUS]

    retriever.search("rest break", k=1)
    assert fake.seen[-1] == "Query: rest break"
    assert len(fake.seen) == len(CORPUS) + 1  # the index is not re-embedded per query


def test_ranks_by_similarity() -> None:
    results = DenseRetriever(CORPUS, FakeEmbedder()).search("pay statement", k=3)
    assert results[0].section_key == "era#s-8"
    assert results[0].score > results[1].score


def test_always_returns_k_results_even_when_nothing_is_similar() -> None:
    # Unlike BM25, similarity search has no "no match": weak results still come back, ranked.
    assert len(DenseRetriever(CORPUS, FakeEmbedder()).search("zzz qqq", k=3)) == 3


def test_empty_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        DenseRetriever([], FakeEmbedder())


def test_build_retriever_modes() -> None:
    settings = EmbeddingSettings(query_instruction="Q: ")
    assert isinstance(build_retriever("bm25", CORPUS, settings), BM25Retriever)
    dense = build_retriever("dense", CORPUS, settings, embedder=FakeEmbedder())
    assert isinstance(dense, DenseRetriever)
    assert dense.query_instruction == "Q: "
    with pytest.raises(ValueError, match="unknown mode 'magic'"):
        build_retriever("magic", CORPUS, settings)


# --- the real model: runs only where the `ml` extra is installed (not in CI) ----------------------


@pytest.fixture(scope="module")
def real_embedder() -> Iterator[Embedder]:
    pytest.importorskip("sentence_transformers")
    from lexrag.index import SentenceTransformerEmbedder

    yield SentenceTransformerEmbedder(EmbeddingSettings())


LEGAL = {
    "fsa#s-14": "Selling food not of the nature or substance or quality demanded. Any person who "
    "sells to the purchaser's prejudice any food which is not of the nature or substance or "
    "quality demanded by the purchaser shall be guilty of an offence.",
    "wtr#reg-12": "Rest breaks. Where a worker's daily working time is more than six hours, he is "
    "entitled to a rest break.",
    "era#s-86": "Rights of employer and employee to minimum notice. The notice required to be "
    "given by an employer to terminate the contract of employment.",
    "era#s-8": "Itemised pay statement. A worker has the right to be given by his employer a "
    "written itemised pay statement.",
}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("chef told us to plate up basa when the menu says cod. is that illegal?", "fsa#s-14"),
        ("my boss doesnt give us payslips, the money just goes in the bank", "era#s-8"),
    ],
)
def test_real_model_bridges_vocabulary_mismatch(
    real_embedder: Embedder, question: str, expected: str
) -> None:
    chunks = [
        Chunk(chunk_id=k + "/0", section_keys=(k,), text=v, embed_text=v) for k, v in LEGAL.items()
    ]
    assert BM25Retriever(chunks).search(question, k=1) == []  # keywords share nothing...
    dense = DenseRetriever(chunks, real_embedder, EmbeddingSettings().query_instruction)
    assert dense.search(question, k=1)[0].section_key == expected  # ...meaning does
