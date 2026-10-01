import numpy as np
import pytest

from lexrag.index import BM25Retriever, tokenize, top_sections
from lexrag.models import Chunk


def chunk(chunk_id: str, text: str, *section_keys: str) -> Chunk:
    keys = section_keys or (chunk_id.rsplit("/", 1)[0],)
    return Chunk(chunk_id=chunk_id, section_keys=keys, text=text, embed_text=text)


CORPUS = [
    chunk("wtr#reg-12/0", "Rest breaks. A worker is entitled to a rest break of 20 minutes."),
    chunk("wtr#reg-10/0", "Daily rest. A worker is entitled to eleven consecutive hours rest."),
    chunk("era#s-8/0", "Itemised pay statement. A worker has the right to a payslip."),
    chunk("era#s-86/0", "Minimum notice. The employer must give one week of notice."),
    chunk("era#s-86/1", "Notice of twelve weeks after twelve years of employment."),
    chunk("fsa#s-14/0", "Food not of the nature or substance or quality demanded."),
]


def test_tokenize_lowercases_and_keeps_decimals() -> None:
    assert tokenize("£11.10 a DAY, see reg. 15B (12.07%)") == [
        "11.10", "a", "day", "see", "reg", "15b", "12.07",
    ]  # fmt: skip


def test_bm25_finds_the_section_using_the_same_words() -> None:
    results = BM25Retriever(CORPUS).search("do I get a payslip", k=3)
    assert results[0].section_key == "era#s-8"


def test_results_are_distinct_sections_scored_by_their_best_chunk() -> None:
    results = BM25Retriever(CORPUS).search("notice twelve weeks", k=5)
    keys = [r.section_key for r in results]
    assert keys[0] == "era#s-86"
    assert keys.count("era#s-86") == 1  # two matching chunks, one result
    assert results[0].chunk.chunk_id == "era#s-86/1"  # the better of the two


def test_vocabulary_mismatch_finds_nothing() -> None:
    # The baseline's known weakness: no shared words, no match at all.
    assert BM25Retriever(CORPUS).search("plate up basa when the menu says cod", k=5) == []


def test_k_limits_the_number_of_sections() -> None:
    assert len(BM25Retriever(CORPUS).search("a worker is entitled", k=2)) == 2


def test_a_window_spanning_two_sections_credits_both() -> None:
    spanning = chunk("era@0", "text", "era#s-1", "era#s-2")
    results = top_sections([spanning], np.array([1.0]), k=5)
    assert [r.section_key for r in results] == ["era#s-1", "era#s-2"]
    assert all(r.chunk is spanning for r in results)


def test_top_sections_stops_at_k_even_inside_a_spanning_window() -> None:
    spanning = chunk("era@0", "text", "era#s-1", "era#s-2", "era#s-3")
    assert len(top_sections([spanning], np.array([1.0]), k=2)) == 2


def test_empty_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        BM25Retriever([])
