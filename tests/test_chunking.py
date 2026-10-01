from itertools import pairwise

import pytest
from pydantic import ValidationError

from lexrag.chunking import chunk_sections, header, split_lines
from lexrag.config import ChunkingSettings
from lexrag.models import Section


def make_section(text: str, section_id: str = "regulation-12", doc_id: str = "wtr1998") -> Section:
    return Section(
        doc_id=doc_id,
        section_id=section_id,
        label=section_id.replace("-", " "),
        title="Rest breaks",
        path=("PART II: Rights and obligations",),
        text=text,
        url="https://example.org",
        doc_title="The Working Time Regulations 1998",
        authority="law",
    )


def numbered_lines(n: int, width: int = 50) -> list[str]:
    """Distinct lines like "(7) xxxx...", each exactly `width` characters."""
    return [f"({i}) ".ljust(width, "x") for i in range(1, n + 1)]


# --- structure-aware -----------------------------------------------------------------------------


def test_short_section_is_one_chunk_with_header() -> None:
    section = make_section("(1) A worker is entitled to a rest break.")
    [chunk] = chunk_sections([section], ChunkingSettings())

    assert chunk.chunk_id == "wtr1998#regulation-12/0"
    assert chunk.section_keys == ("wtr1998#regulation-12",)
    assert chunk.text == section.text
    assert chunk.embed_text == (
        "The Working Time Regulations 1998 > PART II: Rights and obligations"
        " > regulation 12: Rest breaks\n(1) A worker is entitled to a rest break."
    )


def test_header_without_title_uses_label_only() -> None:
    section = make_section("text").model_copy(update={"title": "", "path": ()})
    assert header(section) == "The Working Time Regulations 1998 > regulation 12"


def test_long_section_splits_between_lines_and_loses_nothing() -> None:
    lines = numbered_lines(20)  # 20 x 50 chars + newlines = 1019 chars
    pieces = split_lines("\n".join(lines), max_chars=300, overlap_chars=60)

    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    piece_lines = [line for p in pieces for line in p.split("\n")]
    assert set(piece_lines) == set(lines)  # every line kept whole, none invented
    # ...and in their original order once the repeated overlap lines are removed
    assert list(dict.fromkeys(piece_lines)) == lines


def test_each_piece_repeats_the_previous_last_line_as_overlap() -> None:
    pieces = split_lines("\n".join(numbered_lines(20)), max_chars=300, overlap_chars=60)
    for before, after in pairwise(pieces):
        assert after.split("\n")[0] == before.split("\n")[-1]


def test_no_overlap_when_disabled() -> None:
    pieces = split_lines("\n".join(numbered_lines(20)), max_chars=300, overlap_chars=0)
    piece_lines = [line for p in pieces for line in p.split("\n")]
    assert len(piece_lines) == 20


def test_overlong_line_splits_at_sentence_ends() -> None:
    sentences = [f"Sentence number {i} says something about rest breaks." for i in range(30)]
    line = "  " + " ".join(sentences)  # one indented line, ~1600 chars
    pieces = split_lines(line, max_chars=200, overlap_chars=0)

    assert all(len(p) <= 200 for p in pieces)
    assert all(p.startswith("  ") for p in pieces)  # indentation kept on every piece
    assert all(p.rstrip().endswith(".") for p in pieces)  # cut at sentence ends
    assert " ".join(p.strip() for p in pieces).split() == line.split()  # no words lost


def test_word_longer_than_limit_is_hard_cut() -> None:
    pieces = split_lines("x" * 450, max_chars=200, overlap_chars=50)
    assert all(len(p) <= 200 for p in pieces)
    assert "".join(p for p in pieces).count("x") >= 450


def test_each_long_section_gets_numbered_chunks() -> None:
    section = make_section("\n".join(numbered_lines(60)))  # ~3000 chars
    chunks = chunk_sections([section], ChunkingSettings())
    assert [c.chunk_id for c in chunks] == [
        f"wtr1998#regulation-12/{i}" for i in range(len(chunks))
    ]
    assert len(chunks) == 2
    assert all(c.embed_text.startswith("The Working Time Regulations 1998 >") for c in chunks)


# --- fixed-size baseline -------------------------------------------------------------------------


def test_fixed_windows_ignore_structure_but_record_every_section_they_touch() -> None:
    words = " ".join(f"word{i}" for i in range(400))  # ~3000 chars per section
    sections = [make_section(words, "regulation-1"), make_section(words, "regulation-2")]
    settings = ChunkingSettings(strategy="fixed", max_chars=1000, overlap_chars=100)
    chunks = chunk_sections(sections, settings)

    assert [c.chunk_id for c in chunks][:2] == ["wtr1998@0", "wtr1998@1"]
    assert all(len(c.text) <= 1000 for c in chunks)
    assert all(c.embed_text == c.text for c in chunks)  # no header: that's the point
    assert all(not c.text.startswith(" ") and not c.text.endswith(" ") for c in chunks)
    spanning = [c for c in chunks if len(c.section_keys) == 2]
    assert spanning
    assert spanning[0].section_keys == ("wtr1998#regulation-1", "wtr1998#regulation-2")


def test_fixed_windows_overlap_and_cover_the_whole_document() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    settings = ChunkingSettings(strategy="fixed", max_chars=500, overlap_chars=100)
    chunks = chunk_sections([make_section(text)], settings)

    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(text.split())
    for before, after in pairwise(chunks):
        assert after.text.split()[0] in before.text.split()  # consecutive windows overlap


def test_fixed_windows_restart_for_each_document() -> None:
    a = make_section("alpha " * 50, doc_id="era1996")
    b = make_section("beta " * 50, doc_id="wtr1998")
    chunks = chunk_sections([a, b], ChunkingSettings(strategy="fixed"))
    assert [c.chunk_id for c in chunks] == ["era1996@0", "wtr1998@0"]


# --- settings ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("max_chars", "overlap_chars"), [(200, 200), (200, 300), (0, 0)])
def test_invalid_chunking_settings_are_rejected(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings(max_chars=max_chars, overlap_chars=overlap_chars)
