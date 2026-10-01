"""Split sections into retrievable chunks.

Two strategies, compared in the ablation table:

* "structure" (default): one chunk per section. A section longer than `max_chars` is split at
  line boundaries, i.e. between numbered provisions such as (1), (2), (a), never mid-sentence,
  and every piece keeps a header naming the Act, Part and section it came from.
* "fixed": the usual naive baseline. Each document's text is cut into overlapping windows of
  `max_chars`, ignoring section boundaries, with no header.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from itertools import groupby

from lexrag.config import ChunkingSettings
from lexrag.models import Chunk, Section

_SENTENCE_END = re.compile(r"(?<=[.;:—])\s+")


def chunk_sections(sections: Iterable[Section], settings: ChunkingSettings) -> list[Chunk]:
    if settings.strategy == "structure":
        return [
            chunk
            for section in sections
            for chunk in _structure_chunks(section, settings.max_chars, settings.overlap_chars)
        ]
    return [
        chunk
        for _, doc_sections in groupby(sections, key=lambda s: s.doc_id)
        for chunk in _fixed_chunks(list(doc_sections), settings.max_chars, settings.overlap_chars)
    ]


# --- structure-aware -----------------------------------------------------------------------------


def header(section: Section) -> str:
    """e.g. "The Working Time Regulations 1998 > PART II: ... > regulation 12: Rest breaks"."""
    locator = f"{section.label}: {section.title}" if section.title else section.label
    return " > ".join((section.doc_title, *section.path, locator))


def _structure_chunks(section: Section, max_chars: int, overlap_chars: int) -> list[Chunk]:
    head = header(section)
    return [
        Chunk(
            chunk_id=f"{section.key}/{i}",
            section_keys=(section.key,),
            text=piece,
            embed_text=f"{head}\n{piece}",
        )
        for i, piece in enumerate(split_lines(section.text, max_chars, overlap_chars))
    ]


def split_lines(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Pack whole lines into pieces of at most `max_chars`.

    Each new piece repeats the previous piece's last lines, up to `overlap_chars`, so a
    provision's lead-in ("(1) The notice required ... is—") stays next to its sub-paragraphs.
    """
    if len(text) <= max_chars:
        return [text]
    lines = [part for line in text.split("\n") for part in _fit(line, max_chars)]
    pieces: list[list[str]] = [[]]
    for line in lines:
        current = pieces[-1]
        if current and _length([*current, line]) > max_chars:
            carried = _tail(current, overlap_chars)
            while carried and _length([*carried, line]) > max_chars:
                carried.pop(0)
            pieces.append([*carried, line])
        else:
            current.append(line)
    return ["\n".join(piece) for piece in pieces]


def _length(lines: list[str]) -> int:
    return sum(map(len, lines)) + len(lines) - 1


def _tail(lines: list[str], budget: int) -> list[str]:
    """The longest run of whole lines from the end of `lines` that fits in `budget`."""
    tail: list[str] = []
    for line in reversed(lines):
        if _length([line, *tail]) > budget:
            break
        tail.insert(0, line)
    return tail


def _fit(line: str, max_chars: int) -> Iterator[str]:
    """Break a single over-long line at sentence ends, then at spaces as a last resort."""
    if len(line) <= max_chars:
        yield line
        return
    indent = line[: len(line) - len(line.lstrip())]
    buffer = ""
    for sentence in _SENTENCE_END.split(line.strip()):
        for word_run in _wrap(sentence, max_chars - len(indent)):
            candidate = f"{buffer} {word_run}" if buffer else word_run
            if len(indent) + len(candidate) <= max_chars:
                buffer = candidate
            else:
                yield indent + buffer
                buffer = word_run
    if buffer:
        yield indent + buffer


def _wrap(text: str, width: int) -> Iterator[str]:
    """Split text at spaces into runs of at most `width` (a word longer than that is cut)."""
    run = ""
    for word in text.split(" "):
        while len(word) > width:
            if run:
                yield run
                run = ""
            yield word[:width]
            word = word[width:]
        if run and len(run) + 1 + len(word) > width:
            yield run
            run = word
        else:
            run = f"{run} {word}" if run else word
    if run:
        yield run


# --- fixed-size baseline -------------------------------------------------------------------------


def _fixed_chunks(sections: list[Section], max_chars: int, overlap_chars: int) -> list[Chunk]:
    text = ""
    spans: list[tuple[int, int, str]] = []  # (start, end, section key) within `text`
    for section in sections:
        if text:
            text += "\n"
        spans.append((len(text), len(text) + len(section.text), section.key))
        text += section.text

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):  # end on a word boundary, as common text splitters do
            space = text.rfind(" ", start + max_chars // 2, end)
            end = space if space != -1 else end
        window = text[start:end].strip()
        if window:
            keys = tuple(key for s, e, key in spans if s < end and e > start)
            doc_id = sections[0].doc_id
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}@{len(chunks)}",
                    section_keys=keys,
                    text=window,
                    embed_text=window,
                )
            )
        if end == len(text):
            break
        # Step back by the overlap, then forward to a word start, so no window opens mid-word.
        next_start = end - overlap_chars
        space = text.find(" ", next_start, end)
        start = max(space + 1 if space != -1 else next_start, start + 1)
    return chunks
