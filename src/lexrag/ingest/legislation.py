"""Parse legislation.gov.uk CLML XML (https://www.legislation.gov.uk/<ref>/data.xml) into Sections.

One Section per top-level provision: every `P1` with an `id` (section-86, regulation-12,
schedule-1-paragraph-2) that is part of this document's own body or schedules.

Things the real XML does that this parser handles deliberately:

* Amending Acts contain `P1`s inside `BlockAmendment` (the text being inserted into *another*
  Act). Those have no id and belong to the enclosing provision's text, not to this document.
* A provision with different wording per territorial extent (e.g. ERA 1996 s.236 for N.I.) keeps
  its alternative wording in a separate `Versions` block. Only the main body is parsed, so each
  id appears once; the URI of such a provision carries an extent suffix, so labels come from ids.
* Repealed provisions survive as an empty shell with a dotted title; they are dropped.
* Amendment markup (`Substitution`, `Addition`) wraps current text and is kept; editorial
  commentaries are removed from the text but kept as `Section.notes`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from lxml import etree

from lexrag.models import Section

NS = "http://www.legislation.gov.uk/namespaces/legislation"
DC = "http://purl.org/dc/elements/1.1/"
_L = f"{{{NS}}}"

# Containers that contribute a breadcrumb to `Section.path`, outermost first.
_STRUCTURAL = {"Part", "Chapter", "Pblock", "PsubBlock", "Schedule"}
# Elements whose content is never part of the provision text.
_SKIP = {
    "Pnumber", "CommentaryRef", "FootnoteRef", "MarginNoteRef", "Footnote", "MarginNote",
    "Figure", "Image", "IncludedDocument", "Reference",
}  # fmt: skip
_NUMBERED = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P"}
# Commentary types worth keeping as notes: F = textual amendment, I = commencement,
# E = extent, C = modification. (M = marginal citation is noise.)
_NOTE_TYPES = {"F", "I", "E", "C"}
_WS = re.compile(r"\s+")
# legislation.gov.uk replaces repealed, revoked or omitted text with a spaced run of dots
# (". . . ."), ~32 tokens of nothing. Unspaced "..." is an in-sentence omission and is kept.
_REMOVED_TEXT = re.compile(r"\.(?: \.)+(?: ?\.)*")
NO_LONGER_IN_FORCE = "[no longer in force]"


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _inline(el: etree._Element) -> str:
    """Text content of `el`, skipping editorial markers."""
    parts = [el.text or ""]
    for child in el:  # all nodes, not just elements: a comment's tail is still provision text
        if isinstance(child.tag, str) and _local(child) not in _SKIP:
            parts.append(_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _render(el: etree._Element, depth: int = 0) -> list[tuple[int, str]]:
    """Render a provision body as (indent-level, line) pairs, preserving (1)/(a)/(i) structure."""
    lines: list[tuple[int, str]] = []
    for child in el.iterchildren(etree.Element):  # elements only: skips comments and PIs
        name = _local(child)
        if name in _SKIP:
            continue
        if name in {"Text", "Title", "Number"}:
            if s := _REMOVED_TEXT.sub(NO_LONGER_IN_FORCE, _clean(_inline(child))):
                lines.append((depth, s))
        elif name in _NUMBERED:
            num = child.find(f"{_L}Pnumber")
            label = _clean(_inline(num)) if num is not None else ""
            sub = _render(child, depth + 1)
            if label and name != "P1":
                label = f"({label})"
            if label and sub:
                sub[0] = (depth, f"{label} {sub[0][1]}")
            elif label:
                sub = [(depth, label)]
            lines.extend(sub)
        elif name == "ListItem":
            sub = _render(child, depth + 1)
            if sub:
                sub[0] = (depth, f"- {sub[0][1]}")
            lines.extend(sub)
        elif name == "tr":
            cells = [_clean(_inline(c)) for c in child.iterchildren(etree.Element)]
            if any(cells):
                lines.append((depth, " | ".join(cells)))
        else:  # P1para, P2para, BlockAmendment, lists, tables, Para, ...: descend
            lines.extend(_render(child, depth))
    return lines


def _text(el: etree._Element) -> str:
    return "\n".join("  " * d + s for d, s in _render(el))


def _title(el: etree._Element | None) -> str:
    if el is None:
        return ""
    t = el.find(f"{_L}Title")
    if t is None:
        t = el.find(f"{_L}TitleBlock/{_L}Title")
    return _clean(_inline(t)) if t is not None else ""


def _breadcrumbs(p1: etree._Element) -> tuple[str, ...]:
    crumbs = []
    for anc in p1.iterancestors(*(f"{_L}{t}" for t in _STRUCTURAL)):
        num = anc.find(f"{_L}Number")
        parts = [_clean(_inline(num)) if num is not None else "", _title(anc)]
        if crumb := ": ".join(p for p in parts if p):
            crumbs.append(crumb)
    return tuple(reversed(crumbs))


def _extent(el: etree._Element) -> str | None:
    for node in (el, *el.iterancestors()):
        if ext := node.get("RestrictExtent"):
            return ext
    return None


def _label(section_id: str) -> str:
    """Human-readable locator: section-86 -> "section 86", schedule-1-paragraph-2 -> ..."""
    return section_id.replace("-", " ")


def _notes(scope: etree._Element, commentaries: dict[str, tuple[str, str]]) -> tuple[str, ...]:
    refs: list[str] = []
    for el in scope.iter(etree.Element):
        ref = el.get("Ref") if _local(el) == "CommentaryRef" else el.get("CommentaryRef")
        if ref and ref not in refs:
            refs.append(ref)
    out = []
    for ref in refs:
        if ref in commentaries:
            kind, text = commentaries[ref]
            if kind in _NOTE_TYPES and text:
                out.append(f"[{kind}] {text}")
    return tuple(out)


def within(section_id: str, ancestor: str) -> bool:
    """True if `section_id` is `ancestor` or a sub-provision of it (schedule-3-paragraph-2)."""
    return section_id == ancestor or section_id.startswith(f"{ancestor}-")


def _is_repealed(title: str, text: str) -> bool:
    # Repealed provisions keep only their number and a dotted-out title.
    words = re.compile(r"[A-Za-z]{3}")
    return not words.search(text.replace(NO_LONGER_IN_FORCE, "")) and not words.search(title)


def _provisions(doc: etree._Element) -> Iterator[etree._Element]:
    for p1 in doc.iter(f"{_L}P1"):
        in_amendment = next(p1.iterancestors(f"{_L}BlockAmendment"), None) is not None
        if p1.get("id") and not in_amendment:
            yield p1


def parse_clml(xml: bytes, doc_id: str, exclude: tuple[str, ...] = ()) -> list[Section]:
    """Parse one CLML document, dropping provisions `within` any id in `exclude`."""
    root = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    title_el = root.find(f".//{{{DC}}}title")
    doc_title = _clean(title_el.text or "") if title_el is not None else doc_id
    doc = root.find(f"{_L}Primary")
    if doc is None:
        doc = root.find(f"{_L}Secondary")
    if doc is None:
        raise ValueError(f"{doc_id}: no Primary/Secondary element; not a CLML document?")

    commentaries = {
        c.get("id", ""): (c.get("Type", ""), _clean(_inline(c)))
        for c in root.iter(f"{_L}Commentary")
    }

    sections: list[Section] = []
    seen: set[str] = set()
    for p1 in _provisions(doc):
        parent = p1.getparent()
        group = parent if parent is not None and _local(parent) == "P1group" else None
        title = _title(group) or _title(p1)
        text = _text(p1)
        if _is_repealed(title, text):
            continue
        section_id = p1.get("id", "")
        if section_id in seen or any(within(section_id, x) for x in exclude):
            continue
        seen.add(section_id)
        sections.append(
            Section(
                doc_id=doc_id,
                section_id=section_id,
                label=_label(section_id),
                title=title,
                path=_breadcrumbs(p1),
                text=text,
                url=(p1.get("DocumentURI") or "").replace("http://", "https://", 1),
                doc_title=doc_title,
                authority="law",
                extent=_extent(p1),
                notes=_notes(group if group is not None else p1, commentaries),
            )
        )
    return sections
