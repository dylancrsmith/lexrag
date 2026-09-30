"""Parse GOV.UK Content API responses (https://www.gov.uk/api/content/<path>) into Sections.

* Guides (`details.parts`) become one Section per part, which is how GOV.UK itself pages them.
* Single-page formats (`details.body`) are split at `<h2>` headings, so an answer page covering
  several topics does not become one oversized, unfocused Section.
"""

from __future__ import annotations

import re
from typing import Any

from lxml import html

from lexrag.models import Section

_WS = re.compile(r"\s+")
_BLOCKS = {"p", "h2", "h3", "h4", "h5", "h6", "li", "tr", "dt", "dd", "blockquote", "pre"}


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _element_lines(el: html.HtmlElement) -> list[str]:
    """Render one HTML element as plain-text lines: headings prefixed "#", list items "-"."""
    tag = el.tag
    if tag in {"script", "style"}:
        return []
    if tag == "tr":
        cells = [_clean(c.text_content()) for c in el.iterchildren("td", "th")]
        return [" | ".join(cells)] if any(cells) else []
    if tag in _BLOCKS and next(el.iterchildren(*_BLOCKS, "ul", "ol"), None) is None:
        text = _clean(el.text_content())
        if not text:
            return []
        if tag.startswith("h") and tag[1:].isdigit():
            return [f"{'#' * int(tag[1])} {text}"]
        return [f"- {text}" if tag == "li" else text]
    # Container (div, ul, table, or a block with nested blocks).
    lines = _children_lines(el)
    if tag == "li" and lines:
        lines[0] = f"- {lines[0]}"
    return lines


def _children_lines(el: html.HtmlElement) -> list[str]:
    """Lines for everything inside `el`: its own text, child elements, and text between them."""
    lines = [_clean(el.text)] if el.text and el.text.strip() else []
    for child in el:  # all nodes: a comment's content is skipped but its tail is real text
        if isinstance(child.tag, str):
            lines.extend(_element_lines(child))
        if child.tail and child.tail.strip():
            lines.append(_clean(child.tail))
    return lines


def _root(fragment: str) -> html.HtmlElement | None:
    """Parse an HTML fragment inside a wrapper <div>, so leading loose text is kept as its .text."""
    if not fragment.strip():
        return None
    return html.fragment_fromstring(fragment, create_parent="div")


def html_to_text(fragment: str) -> str:
    root = _root(fragment)
    return "\n".join(_children_lines(root)) if root is not None else ""


def _body_html(body: Any) -> str:
    """`details.body` is either an HTML string or a list of {content_type, content}."""
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        for item in body:
            if isinstance(item, dict) and item.get("content_type") == "text/html":
                return str(item.get("content", ""))
    return ""


def _split_h2(fragment: str) -> list[tuple[str, str, str]]:
    """Split HTML at top-level <h2>s into (id, title, text); content before the first is "intro"."""
    root = _root(fragment)
    if root is None:
        return []
    groups: list[tuple[str, str, list[str]]] = [("intro", "", [])]
    if root.text and root.text.strip():
        groups[0][2].append(_clean(root.text))
    for child in root:
        if child.tag == "h2":
            title = _clean(child.text_content())
            groups.append((child.get("id") or _slug(title), title, []))
        elif isinstance(child.tag, str):
            groups[-1][2].extend(_element_lines(child))
        if child.tail and child.tail.strip():
            groups[-1][2].append(_clean(child.tail))
    return [(gid, title, "\n".join(lines)) for gid, title, lines in groups if lines]


def parse_govuk(data: dict[str, Any], doc_id: str) -> list[Section]:
    doc_title = _clean(str(data.get("title", doc_id)))
    base_url = f"https://www.gov.uk{data.get('base_path', '')}"
    details = data.get("details") or {}

    def section(section_id: str, label: str, title: str, text: str, url: str) -> Section:
        return Section(
            doc_id=doc_id,
            section_id=section_id,
            label=label,
            title=title,
            text=text,
            url=url,
            doc_title=doc_title,
            authority="guidance",
        )

    sections = []
    if parts := details.get("parts"):
        for i, part in enumerate(parts):
            title = _clean(part.get("title", ""))
            slug = part.get("slug") or _slug(title)
            if text := html_to_text(_body_html(part.get("body", ""))):
                url = base_url if i == 0 else f"{base_url}/{slug}"
                sections.append(section(slug, title, title, text, url))
    else:
        for gid, title, text in _split_h2(_body_html(details.get("body", ""))):
            url = base_url if gid == "intro" else f"{base_url}#{gid}"
            sections.append(section(gid, title or "introduction", title or doc_title, text, url))
    if not sections:
        raise ValueError(f"{doc_id}: no text found (schema {data.get('schema_name')!r})")
    return sections
