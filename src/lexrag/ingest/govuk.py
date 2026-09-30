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
    tag = el.tag if isinstance(el.tag, str) else ""
    if tag in {"script", "style"}:
        return []
    if tag == "tr":
        cells = [_clean(c.text_content()) for c in el if c.tag in {"td", "th"}]
        return [" | ".join(cells)] if any(cells) else []
    if tag in _BLOCKS and not any(child.tag in _BLOCKS or child.tag in {"ul", "ol"} for child in el):
        text = _clean(el.text_content())
        if not text:
            return []
        if tag.startswith("h") and tag[1:].isdigit():
            return [f"{'#' * int(tag[1])} {text}"]
        return [f"- {text}" if tag == "li" else text]
    # Container (div, ul, table, or a block with nested blocks): keep its own leading text.
    lines = []
    if el.text and el.text.strip():
        lines.append(("- " if tag == "li" else "") + _clean(el.text))
    for child in el:
        lines.extend(_element_lines(child))
        if child.tail and child.tail.strip():
            lines.append(_clean(child.tail))
    return lines


def html_to_text(fragment: str) -> str:
    return "\n".join(line for el in _fragments(fragment) for line in _element_lines(el))


def _fragments(fragment: str) -> list[html.HtmlElement]:
    if not fragment.strip():
        return []
    return [el for el in html.fragments_fromstring(fragment) if not isinstance(el, str)]


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
    groups: list[tuple[str, str, list[html.HtmlElement]]] = [("intro", "", [])]
    for el in _fragments(fragment):
        if el.tag == "h2":
            title = _clean(el.text_content())
            groups.append((el.get("id") or _slug(title), title, []))
        else:
            groups[-1][2].append(el)
    out = []
    for gid, title, els in groups:
        text = "\n".join(line for el in els for line in _element_lines(el))
        if text:
            out.append((gid, title, text))
    return out


def parse_govuk(data: dict[str, Any], doc_id: str) -> list[Section]:
    doc_title = _clean(str(data.get("title", doc_id)))
    base_path = str(data.get("base_path", ""))
    base_url = f"https://www.gov.uk{base_path}"
    details = data.get("details") or {}

    common = {"doc_id": doc_id, "doc_title": doc_title, "authority": "guidance"}
    sections = []
    if parts := details.get("parts"):
        for i, part in enumerate(parts):
            title = _clean(part.get("title", ""))
            slug = part.get("slug") or _slug(title)
            if text := html_to_text(_body_html(part.get("body", ""))):
                sections.append(
                    Section(
                        section_id=slug,
                        label=title,
                        title=title,
                        text=text,
                        url=base_url if i == 0 else f"{base_url}/{slug}",
                        **common,
                    )
                )
    else:
        for gid, title, text in _split_h2(_body_html(details.get("body", ""))):
            sections.append(
                Section(
                    section_id=gid,
                    label=title or "introduction",
                    title=title or doc_title,
                    text=text,
                    url=base_url if gid == "intro" else f"{base_url}#{gid}",
                    **common,
                )
            )
    if not sections:
        raise ValueError(f"{doc_id}: no text found (schema {data.get('schema_name')!r})")
    return sections
