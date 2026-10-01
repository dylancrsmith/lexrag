"""Core data types passed between pipeline stages.

A `Section` is the atomic unit of source material: one provision of legislation (a section,
regulation or schedule paragraph) or one part of a GOV.UK guide. Its `key`
("<doc_id>#<section_id>") is what answers cite and what the test set's gold labels refer to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Authority = Literal["law", "guidance"]


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    """Short, stable id from domain.yaml, e.g. "era1996"."""
    section_id: str
    """Id within the document, e.g. "section-86", "regulation-12", "schedule-1-paragraph-2"."""
    label: str
    """Human-readable locator, e.g. "section 86" or a guide part title."""
    title: str
    """Heading of the provision, e.g. "Rights of employer and employee to minimum notice"."""
    path: tuple[str, ...] = ()
    """Structural breadcrumbs, outermost first, e.g. ("Part IX: Termination of employment",)."""
    text: str
    url: str
    doc_title: str
    authority: Authority
    extent: str | None = None
    """Territorial extent from legislation.gov.uk, e.g. "E+W+S". None for guidance."""
    notes: tuple[str, ...] = ()
    """Editorial annotations (amendments, commencement, extent). Kept out of the embedded text."""

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.section_id}"

    @property
    def citation(self) -> str:
        """e.g. "Employment Rights Act 1996, section 86"."""
        return f"{self.doc_title}, {self.label}" if self.label else self.doc_title


class Chunk(BaseModel):
    """A retrievable passage. Search runs over chunks; results and metrics are per section."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    """"<section key>/<n>" for structure chunks, "<doc_id>@<n>" for fixed windows."""
    section_keys: tuple[str, ...]
    """The section(s) this chunk's text comes from. A fixed window can span several."""
    text: str
    """The passage itself: what the LLM is shown."""
    embed_text: str
    """What gets embedded and keyword-indexed: `text`, plus a header for structure chunks."""
