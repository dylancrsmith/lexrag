"""Ingest a domain: download (or reuse cached) sources, parse them, write sections.jsonl.

Raw downloads are cached under data/<domain>/raw/ and recorded in a manifest with their SHA-256,
so every evaluation run can be tied to the exact snapshot of the law it was run against.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from lexrag.config import Domain, GovukSource, LegislationSource, Settings, Source
from lexrag.ingest.govuk import parse_govuk
from lexrag.ingest.legislation import parse_clml
from lexrag.models import Section

_RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class SourceReport:
    source_id: str
    sections: int
    chars: int
    cached: bool


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _download(client: httpx.Client, url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        resp = client.get(url)
        if resp.status_code not in _RETRY_STATUS or attempt == attempts - 1:
            resp.raise_for_status()
            return resp.content
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _raw_name(source: Source) -> str:
    return f"{source.id}.xml" if isinstance(source, LegislationSource) else f"{source.id}.json"


def parse_source(source: Source, raw: bytes) -> list[Section]:
    if isinstance(source, LegislationSource):
        return parse_clml(raw, source.id, exclude=source.exclude)
    if isinstance(source, GovukSource):
        return parse_govuk(json.loads(raw), source.id)
    raise TypeError(f"unknown source type: {type(source).__name__}")


def ingest(
    domain: Domain, settings: Settings, *, refresh: bool = False, client: httpx.Client | None = None
) -> list[SourceReport]:
    raw_dir = settings.data_path(domain.name, "raw")
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": settings.user_agent},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
        transport=httpx.HTTPTransport(retries=2),
    )
    sections: list[Section] = []
    reports: list[SourceReport] = []
    try:
        for source in domain.sources:
            path = raw_dir / _raw_name(source)
            cached = path.exists() and not refresh
            if cached:
                raw = path.read_bytes()
            else:
                raw = _download(client, source.url)
                _write_atomic(path, raw)
                manifest[source.id] = {
                    "url": source.url,
                    "file": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                # Save now, not at the end: if a later source fails, the manifest must still
                # describe every raw file already on disk.
                _write_atomic(
                    manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode()
                )
            parsed = parse_source(source, raw)
            sections.extend(parsed)
            reports.append(
                SourceReport(source.id, len(parsed), sum(len(s.text) for s in parsed), cached)
            )
    finally:
        if own_client:
            client.close()

    _check_unique(sections)
    lines = (s.model_dump_json() for s in sections)
    _write_atomic(sections_path(domain.name, settings), ("\n".join(lines) + "\n").encode())
    return reports


def _check_unique(sections: list[Section]) -> None:
    seen: set[str] = set()
    for s in sections:
        if s.key in seen:
            raise ValueError(f"duplicate section key {s.key}")
        seen.add(s.key)


def sections_path(domain: str, settings: Settings) -> Path:
    return settings.data_path(domain, "sections.jsonl")


def load_sections(domain: str, settings: Settings) -> list[Section]:
    path = sections_path(domain, settings)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `lexrag ingest {domain}` first")
    with path.open(encoding="utf-8") as f:
        return [Section.model_validate_json(line) for line in f if line.strip()]
