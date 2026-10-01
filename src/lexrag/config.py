"""Configuration: engine settings (config/settings.yaml) and per-domain definitions (domains/*/).

The engine is domain-agnostic. Everything specific to an area of law — which documents to ingest,
who the audience is, what is in or out of scope — lives in domains/<name>/domain.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT_ENV_VAR = "LEXRAG_ROOT"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- domain definition -------------------------------------------------------------------------


class LegislationSource(_Strict):
    type: Literal["legislation"]
    id: str
    ref: Annotated[str, Field(pattern=r"^[a-z]+/\d{4}/\d+$")]
    """legislation.gov.uk type/year/number, e.g. "ukpga/1996/18"."""
    exclude: tuple[str, ...] = ()
    """Provision ids to drop with their sub-provisions, e.g. ("schedule-3",) for a repeals table."""

    @property
    def url(self) -> str:
        return f"https://www.legislation.gov.uk/{self.ref}/data.xml"


class GovukSource(_Strict):
    type: Literal["govuk"]
    id: str
    path: Annotated[str, Field(pattern=r"^/[a-z0-9-/]+$")]
    """GOV.UK base path, e.g. "/rest-breaks-work"."""

    @property
    def url(self) -> str:
        return f"https://www.gov.uk/api/content{self.path}"


Source = Annotated[LegislationSource | GovukSource, Field(discriminator="type")]


class Domain(_Strict):
    name: str
    title: str
    description: str
    jurisdiction: str
    audience: str
    in_scope: list[str]
    out_of_scope: list[str]
    disclaimer: str
    sources: list[Source]

    @model_validator(mode="after")
    def _unique_source_ids(self) -> Domain:
        ids = [s.id for s in self.sources]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate source ids: {dupes}")
        return self


# --- engine settings ---------------------------------------------------------------------------


class ChunkingSettings(_Strict):
    strategy: Literal["structure", "fixed"] = "structure"
    # bge-small reads 512 tokens and silently drops the rest. Measured on this corpus:
    # ~4.5 chars/token and a <=41-token header for 95% of sections, so 1800 chars fits with margin.
    max_chars: Annotated[int, Field(gt=0)] = 1800
    overlap_chars: Annotated[int, Field(ge=0)] = 200

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> ChunkingSettings:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self


class Settings(_Strict):
    root: Path = Field(default_factory=lambda: Path(os.environ.get(ROOT_ENV_VAR, ".")).resolve())
    data_dir: Path = Path("data")
    domains_dir: Path = Path("domains")
    results_dir: Path = Path("results")
    chunking: ChunkingSettings = ChunkingSettings()
    user_agent: str = "lexrag/0.1 (+https://github.com/dylancrsmith/lexrag)"

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    def domain_file(self, domain: str) -> Path:
        return self.resolve(self.domains_dir) / domain / "domain.yaml"

    def data_path(self, domain: str, *parts: str) -> Path:
        """Path under data/<domain>/, e.g. data_path("hospitality", "raw")."""
        return self.resolve(self.data_dir).joinpath(domain, *parts)


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from `path` (default: <root>/config/settings.yaml, if it exists)."""
    root = Path(os.environ.get(ROOT_ENV_VAR, ".")).resolve()
    path = path or root / "config" / "settings.yaml"
    data = _read_yaml(path) if path.exists() else {}
    return Settings(root=root, **data)  # type: ignore[arg-type]


def load_domain(name: str, settings: Settings) -> Domain:
    path = settings.domain_file(name)
    if not path.exists():
        available = ", ".join(list_domains(settings)) or "none"
        raise FileNotFoundError(f"no domain {name!r} at {path} (available: {available})")
    domain = Domain.model_validate(_read_yaml(path))
    if domain.name != name:
        raise ValueError(f"{path}: name {domain.name!r} does not match its folder {name!r}")
    return domain


def list_domains(settings: Settings) -> list[str]:
    base = settings.resolve(settings.domains_dir)
    return sorted(p.parent.name for p in base.glob("*/domain.yaml"))
