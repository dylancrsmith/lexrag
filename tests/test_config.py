from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from lexrag.config import (
    GovukSource,
    LegislationSource,
    Settings,
    list_domains,
    load_domain,
    load_settings,
)

REPO_ROOT = Path(__file__).parent.parent


def minimal_domain(**overrides: Any) -> dict[str, Any]:
    domain: dict[str, Any] = {
        "name": "demo",
        "title": "Demo",
        "description": "A test domain.",
        "jurisdiction": "England",
        "audience": "Testers",
        "in_scope": ["testing"],
        "out_of_scope": ["everything else"],
        "disclaimer": "Not legal advice.",
        "sources": [
            {"id": "wtr1998", "type": "legislation", "ref": "uksi/1998/1833"},
            {"id": "govuk-breaks", "type": "govuk", "path": "/rest-breaks-work"},
        ],
    }
    domain.update(overrides)
    return domain


def write_domain(root: Path, folder: str, data: dict[str, Any]) -> Settings:
    path = root / "domains" / folder / "domain.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Settings(root=root)


# --- the real domains in this repo ---------------------------------------------------------------


@pytest.mark.parametrize("name", list_domains(Settings(root=REPO_ROOT)))
def test_every_shipped_domain_is_valid(name: str) -> None:
    domain = load_domain(name, Settings(root=REPO_ROOT))
    assert domain.sources


def test_hospitality_is_shipped() -> None:
    assert "hospitality" in list_domains(Settings(root=REPO_ROOT))


# --- validation ----------------------------------------------------------------------------------


def test_minimal_domain_loads(tmp_path: Path) -> None:
    settings = write_domain(tmp_path, "demo", minimal_domain())
    domain = load_domain("demo", settings)
    leg, guide = domain.sources
    assert isinstance(leg, LegislationSource)
    assert leg.url == "https://www.legislation.gov.uk/uksi/1998/1833/data.xml"
    assert leg.exclude == ()
    assert isinstance(guide, GovukSource)
    assert guide.url == "https://www.gov.uk/api/content/rest-breaks-work"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"sources": [{"id": "a", "type": "legislation", "ref": "ukpga/1996/18"}] * 2},
            "duplicate source ids",
        ),
        (
            {"sources": [{"id": "a", "type": "legislation", "ref": "Employment Rights Act"}]},
            "should match pattern",
        ),
        (
            {"sources": [{"id": "a", "type": "govuk", "path": "rest-breaks-work"}]},
            "should match pattern",
        ),
        (
            {"sources": [{"id": "a", "type": "pdf", "url": "x"}]},
            "does not match any of the expected tags",
        ),
        (  # typo in an optional field must not be silently ignored
            {"sources": [{"id": "a", "type": "legislation", "ref": "ukpga/1996/18", "exlude": []}]},
            "Extra inputs are not permitted",
        ),
        ({"disclaimer": None}, "disclaimer"),
    ],
)
def test_invalid_domains_are_rejected(
    tmp_path: Path, overrides: dict[str, Any], error: str
) -> None:
    settings = write_domain(tmp_path, "demo", minimal_domain(**overrides))
    with pytest.raises(ValidationError, match=error):
        load_domain("demo", settings)


def test_name_must_match_folder(tmp_path: Path) -> None:
    settings = write_domain(tmp_path, "other", minimal_domain())
    with pytest.raises(ValueError, match="does not match its folder"):
        load_domain("other", settings)


def test_missing_domain_lists_available(tmp_path: Path) -> None:
    settings = write_domain(tmp_path, "demo", minimal_domain())
    with pytest.raises(FileNotFoundError, match=r"available: demo"):
        load_domain("nope", settings)


# --- settings ------------------------------------------------------------------------------------


def test_settings_default_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXRAG_ROOT", str(tmp_path))
    settings = load_settings()
    assert settings.root == tmp_path.resolve()
    assert settings.data_path("demo", "raw") == tmp_path.resolve() / "data" / "demo" / "raw"


def test_settings_file_overrides_and_rejects_unknown_keys(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"  # absolute on every OS, unlike "/elsewhere" on Windows
    path = tmp_path / "settings.yaml"
    path.write_text(f"data_dir: {elsewhere.as_posix()}\n", encoding="utf-8")
    assert load_settings(path).data_path("demo") == elsewhere / "demo"

    path.write_text("dataa_dir: typo\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_settings(path)
