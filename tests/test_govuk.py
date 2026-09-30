import json
from pathlib import Path
from typing import Any

import pytest

from lexrag.ingest.govuk import html_to_text, parse_govuk

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


# --- guides: one section per part ----------------------------------------------------------------


def test_guide_has_one_section_per_non_empty_part() -> None:
    sections = parse_govuk(load("govuk_guide.json"), "govuk-breaks")
    assert [s.key for s in sections] == ["govuk-breaks#overview", "govuk-breaks#young-workers"]


def test_guide_part_metadata_and_urls() -> None:
    overview, young = parse_govuk(load("govuk_guide.json"), "govuk-breaks")
    assert overview.url == "https://www.gov.uk/rest-breaks-work"  # first part lives at the root
    assert young.url == "https://www.gov.uk/rest-breaks-work/young-workers"
    assert young.citation == "Rest breaks at work, Young workers"
    assert young.authority == "guidance"
    assert young.extent is None


def test_guide_part_keeps_inner_headings_and_lists() -> None:
    overview = parse_govuk(load("govuk_guide.json"), "govuk-breaks")[0]
    assert overview.text.splitlines() == [
        "Workers over 18 are usually entitled to 3 types of break:",
        "- rest breaks at work",
        "- daily rest",
        "## Rest breaks at work",
        "Workers have the right to one uninterrupted 20 minute rest break during their working "
        "day, if they work more than 6 hours a day.",
    ]


# --- single-page formats: split at <h2> ----------------------------------------------------------


@pytest.fixture(scope="module")
def answer() -> dict[str, Any]:
    return {s.section_id: s for s in parse_govuk(load("govuk_answer.json"), "govuk-nmw")}


def test_answer_is_split_at_h2(answer: dict[str, Any]) -> None:
    assert list(answer) == ["intro", "current-rates", "who-gets-the-minimum-wage"]


def test_intro_keeps_loose_leading_text(answer: dict[str, Any]) -> None:
    intro = answer["intro"]
    assert intro.text.splitlines() == [
        "The hourly rate depends on your age.",
        "You must be at least school leaving age.",
    ]
    assert intro.url == "https://www.gov.uk/national-minimum-wage-rates"
    assert intro.label == "introduction"


def test_h2_section_urls_use_anchor(answer: dict[str, Any]) -> None:
    rates = answer["current-rates"]
    assert rates.title == "Current rates"
    assert rates.url == "https://www.gov.uk/national-minimum-wage-rates#current-rates"


def test_table_rows_and_text_after_comment(answer: dict[str, Any]) -> None:
    assert answer["current-rates"].text.splitlines() == [
        " | 21 and over | Under 18",
        "April 2026 | £12.71 | £8",
        "The rates change every April.",
    ]


def test_list_items_wrapping_paragraphs_keep_bullets(answer: dict[str, Any]) -> None:
    assert answer["who-gets-the-minimum-wage"].text.splitlines() == [
        "- Check your pay.",
        "- Contact Acas if it is too low.",
    ]


def test_govspeak_source_is_ignored(answer: dict[str, Any]) -> None:
    assert not any("Govspeak" in s.text for s in answer.values())


def test_document_without_text_is_an_error() -> None:
    with pytest.raises(ValueError, match="no text found"):
        parse_govuk({"title": "Empty", "details": {"body": ""}}, "empty")


# --- html_to_text edge cases ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("<p>one</p><p>two</p>", "one\ntwo"),
        ("<p>  lots   of\n spaces </p>", "lots of spaces"),
        ("<h3>Heading</h3>", "### Heading"),
        ("<ul><li>a</li><li>b</li></ul>", "- a\n- b"),
        ("<script>alert(1)</script><p>kept</p>", "kept"),
        ("<p>Contact <a href='/acas'>Acas</a> for advice</p>", "Contact Acas for advice"),
    ],
)
def test_html_to_text(html: str, expected: str) -> None:
    assert html_to_text(html) == expected
