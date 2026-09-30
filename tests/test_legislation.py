from pathlib import Path

import pytest

from lexrag.ingest.legislation import parse_clml, within
from lexrag.models import Section

FIXTURE = Path(__file__).parent / "fixtures" / "clml_sample.xml"


@pytest.fixture(scope="module")
def sections() -> dict[str, Section]:
    return {s.section_id: s for s in parse_clml(FIXTURE.read_bytes(), "era1996")}


def test_one_section_per_numbered_provision(sections: dict[str, Section]) -> None:
    assert list(sections) == [
        "section-86",
        "section-200",
        "section-236",
        "schedule-1-paragraph-1",
    ]


def test_metadata(sections: dict[str, Section]) -> None:
    s = sections["section-86"]
    assert s.key == "era1996#section-86"
    assert s.doc_title == "Employment Rights Act 1996"
    assert s.citation == "Employment Rights Act 1996, section 86"
    assert s.title == "Rights of employer and employee to minimum notice."
    assert s.url == "https://www.legislation.gov.uk/ukpga/1996/18/section/86"
    assert s.authority == "law"
    assert s.extent == "E+W+S"


def test_breadcrumbs_come_from_enclosing_structure(sections: dict[str, Section]) -> None:
    assert sections["section-86"].path == (
        "Part IX: Termination of employment",
        "Minimum period of notice",
    )
    assert sections["schedule-1-paragraph-1"].path == ("SCHEDULE 1: Workforce agreements",)


def test_text_keeps_numbered_structure_and_amended_wording(sections: dict[str, Section]) -> None:
    lines = sections["section-86"].text.splitlines()
    assert lines[0].startswith("(1) The notice required")
    assert lines[1].startswith("  (a) is not less than one week’s notice")
    assert lines[3].startswith("(2) The notice required to be given by an employee")


def test_xml_comment_is_skipped_but_following_text_kept(sections: dict[str, Section]) -> None:
    assert sections["section-86"].text.endswith("is not less than one week.")


def test_notes_keep_amendments_but_not_marginal_citations(sections: dict[str, Section]) -> None:
    assert sections["section-86"].notes == (
        "[F] Word in s. 86(2) substituted (1.10.2002) by S.I. 2002/2034, reg. 11.",
    )


def test_inserted_text_stays_inside_the_amending_section(sections: dict[str, Section]) -> None:
    text = sections["section-200"].text
    assert "27D (1) An employer must ensure" in text
    assert not any(key.startswith("section-27") for key in sections)


def test_alternative_extent_versions_are_ignored(sections: dict[str, Section]) -> None:
    s = sections["section-236"]
    assert "NORTHERN IRELAND" not in s.text
    assert s.label == "section 236"  # not "section 236 england+wales+scotland" from the URI


def test_schedule_paragraph_without_title_and_with_table(sections: dict[str, Section]) -> None:
    s = sections["schedule-1-paragraph-1"]
    assert s.title == ""
    assert s.label == "schedule 1 paragraph 1"
    assert s.extent == "E+W+S+N.I."  # inherited from the document root
    assert "Condition | Requirement\nForm | in writing" in s.text


def test_exclude_drops_provision_and_its_sub_provisions() -> None:
    parsed = parse_clml(FIXTURE.read_bytes(), "era1996", exclude=("schedule-1",))
    assert "schedule-1-paragraph-1" not in {s.section_id for s in parsed}


@pytest.mark.parametrize(
    ("section_id", "ancestor", "expected"),
    [
        ("schedule-3", "schedule-3", True),
        ("schedule-3-paragraph-2", "schedule-3", True),
        ("schedule-30-paragraph-1", "schedule-3", False),  # prefix match must respect "-"
        ("section-27", "section-27C", False),
    ],
)
def test_within(section_id: str, ancestor: str, expected: bool) -> None:
    assert within(section_id, ancestor) is expected


def test_rejects_non_clml_documents() -> None:
    with pytest.raises(ValueError, match="not a CLML document"):
        parse_clml(b"<html><body>Page not found</body></html>", "bogus")


def test_repealed_provisions_are_dropped(sections: dict[str, Section]) -> None:
    assert "section-96" not in sections
