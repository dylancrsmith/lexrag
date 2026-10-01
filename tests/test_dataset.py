import json
from pathlib import Path
from typing import Any

import pytest

from lexrag.config import Settings, list_domains
from lexrag.eval.dataset import load_questions, questions_path

REPO_ROOT = Path(__file__).parent.parent


def row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q1",
        "category": "factual",
        "question": "do i get a break?",
        "gold_sources": [["wtr1998#regulation-12", "govuk-breaks#overview"]],
        "reference_answer": "Yes.",
        "should_refuse": False,
    }
    return base | overrides


def write(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("domain", list_domains(Settings(root=REPO_ROOT)))
def test_every_shipped_test_set_is_valid(domain: str) -> None:
    path = questions_path(domain, Settings(root=REPO_ROOT))
    if path.exists():
        assert load_questions(path)


def test_valid_rows_load(tmp_path: Path) -> None:
    refusal = row(id="q2", category="out_of_scope", gold_sources=[], should_refuse=True)
    [answerable, refused] = load_questions(write(tmp_path, row(), refusal))
    assert answerable.gold_sources == [["wtr1998#regulation-12", "govuk-breaks#overview"]]
    assert answerable.verified is False
    assert refused.should_refuse


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"category": "trivia"}, "category"),
        ({"gold_sources": []}, "need some"),
        ({"gold_sources": [[]]}, "empty requirement group"),
        ({"gold_sources": ["wtr1998#regulation-12"]}, "gold_sources"),  # must be groups
        ({"category": "unanswerable"}, "should_refuse must be true"),
        ({"should_refuse": True}, "should_refuse must be true"),
        ({"verifed": True}, "Extra inputs"),
    ],
)
def test_invalid_rows_name_the_line(tmp_path: Path, overrides: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=rf"(?s)questions\.jsonl:2: .*{error}"):
        load_questions(write(tmp_path, row(id="ok"), row(**overrides)))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate question ids"):
        load_questions(write(tmp_path, row(), row()))
