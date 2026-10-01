"""The evaluation test set: domains/<name>/eval/questions.jsonl, one question per line.

Gold labels are *requirement groups*: a list of requirements, each satisfied by any one of its
sections. `[["era1996#section-8", "govuk-payslips#intro"]]` is one requirement that either the Act
or the guidance meets; `[["wtr1998#regulation-12"], ["wtr1998#regulation-10"]]` is a multi-hop
question needing both. Questions that should be refused have no requirements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from lexrag.config import Settings

Category = Literal["factual", "multi_hop", "unanswerable", "out_of_scope", "false_premise"]
REFUSAL_CATEGORIES = {"unanswerable", "out_of_scope"}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: Category
    question: str
    gold_sources: list[list[str]]
    reference_answer: str
    should_refuse: bool
    verified: bool = False
    """Set only once a person has checked the gold labels and answer against the source text."""

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        if self.should_refuse != (self.category in REFUSAL_CATEGORIES):
            allowed = ", ".join(sorted(REFUSAL_CATEGORIES))
            raise ValueError(f"{self.id}: should_refuse must be true exactly for {allowed}")
        if self.should_refuse == bool(self.gold_sources):
            raise ValueError(f"{self.id}: refusals have no gold sources; answerable ones need some")
        if any(not group for group in self.gold_sources):
            raise ValueError(f"{self.id}: empty requirement group")
        return self


def questions_path(domain: str, settings: Settings) -> Path:
    return settings.resolve(settings.domains_dir) / domain / "eval" / "questions.jsonl"


def load_questions(path: Path) -> list[Question]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    questions.append(Question.model_validate(json.loads(line)))
                except ValueError as e:  # pydantic's ValidationError and JSONDecodeError
                    raise ValueError(f"{path}:{line_no}: {e}") from e
    ids = [q.id for q in questions]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"{path}: duplicate question ids {dupes}")
    return questions
