"""Run a retriever over a test set and score it.

Only answerable questions are scored here; whether the system correctly *refuses* the others is an
answer-level metric, measured once generation exists.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from lexrag.eval.dataset import Question
from lexrag.eval.metrics import (
    complete_at_k,
    first_ranks,
    hit_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from lexrag.models import Retrieved

METRICS = ("recall", "hit", "complete", "mrr", "ndcg")


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Retrieved]: ...


class QuestionResult(BaseModel):
    id: str
    category: str
    question: str
    retrieved: list[str]
    requirement_ranks: list[int | None]
    recall: float
    hit: float
    complete: float
    mrr: float
    ndcg: float


def evaluate_retrieval(
    retriever: Retriever, questions: Sequence[Question], k: int
) -> list[QuestionResult]:
    results = []
    for q in questions:
        if q.should_refuse:
            continue
        keys = [r.section_key for r in retriever.search(q.question, k)]
        results.append(
            QuestionResult(
                id=q.id,
                category=q.category,
                question=q.question,
                retrieved=keys,
                requirement_ranks=first_ranks(keys, q.gold_sources),
                recall=recall_at_k(keys, q.gold_sources, k),
                hit=hit_at_k(keys, q.gold_sources, k),
                complete=complete_at_k(keys, q.gold_sources, k),
                mrr=reciprocal_rank(keys, q.gold_sources),
                ndcg=ndcg_at_k(keys, q.gold_sources, k),
            )
        )
    return results


def summarize(results: Sequence[QuestionResult]) -> dict[str, dict[str, float]]:
    """Mean of each metric per category, plus "all"."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
        groups["all"].append(r)
    return {
        name: {"n": len(rs)} | {m: mean([getattr(r, m) for r in rs]) for m in METRICS}
        for name, rs in sorted(groups.items(), key=lambda kv: (kv[0] == "all", kv[0]))
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_run(
    out_dir: Path,
    config: dict[str, Any],
    results: Sequence[QuestionResult],
    summary: dict[str, dict[str, float]],
) -> Path:
    """Write one run as JSON: configuration, inputs' hashes, aggregate and per-question results."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}_{config['mode']}_{config['chunking']}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": stamp,
        "config": config,
        "summary": summary,
        "questions": [r.model_dump() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
