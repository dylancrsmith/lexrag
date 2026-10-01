"""Retrieval metrics over requirement groups (see eval/dataset.py).

`retrieved` is a ranked list of section keys; `gold` is a list of requirements, each satisfied by
any one of its keys. A retrieved key matches a gold key if it is the same section, a sub-provision
of it, or the gold key names a whole document ("era1996").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from lexrag.ingest.legislation import within

Gold = Sequence[Sequence[str]]


def matches(retrieved: str, gold: str) -> bool:
    if "#" not in gold:  # document-level label
        return retrieved.split("#", 1)[0] == gold
    doc, _, section = retrieved.partition("#")
    gold_doc, _, gold_section = gold.partition("#")
    return doc == gold_doc and within(section, gold_section)


def first_ranks(retrieved: Sequence[str], gold: Gold) -> list[int | None]:
    """For each requirement, the 1-based rank at which it is first satisfied (None if never)."""
    ranks: list[int | None] = []
    for group in gold:
        rank = next(
            (i for i, key in enumerate(retrieved, 1) if any(matches(key, g) for g in group)),
            None,
        )
        ranks.append(rank)
    return ranks


def _met(retrieved: Sequence[str], gold: Gold, k: int) -> list[bool]:
    return [r is not None and r <= k for r in first_ranks(retrieved, gold)]


def recall_at_k(retrieved: Sequence[str], gold: Gold, k: int) -> float:
    """Fraction of requirements satisfied in the top k."""
    met = _met(retrieved, gold, k)
    return sum(met) / len(met) if met else 0.0


def hit_at_k(retrieved: Sequence[str], gold: Gold, k: int) -> float:
    """1 if at least one requirement is satisfied in the top k."""
    return float(any(_met(retrieved, gold, k)))


def complete_at_k(retrieved: Sequence[str], gold: Gold, k: int) -> float:
    """1 if every requirement is satisfied in the top k: what a multi-hop answer actually needs."""
    met = _met(retrieved, gold, k)
    return float(bool(met) and all(met))


def reciprocal_rank(retrieved: Sequence[str], gold: Gold) -> float:
    """1 / rank of the first result meeting any requirement (0 if none); averaged, this is MRR."""
    ranks = [r for r in first_ranks(retrieved, gold) if r is not None]
    return 1 / min(ranks) if ranks else 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Gold, k: int) -> float:
    """Binary-gain nDCG: each requirement earns 1/log2(rank+1) where it is first satisfied.

    The ideal ranking satisfies one requirement per rank. A single section that satisfies two
    requirements at once can beat that, so the score is capped at 1.
    """
    ranks = [r for r in first_ranks(retrieved, gold) if r is not None and r <= k]
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return min(dcg / ideal, 1.0) if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan
