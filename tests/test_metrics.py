import math

import pytest

from lexrag.eval.metrics import (
    complete_at_k,
    first_ranks,
    hit_at_k,
    matches,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Two requirements: rest breaks (Act or guidance), and daily rest.
GOLD = [["wtr#reg-12", "govuk#breaks"], ["wtr#reg-10"]]


@pytest.mark.parametrize(
    ("retrieved", "gold", "expected"),
    [
        ("wtr#reg-12", "wtr#reg-12", True),
        ("wtr#reg-12-3", "wtr#reg-12", True),  # sub-provision of the gold section
        ("wtr#reg-12", "wtr#reg-12-3", False),  # the whole is not the part
        ("wtr#reg-120", "wtr#reg-12", False),  # prefix but a different regulation
        ("era#reg-12", "wtr#reg-12", False),  # same id, different document
        ("wtr#reg-12", "wtr", True),  # document-level gold label
        ("wtr2#reg-12", "wtr", False),
    ],
)
def test_matches(retrieved: str, gold: str, expected: bool) -> None:
    assert matches(retrieved, gold) is expected


def test_first_ranks_uses_any_key_in_a_group() -> None:
    assert first_ranks(["x", "govuk#breaks", "wtr#reg-12"], GOLD) == [2, None]


def test_all_requirements_found() -> None:
    retrieved = ["wtr#reg-10", "x", "govuk#breaks"]
    assert recall_at_k(retrieved, GOLD, k=3) == 1.0
    assert complete_at_k(retrieved, GOLD, k=3) == 1.0
    assert hit_at_k(retrieved, GOLD, k=3) == 1.0
    assert reciprocal_rank(retrieved, GOLD) == 1.0


def test_k_cuts_off_later_results() -> None:
    retrieved = ["wtr#reg-10", "x", "govuk#breaks"]
    assert recall_at_k(retrieved, GOLD, k=2) == 0.5
    assert complete_at_k(retrieved, GOLD, k=2) == 0.0
    assert hit_at_k(retrieved, GOLD, k=2) == 1.0


def test_alternatives_in_one_group_count_once() -> None:
    # Both the Act and the guidance meet the same requirement: still only 1 of 2 met.
    assert recall_at_k(["wtr#reg-12", "govuk#breaks"], GOLD, k=5) == 0.5


def test_nothing_found() -> None:
    retrieved = ["x", "y"]
    assert recall_at_k(retrieved, GOLD, k=5) == 0.0
    assert hit_at_k(retrieved, GOLD, k=5) == 0.0
    assert reciprocal_rank(retrieved, GOLD) == 0.0
    assert ndcg_at_k(retrieved, GOLD, k=5) == 0.0


def test_reciprocal_rank_is_one_over_the_first_relevant_rank() -> None:
    assert reciprocal_rank(["x", "y", "y", "wtr#reg-10"], GOLD) == 0.25


def test_ndcg_rewards_order() -> None:
    best = ndcg_at_k(["wtr#reg-12", "wtr#reg-10", "x"], GOLD, k=3)
    worse = ndcg_at_k(["x", "wtr#reg-12", "wtr#reg-10"], GOLD, k=3)
    assert best == 1.0
    assert 0 < worse < best
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_ndcg_is_capped_when_one_section_meets_two_requirements() -> None:
    gold = [["govuk#young"], ["govuk#young"]]
    assert ndcg_at_k(["govuk#young"], gold, k=5) == 1.0


def test_mean_of_nothing_is_nan() -> None:
    assert math.isnan(mean([]))
    assert mean([1.0, 0.0]) == 0.5
