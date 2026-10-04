"""Tests for majority vote, answer normalization, and tie-breaking."""

import pytest

from self_consistency.aggregation import majority_vote, normalize_answer


def test_majority_vote_basic():
    winner, counts = majority_vote(["5", "5", "6", "5", "7"])
    assert winner == "5"
    assert counts == {"5": 3, "6": 1, "7": 1}


def test_majority_vote_tie_breaks_by_first_seen():
    # "6" and "5" both have 2 votes; "6" appeared first -> deterministic winner
    winner, counts = majority_vote(["6", "5", "6", "5"])
    assert winner == "6"
    assert counts["6"] == counts["5"] == 2


def test_majority_vote_tie_breaking_is_stable():
    first = majority_vote(["b", "a", "b", "a"])[0]
    second = majority_vote(["b", "a", "b", "a"])[0]
    assert first == second == "b"


def test_majority_vote_single_answer():
    winner, counts = majority_vote(["42"])
    assert winner == "42"
    assert counts == {"42": 1}


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  42  ", "42"),
        ("42.", "42"),
        ("42.0", "42"),
        ("3.50", "3.5"),
        ("1,000", "1000"),
        ("1,234,567", "1234567"),
        ("-7", "-7"),
        ("0.5", "0.5"),
        ("  Hello World ", "helloworld"),
        ("FORTY-TWO", "forty-two"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_answer_none_and_empty():
    assert normalize_answer(None) == ""
    assert normalize_answer("") == ""
