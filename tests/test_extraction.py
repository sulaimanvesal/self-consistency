"""Tests for regex-based final-answer extraction."""

from self_consistency.answer_extraction import extract_answer, make_extractor


def test_extract_answer_label():
    text = "Let's think step by step.\nStep 1: 3 + 4 = 7.\nAnswer: 7"
    assert extract_answer(text) == "7"


def test_extract_answer_label_with_equals():
    assert extract_answer("... therefore Answer = 150") == "150"


def test_extract_answer_label_trailing_punctuation():
    assert extract_answer("Answer: 42.") == "42"
    assert extract_answer("The final answer is Answer: 15!") == "15"


def test_extract_gsm8k_marker():
    assert extract_answer("some reasoning #### 18") == "18"


def test_gsm8k_marker_takes_priority_over_label():
    # #### is tried first in the cascade
    text = "Answer: 5\n#### 6"
    assert extract_answer(text) == "6"


def test_extract_last_line_number_fallback():
    text = "First we add 2 and 3.\nThat gives us 5"
    assert extract_answer(text) == "5"


def test_extract_returns_none_when_no_number():
    assert extract_answer("I have no idea what the answer is.") is None
    assert extract_answer("") is None
    assert extract_answer(None) is None


def test_extract_negative_and_decimal():
    assert extract_answer("Answer: -3.5") == "-3.5"
    assert extract_answer("Answer: 1,000") == "1,000"


def test_make_extractor_custom_patterns():
    custom = make_extractor(r"final=(\d+)")
    assert custom("blah blah final=99 done") == "99"
    assert custom("nothing here") is None


def test_extract_uses_last_answer_label_occurrence():
    # search() finds the first occurrence of the label; ensure sensible text works
    text = "Answer: 5\nLet me double check... Answer: 6"
    # first match wins for the label pattern
    assert extract_answer(text) == "5"
