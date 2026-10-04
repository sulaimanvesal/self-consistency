"""End-to-end tests for the decoder, greedy baseline, and evaluate()."""

from self_consistency.aggregation import majority_vote, normalize_answer
from self_consistency.backends import MockBackend
from self_consistency.datasets import SAMPLE_PROBLEMS
from self_consistency.decoding import (
    DecodingResult,
    SelfConsistencyDecoder,
    evaluate,
    greedy_decode,
)


def _decoder(seed=0, n_samples=11):
    return SelfConsistencyDecoder(
        MockBackend(seed=seed, problems=SAMPLE_PROBLEMS),
        n_samples=n_samples,
    )


def test_decoder_result_matches_majority_vote():
    decoder = _decoder(seed=3)
    q = SAMPLE_PROBLEMS[0]["question"]
    result = decoder.decode(q)
    assert isinstance(result, DecodingResult)
    assert result.n_samples == 11
    assert len(result.paths) == 11
    assert len(result.extracted_answers) == 11
    votable = [a for a in result.extracted_answers if a is not None]
    winner, counts = majority_vote(votable)
    assert result.final_answer == winner
    assert result.vote_counts == counts
    assert result.confidence == counts[winner] / len(votable)
    assert 0.0 < result.confidence <= 1.0


def test_decoder_final_answer_is_gold_on_sample_set():
    # voting over 11 diverse paths recovers the gold answer on every problem
    # (the paper's core claim, reproduced against the seeded mock)
    for seed in (0, 1, 2):
        decoder = _decoder(seed=seed)
        for p in SAMPLE_PROBLEMS:
            result = decoder.decode(p["question"])
            assert result.final_answer == normalize_answer(p["gold"]), (
                f"seed={seed} {p['id']}: got {result.final_answer}, expected {p['gold']}"
            )


def test_decoder_rejects_bad_n_samples():
    with __import__("pytest").raises(ValueError):
        SelfConsistencyDecoder(MockBackend(seed=0), n_samples=0)


def test_greedy_decode_single_path():
    backend = MockBackend(seed=5, problems=SAMPLE_PROBLEMS)
    result = greedy_decode(SAMPLE_PROBLEMS[0]["question"], backend)
    assert result.n_samples == 1
    assert len(result.paths) == 1
    assert result.confidence == 1.0
    # deterministic: a fresh backend with the same seed gives the same path
    backend2 = MockBackend(seed=5, problems=SAMPLE_PROBLEMS)
    result2 = greedy_decode(SAMPLE_PROBLEMS[0]["question"], backend2)
    assert result2.paths == result.paths


def test_evaluate_runs_and_returns_summary():
    summary = evaluate(n_samples=5, seed=0)
    assert summary["n_problems"] == len(SAMPLE_PROBLEMS) == 12
    assert summary["n_samples"] == 5
    assert len(summary["rows"]) == 12
    for row in summary["rows"]:
        assert set(row) >= {
            "id", "gold", "greedy_answer", "greedy_correct",
            "sc_answer", "sc_correct", "sc_confidence", "votes",
        }
    assert 0.0 <= summary["greedy_accuracy"] <= 1.0
    assert 0.0 <= summary["sc_accuracy"] <= 1.0
    # the paper's headline result, reproduced on the toy set: voting over
    # diverse paths beats (or matches) the single greedy path
    assert summary["sc_accuracy"] >= summary["greedy_accuracy"]
    assert summary["sc_accuracy"] >= 0.9


def test_evaluate_custom_problems_subset():
    summary = evaluate(problems=SAMPLE_PROBLEMS[:2], n_samples=3, seed=1)
    assert summary["n_problems"] == 2
    assert len(summary["rows"]) == 2
