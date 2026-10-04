"""Tests for the MockBackend: determinism, diversity, majority-correctness."""

from self_consistency.aggregation import normalize_answer
from self_consistency.answer_extraction import extract_answer
from self_consistency.backends import MockBackend
from self_consistency.datasets import SAMPLE_PROBLEMS

PROMPT = "Question: {q}\nSolve step by step. End with 'Answer: <value>'."


def _answers(backend, problem, n=11):
    out = []
    for _ in range(n):
        path = backend.generate(PROMPT.format(q=problem["question"]), temperature=0.7)
        raw = extract_answer(path)
        assert raw is not None, f"no answer extracted from path:\n{path}"
        out.append(normalize_answer(raw))
    return out


def test_mock_backend_determinism_same_seed_same_paths():
    b1 = MockBackend(seed=123, problems=SAMPLE_PROBLEMS)
    b2 = MockBackend(seed=123, problems=SAMPLE_PROBLEMS)
    q = SAMPLE_PROBLEMS[0]["question"]
    paths1 = [b1.generate(q, temperature=0.7) for _ in range(8)]
    paths2 = [b2.generate(q, temperature=0.7) for _ in range(8)]
    assert paths1 == paths2


def test_mock_backend_different_seeds_differ():
    b1 = MockBackend(seed=1, problems=SAMPLE_PROBLEMS)
    b2 = MockBackend(seed=2, problems=SAMPLE_PROBLEMS)
    q = SAMPLE_PROBLEMS[0]["question"]
    paths1 = [b1.generate(q, temperature=0.7) for _ in range(8)]
    paths2 = [b2.generate(q, temperature=0.7) for _ in range(8)]
    assert paths1 != paths2


def test_mock_backend_paths_are_diverse():
    b = MockBackend(seed=7, problems=SAMPLE_PROBLEMS)
    q = SAMPLE_PROBLEMS[0]["question"]
    paths = {b.generate(q, temperature=0.7) for _ in range(12)}
    assert len(paths) > 3, "expected diverse reasoning-path variants"


def test_mock_backend_greedy_is_deterministic():
    # temperature 0 = one deterministic draw from the path distribution:
    # same seed -> same path, and it may be a wrong path (that's the point
    # of the greedy baseline).
    b1 = MockBackend(seed=11, problems=SAMPLE_PROBLEMS)
    b2 = MockBackend(seed=11, problems=SAMPLE_PROBLEMS)
    for p in SAMPLE_PROBLEMS:
        assert b1.generate(p["question"], temperature=0.0) == b2.generate(
            p["question"], temperature=0.0
        )


def test_mock_backend_greedy_can_be_wrong():
    # over the sample set the greedy single-path baseline should miss at
    # least one question for some seed (else the comparison is vacuous)
    missed = False
    for seed in range(10):
        b = MockBackend(seed=seed, problems=SAMPLE_PROBLEMS)
        for p in SAMPLE_PROBLEMS:
            path = b.generate(p["question"], temperature=0.0)
            if normalize_answer(extract_answer(path)) != normalize_answer(p["gold"]):
                missed = True
    assert missed


def test_mock_backend_majority_answer_is_correct():
    # Core property the decoder relies on: the correct answer wins a vote.
    from self_consistency.aggregation import majority_vote

    b = MockBackend(seed=0, problems=SAMPLE_PROBLEMS)
    for p in SAMPLE_PROBLEMS:
        answers = _answers(b, p, n=11)
        winner, _counts = majority_vote(answers)
        assert winner == normalize_answer(p["gold"]), (
            f"{p['id']}: majority {winner} != gold {p['gold']}; votes={answers}"
        )


def test_mock_backend_unknown_prompt_gives_fallback():
    b = MockBackend(seed=0, problems=SAMPLE_PROBLEMS)
    path = b.generate("some totally unrelated prompt", temperature=0.7)
    assert isinstance(path, str) and len(path) > 0
