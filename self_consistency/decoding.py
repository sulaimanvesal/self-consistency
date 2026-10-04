"""Self-consistency decoding.

Implements the decoding procedure of Wang et al. (ICLR 2023):

1. Sample ``n_samples`` diverse chain-of-thought reasoning paths from the
   backend at temperature > 0 (the MockBackend emulates this offline via
   seeded template variants).
2. Extract each path's final answer.
3. Marginalize: pick the answer with the most votes (see
   ``aggregation.majority_vote``).

``greedy_decode`` is the baseline from the paper: a single greedy
(temperature 0) path, no voting — the "standard CoT prompting" control.
``evaluate`` runs both over the built-in sample set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .aggregation import majority_vote, normalize_answer
from .answer_extraction import extract_answer
from .backends import LLMBackend, MockBackend
from .datasets import SAMPLE_PROBLEMS

DEFAULT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Solve it step by step, showing your reasoning. "
    "End your response with the final answer on its own as 'Answer: <value>'."
)


@dataclass
class DecodingResult:
    """Outcome of one self-consistency decode run."""

    question: str
    paths: List[str]
    extracted_answers: List[Optional[str]]
    vote_counts: Dict[str, int]
    final_answer: str
    confidence: float  # winner's share of the votes, in [0, 1]
    n_samples: int


class SelfConsistencyDecoder:
    """Sample diverse reasoning paths and take the majority-vote answer.

    Parameters
    ----------
    backend:
        Any ``LLMBackend`` (``generate(prompt, temperature) -> str``).
    n_samples:
        Number of reasoning paths to sample (m in the paper).
    answer_extractor:
        Callable mapping a path string to its final answer string.
    temperature:
        Sampling temperature for the diverse paths (> 0).
    prompt_template:
        ``str.format`` template with a ``{question}`` slot.
    """

    def __init__(
        self,
        backend: LLMBackend,
        n_samples: int = 5,
        answer_extractor: Callable[[str], Optional[str]] = extract_answer,
        temperature: float = 0.7,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.backend = backend
        self.n_samples = n_samples
        self.answer_extractor = answer_extractor
        self.temperature = temperature
        self.prompt_template = prompt_template

    def decode(self, question: str) -> DecodingResult:
        prompt = self.prompt_template.format(question=question)
        paths: List[str] = []
        answers: List[Optional[str]] = []
        for _ in range(self.n_samples):
            path = self.backend.generate(prompt, temperature=self.temperature)
            paths.append(path)
            raw = self.answer_extractor(path)
            answers.append(normalize_answer(raw) if raw is not None else None)
        # Paths whose answer could not be extracted abstain (paper: they are
        # simply not counted); if every path abstains, we report no answer.
        votable = [a for a in answers if a is not None]
        if votable:
            winner, counts = majority_vote(votable)
            confidence = counts[winner] / len(votable)
        else:
            winner, counts, confidence = "", {}, 0.0
        return DecodingResult(
            question=question,
            paths=paths,
            extracted_answers=answers,
            vote_counts=counts,
            final_answer=winner,
            confidence=confidence,
            n_samples=self.n_samples,
        )


def greedy_decode(
    question: str,
    backend: LLMBackend,
    answer_extractor: Callable[[str], Optional[str]] = extract_answer,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> DecodingResult:
    """Baseline: one greedy (temperature 0) reasoning path, no voting."""
    prompt = prompt_template.format(question=question)
    path = backend.generate(prompt, temperature=0.0)
    raw = answer_extractor(path)
    answer = normalize_answer(raw) if raw is not None else ""
    return DecodingResult(
        question=question,
        paths=[path],
        extracted_answers=[answer],
        vote_counts={answer: 1},
        final_answer=answer,
        confidence=1.0,
        n_samples=1,
    )


def evaluate(
    problems: Optional[List[dict]] = None,
    n_samples: int = 5,
    seed: int = 0,
    temperature: float = 0.7,
    verbose: bool = False,
) -> dict:
    """Run greedy vs. self-consistency over the sample set.

    Returns a summary dict with per-question rows and both accuracies.
    """
    problems = problems if problems is not None else SAMPLE_PROBLEMS
    backend = MockBackend(seed=seed, problems=problems)
    decoder = SelfConsistencyDecoder(backend, n_samples=n_samples, temperature=temperature)

    rows = []
    for p in problems:
        sc = decoder.decode(p["question"])
        gr = greedy_decode(p["question"], backend)
        gold = normalize_answer(p["gold"])
        rows.append(
            {
                "id": p["id"],
                "gold": gold,
                "greedy_answer": gr.final_answer,
                "greedy_correct": gr.final_answer == gold,
                "sc_answer": sc.final_answer,
                "sc_correct": sc.final_answer == gold,
                "sc_confidence": sc.confidence,
                "votes": sc.vote_counts,
            }
        )
        if verbose:
            print(
                f"[{p['id']}] gold={gold} "
                f"greedy={gr.final_answer}({'OK' if gr.final_answer == gold else 'X'}) "
                f"self-consistency={sc.final_answer}({'OK' if sc.final_answer == gold else 'X'}) "
                f"conf={sc.confidence:.2f} votes={sc.vote_counts}"
            )
    n = len(rows)
    summary = {
        "n_problems": n,
        "n_samples": n_samples,
        "greedy_accuracy": sum(r["greedy_correct"] for r in rows) / n if n else 0.0,
        "sc_accuracy": sum(r["sc_correct"] for r in rows) / n if n else 0.0,
        "rows": rows,
    }
    return summary
