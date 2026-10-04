"""Answer aggregation: normalization + majority vote (answer marginalization).

Paper mapping (Wang et al., ICLR 2023, "Self-Consistency Improves Chain of
Thought Reasoning in Language Models"):

    The paper's core decoding step is *answer marginalization*:

        a* = argmax_a  sum_{i=1..m}  1[a_i == a]

    i.e. sample m diverse reasoning paths r_1..r_m, extract the final answer
    a_i from each path, and pick the answer that the largest number of paths
    agree on. Since each path votes exactly once, the sum over indicator
    functions is precisely a majority vote over the extracted answers.
    `majority_vote` below implements exactly this argmax.

    `normalize_answer` handles the paper's practical wrinkle that the same
    answer can be written in different surface forms ("42", "42.0", "1,000")
    and should count as one vote.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def normalize_answer(text: str) -> str:
    """Normalize an extracted answer so equivalent surface forms merge.

    - strips surrounding whitespace and trailing punctuation
    - removes thousands separators (``1,000`` -> ``1000``)
    - unifies numeric formatting (``42.0`` -> ``42``, ``3.50`` -> ``3.5``)
    - lowercases non-numeric answers for comparison
    """
    if text is None:
        return ""
    t = text.strip().strip(".,;:!?\"'`").replace(" ", "").replace(",", "")
    try:
        value = float(t)
        if value.is_integer():
            return str(int(value))
        # repr-style: shortest round-trip, avoids float noise like 3.5000001
        return str(value)
    except ValueError:
        return t.lower()


def majority_vote(answers: List[str]) -> Tuple[str, Dict[str, int]]:
    """Return ``(winner, counts)`` by marginalizing over reasoning paths.

    Implements ``argmax_a sum_i 1[a_i == a]`` from the paper. Ties are broken
    deterministically: the answer that appeared *first* among the top-tied
    answers wins (insertion order of the counts dict), so identical inputs
    always produce identical outputs.

    Raises
    ------
    ValueError
        If ``answers`` is empty.
    """
    if not answers:
        raise ValueError("majority_vote requires at least one answer")
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best_count = max(counts.values())
    # first key in insertion order with the best count -> deterministic ties
    winner = next(a for a in counts if counts[a] == best_count)
    return winner, counts
