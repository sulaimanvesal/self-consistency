#!/usr/bin/env python3
"""Offline demo: self-consistency decoding on sample arithmetic questions.

No API key, no network. Runs entirely on the seeded MockBackend.

Usage:
    python demo/demo_offline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_consistency import (  # noqa: E402
    MockBackend,
    SelfConsistencyDecoder,
    SAMPLE_PROBLEMS,
    extract_answer,
)


def main() -> int:
    backend = MockBackend(seed=0, problems=SAMPLE_PROBLEMS)
    decoder = SelfConsistencyDecoder(backend, n_samples=11, temperature=0.7)

    for i, p in enumerate(SAMPLE_PROBLEMS[:3], 1):
        result = decoder.decode(p["question"])
        print("=" * 72)
        print(f"QUESTION {i}: {p['question']}")
        print(f"gold answer: {p['gold']}")
        print("-" * 72)
        for j, path in enumerate(result.paths, 1):
            print(f"--- path {j} (extracted: {result.extracted_answers[j - 1]}) ---")
            print(path)
            print()
        print(f"vote counts : {result.vote_counts}")
        print(f"final answer: {result.final_answer} "
              f"(confidence {result.confidence:.2f}, n={result.n_samples})")
        print(f"correct?    : {'YES' if result.final_answer == p['gold'] else 'NO'}")
        print()

    print("Demo finished: all offline, no API key used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
