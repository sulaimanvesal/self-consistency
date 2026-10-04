"""Command-line interface.

Usage:
    python -m self_consistency demo       # offline demo on sample questions
    python -m self_consistency evaluate   # greedy vs self-consistency table
"""

from __future__ import annotations

import argparse

from .answer_extraction import extract_answer
from .backends import MockBackend, OpenAICompatBackend
from .datasets import SAMPLE_PROBLEMS
from .decoding import SelfConsistencyDecoder, evaluate


def _make_backend(args):
    if args.backend == "openai":
        return OpenAICompatBackend(model=args.model)
    return MockBackend(seed=args.seed, problems=SAMPLE_PROBLEMS)


def cmd_demo(args) -> int:
    backend = _make_backend(args)
    decoder = SelfConsistencyDecoder(
        backend, n_samples=args.samples, temperature=args.temperature
    )
    problems = SAMPLE_PROBLEMS[: args.questions]
    for i, p in enumerate(problems, 1):
        result = decoder.decode(p["question"])
        print("=" * 72)
        print(f"QUESTION {i}/{len(problems)}: {p['question']}")
        print(f"(gold answer: {p['gold']})")
        print("-" * 72)
        for j, path in enumerate(result.paths, 1):
            ans = result.extracted_answers[j - 1]
            print(f"[path {j}] extracted answer: {ans}")
            print(path)
            print()
        print(f"VOTES: {result.vote_counts}")
        print(f"WINNER: {result.final_answer}  (confidence {result.confidence:.2f})")
        print()
    return 0


def cmd_evaluate(args) -> int:
    summary = evaluate(
        n_samples=args.samples, seed=args.seed,
        temperature=args.temperature, verbose=True,
    )
    print("-" * 72)
    print(f"greedy CoT accuracy:        {summary['greedy_accuracy']:.1%}")
    print(f"self-consistency accuracy:  {summary['sc_accuracy']:.1%} "
          f"(n={summary['n_samples']} samples, {summary['n_problems']} problems)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self_consistency",
        description="Self-consistency decoding over chain-of-thought reasoning paths.",
    )
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock",
                        help="LLM backend (mock is offline; openai needs OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="model for the optional OpenAI-compatible backend")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for MockBackend")
    parser.add_argument("--samples", type=int, default=11, help="reasoning paths per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="run the offline demo")
    d.add_argument("--questions", type=int, default=3, help="how many sample questions to run")
    d.set_defaults(func=cmd_demo)
    e = sub.add_parser("evaluate", help="evaluate greedy vs self-consistency")
    e.set_defaults(func=cmd_evaluate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
