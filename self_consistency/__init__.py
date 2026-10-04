"""Package init: clean public API for the self-consistency implementation."""

from .backends import LLMBackend, MockBackend, OpenAICompatBackend
from .aggregation import normalize_answer, majority_vote
from .decoding import (
    DecodingResult,
    SelfConsistencyDecoder,
    greedy_decode,
    evaluate,
)
from .answer_extraction import extract_answer
from .datasets import SAMPLE_PROBLEMS

__all__ = [
    "LLMBackend",
    "MockBackend",
    "OpenAICompatBackend",
    "normalize_answer",
    "majority_vote",
    "DecodingResult",
    "SelfConsistencyDecoder",
    "greedy_decode",
    "evaluate",
    "extract_answer",
    "SAMPLE_PROBLEMS",
]

__version__ = "0.1.0"
