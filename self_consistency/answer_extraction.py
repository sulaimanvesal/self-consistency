"""Regex-based final-answer extraction from reasoning paths.

A reasoning path is free-form text; the final answer is located by a small
cascade of patterns, tried in order:

1. ``#### <value>``  (GSM8K's canonical answer marker)
2. ``Answer: <value>`` / ``Answer = <value>`` (our demo's convention)
3. a bare number on the last non-empty line (fallback)

The extractor is configurable: pass ``patterns=...`` to
``make_extractor`` to override the cascade.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

_PATTERN_SPECS: List[tuple] = [
    ("gsm8k_marker", re.compile(r"####\s*([^\n]+)")),
    ("answer_label", re.compile(r"[Aa]nswer\s*[:=]\s*([^\s]+)")),
    ("last_line_number", re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*$")),
]


def _clean(raw: str) -> str:
    # trim trailing punctuation/words; keep the first token-looking chunk
    return raw.strip().strip(".,;:!?\"'`")


def extract_answer(text: str, patterns: Optional[Sequence[re.Pattern]] = None) -> Optional[str]:
    """Extract the final answer string from a reasoning path, or None."""
    if not text:
        return None
    if patterns is None:
        # default cascade: try each spec; for the "last line" fallback only
        # consider the final non-empty line
        for name, pat in _PATTERN_SPECS:
            if name == "last_line_number":
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if not lines:
                    return None
                m = pat.search(lines[-1])
                return _clean(m.group(1)) if m else None
            m = pat.search(text)
            if m:
                return _clean(m.group(1))
        return None
    for pat in patterns:
        m = pat.search(text)
        if m:
            return _clean(m.group(1))
    return None


def make_extractor(*pattern_strings: str) -> Callable[[str], Optional[str]]:
    """Build a custom extractor from regex strings (first group = answer)."""
    compiled = [re.compile(p) for p in pattern_strings]

    def _extract(text: str) -> Optional[str]:
        return extract_answer(text, patterns=compiled)

    return _extract
