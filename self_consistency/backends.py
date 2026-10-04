"""LLM backends.

`LLMBackend` is the protocol the decoder programs against: any object with a
`generate(prompt, temperature) -> str` method works.

`MockBackend` is a fully offline, deterministic (seeded) stand-in for an LLM.
It emulates temperature sampling of diverse chain-of-thought reasoning paths:
each call draws a reasoning *template variant* (different step orders and
phrasing), and with a minority probability it produces a plausible-but-wrong
path (a common arithmetic slip). By construction the most common final answer
across samples is the correct one, which is exactly the regime where
self-consistency helps.

`OpenAICompatBackend` is OPTIONAL: it is only used when the user explicitly
chooses it, the `openai` package is installed, and `OPENAI_API_KEY` is set.
It is imported lazily and never touched by the offline path.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Protocol


class LLMBackend(Protocol):
    """Protocol for a text-generation backend."""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate one completion for *prompt* at the given temperature."""
        ...


_CORRECT_TEMPLATES = [
    # (name, template); {steps} = newline-joined steps, {answer} = gold answer
    ("forward_numbered", "Let's solve this step by step.\n{steps}\nTherefore, Answer: {answer}"),
    ("forward_narrative", "I'll work through this carefully.\n{steps}\nSo the final answer is Answer: {answer}"),
    ("forward_checked", "Let's compute it directly.\n{steps}\nDouble-checking the last step confirms it.\nAnswer: {answer}"),
    ("backward_then_forward", "Let me think about what the final quantity must be.\n{last_step}\nVerifying forward: {steps}\nAnswer: {answer}"),
    ("concise_chain", "Reasoning:\n{steps}\nAnswer: {answer}"),
]

_WRONG_TEMPLATES = [
    # plausible slips: stop one step early / off-by-one / distractor reasoning
    ("stopped_early", "Let's solve this step by step.\n{early_steps}\nThat covers everything, so Answer: {wrong}"),
    ("off_by_one", "Let's solve this step by step.\n{steps}\nHmm, let me recount the last step... Answer: {wrong}"),
    ("distractor", "Let's solve this step by step.\n{steps}\nActually, the key intermediate total is what matters here. Answer: {wrong}"),
]


class MockBackend:
    """Offline, seeded stand-in for an LLM.

    Parameters
    ----------
    seed:
        Seed for the internal RNG. Two backends with the same seed produce
        the same sequence of reasoning paths (deterministic).
    problems:
        List of ``{"question", "gold", "steps", "distractor"}`` dicts. Each
        ``steps`` entry is a ``(text, value)`` tuple describing one correct
        arithmetic step and its intermediate value.
    p_correct:
        Probability that a sampled path ends at the correct answer
        (default 0.8 — a clear majority, matching the paper's setting where
        correct reasoning paths dominate the samples).
    """

    def __init__(self, seed: int = 0, problems=None, p_correct: float = 0.8):
        self.seed = seed
        self.problems = list(problems) if problems else []
        self.p_correct = p_correct
        self._rng = random.Random(seed)
        self._calls = 0

    # -- internals ---------------------------------------------------------
    def _match(self, prompt: str) -> Optional[dict]:
        for p in self.problems:
            if p["question"] in prompt:
                return p
        return None

    def _render_steps(self, steps, prefix: str) -> str:
        lines = []
        for i, (text, _value) in enumerate(steps, start=1):
            lines.append(f"{prefix}{i}: {text}")
        return "\n".join(lines)

    def _correct_path(self, problem: dict) -> str:
        steps, gold = problem["steps"], problem["gold"]
        name, tmpl = self._rng.choice(_CORRECT_TEMPLATES)
        rendered = self._render_steps(steps, "Step ")
        path = tmpl.format(
            steps=rendered,
            last_step=self._render_steps([steps[-1]], "Step "),
            answer=gold,
        )
        return path

    def _wrong_path(self, problem: dict) -> str:
        steps, gold = problem["steps"], problem["gold"]
        name, tmpl = self._rng.choice(_WRONG_TEMPLATES)
        wrong_choices = []
        if len(steps) > 1:
            # slip 1: stop one step early and report the intermediate value
            wrong_choices.append(steps[-2][1])
        # slip 2: off-by-one on the final answer (only when numeric)
        try:
            g = float(gold)
            wrong_choices.append(str(int(g) + 1) if g.is_integer() else str(g + 1))
        except ValueError:
            pass
        # slip 3: the problem's plausible distractor answer
        if problem.get("distractor") and problem["distractor"] != gold:
            wrong_choices.append(problem["distractor"])
        # never "accidentally" be correct
        wrong_choices = [w for w in wrong_choices if w != gold] or [gold + " (approx)"]
        wrong = self._rng.choice(wrong_choices)
        rendered = self._render_steps(steps, "Step ")
        return tmpl.format(
            steps=rendered,
            early_steps=self._render_steps(steps[:-1], "Step ") if len(steps) > 1 else rendered,
            wrong=wrong,
        )

    def _fallback_path(self) -> str:
        # For prompts that don't match a known problem: deterministic filler.
        return "Let me think step by step.\nStep 1: This question is outside my practice set.\nAnswer: unknown"

    # -- public API --------------------------------------------------------
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self._calls += 1
        problem = self._match(prompt)
        if problem is None:
            return self._fallback_path()
        # Greedy (temperature 0) = one deterministic draw from the path
        # distribution (fixed by seed + call order), so repeated runs agree.
        # Sampled (temperature > 0) = one draw per call; the RNG sequence
        # makes each call a different reasoning-path variant.
        if self._rng.random() < self.p_correct:
            return self._correct_path(problem)
        return self._wrong_path(problem)


class OpenAICompatBackend:
    """OPTIONAL backend for any OpenAI-compatible chat-completions endpoint.

    Lazily imports `openai` and reads credentials at call time, so importing
    this module never breaks the offline path. Only used when the user
    explicitly instantiates it (requires `openai` installed and
    `OPENAI_API_KEY` set).
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: int = 512):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatBackend needs the optional `openai` package "
                "(`pip install openai`)."
            ) from exc
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAICompatBackend needs OPENAI_API_KEY set (or api_key passed)."
            )
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
