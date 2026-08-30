"""Groq backend, plus a deterministic scripted provider for tests and replays.

Model availability was probed against the live API rather than taken from the
docs, which still list a Llama/Qwen lineup this account cannot reach. What is
actually reachable: the two gpt-oss models, and the `groq/compound` agentic
systems (skipped -- they carry built-in tools and non-deterministic behaviour we
do not want in a structured-extraction path).

The split: `openai/gpt-oss-120b` for diagnosis and planning, `openai/gpt-oss-20b`
for high-volume cheap passes. `BACKSTOP_LLM_MODEL` overrides either, so the eval
can deliberately run the weaker model to show the policy engine still holds.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from backstop.llm.client import LLMResponse

REASONING_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "openai/gpt-oss-20b"

# Verified reachable on 2026-08-29 by probing the API directly.
PRODUCTION_MODELS = {REASONING_MODEL, FAST_MODEL}


@dataclass
class GroqProvider:
    """Thin wrapper over the Groq SDK with bounded retry on transient errors."""

    api_key: str | None = None
    max_retries: int = 3
    name: str = "groq"
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it, put it in .env, or use "
                "ScriptedProvider for offline runs."
            )

    @property
    def client(self):  # lazily imported so the package works without the SDK installed
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    text=resp.choices[0].message.content or "",
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
            # Deliberately broad. The SDK raises a different type for rate
            # limits, timeouts, connection resets and 5xx, and every one of
            # them is worth one more attempt. Narrowing this to the exceptions
            # known today means a new SDK error class silently stops retrying.
            except Exception as err:  # noqa: BLE001
                last = err
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Groq call failed after {self.max_retries} attempts: {last}") from last


@dataclass
class ScriptedProvider:
    """Returns canned responses in order. Makes the pipeline testable offline.

    The eval harness needs to be reproducible and runnable without a key; a
    scripted backend also lets tests assert policy behaviour against
    deliberately malformed or hostile model output.
    """

    responses: list[str] = field(default_factory=list)
    name: str = "scripted"
    calls: list[tuple[str, str]] = field(default_factory=list, init=False)
    _i: int = field(default=0, init=False)

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self.calls.append((system, user))
        if self._i >= len(self.responses):
            raise RuntimeError("ScriptedProvider exhausted: more calls than scripted responses")
        text = self.responses[self._i]
        self._i += 1
        return LLMResponse(text=text, model=model or "scripted")
