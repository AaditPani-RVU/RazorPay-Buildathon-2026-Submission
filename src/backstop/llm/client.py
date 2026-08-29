"""Provider-agnostic LLM layer.

Two design rules hold the rest of the system together:

1.  The model is reached only through `structured()`, which returns a validated
    pydantic model or raises. No free-form model text ever flows into the
    recovery pipeline; every stage consumes typed objects.
2.  Providers are swappable. The pipeline depends on the `Provider` protocol,
    not on Groq, so the reasoning backend is a configuration choice.

Open-weights models are less reliable at schema adherence than frontier models,
so `structured()` validates and makes a bounded repair attempt before failing.
Parse and repair rates are recorded as metrics -- a malformed plan that never
reaches the policy engine is a fact worth reporting, not hiding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a structured call cannot be satisfied after repair."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Provider(Protocol):
    """Minimal surface a reasoning backend must implement."""

    name: str

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse: ...


@dataclass
class CallStats:
    """Observability for the reasoning layer, surfaced in eval reports."""

    calls: int = 0
    repairs: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def repair_rate(self) -> float:
        return self.repairs / self.calls if self.calls else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


@dataclass
class StructuredCallResult:
    value: BaseModel
    repaired: bool
    raw: str


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull a JSON object out of model output.

    Small models wrap JSON in prose or code fences and reasoning models emit a
    preamble, so accept those shapes rather than demanding a bare object.
    """
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()


@dataclass
class LLMClient:
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    stats: CallStats = field(default_factory=CallStats)

    def structured(
        self, *, system: str, user: str, schema: type[T], model: str | None = None
    ) -> StructuredCallResult:
        """Call the model and return a validated `schema` instance.

        One repair attempt is made, feeding the validation error back. If that
        also fails the caller gets an `LLMError` -- the pipeline treats an
        unparseable plan as "no action proposed", which is safe by default.
        """
        target = model or self.model
        instructions = (
            f"{system}\n\nRespond with a single JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            "Output only the JSON object. No prose, no code fences."
        )
        self.stats.calls += 1
        first = self._complete(instructions, user, target)
        try:
            return StructuredCallResult(schema.model_validate_json(extract_json(first.text)), False, first.text)
        except (ValidationError, ValueError) as err:
            self.stats.repairs += 1
            repair = (
                f"{user}\n\nYour previous response was rejected:\n{first.text[:2000]}\n\n"
                f"It failed validation with:\n{err}\n\nReturn only corrected JSON."
            )
            second = self._complete(instructions, repair, target)
            try:
                return StructuredCallResult(
                    schema.model_validate_json(extract_json(second.text)), True, second.text
                )
            except (ValidationError, ValueError) as err2:
                self.stats.failures += 1
                raise LLMError(f"{schema.__name__} unparseable after repair: {err2}") from err2

    def _complete(self, system: str, user: str, model: str) -> LLMResponse:
        resp = self.provider.complete(
            system=system,
            user=user,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.stats.prompt_tokens += resp.prompt_tokens
        self.stats.completion_tokens += resp.completion_tokens
        return resp
