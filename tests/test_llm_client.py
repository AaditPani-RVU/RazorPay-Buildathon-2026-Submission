"""The reasoning layer must never hand malformed data to the pipeline."""

import pytest
from pydantic import BaseModel, Field

from backstop.llm import LLMClient, LLMError, LLMResponse, ScriptedProvider
from backstop.llm.client import extract_json


class Diagnosis(BaseModel):
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)


def client(*responses: str) -> LLMClient:
    return LLMClient(provider=ScriptedProvider(list(responses)), model="scripted")


def test_parses_bare_json():
    c = client('{"root_cause": "issuer_outage", "confidence": 0.9}')
    result = c.structured(system="s", user="u", schema=Diagnosis)
    assert result.value.root_cause == "issuer_outage"
    assert not result.repaired
    assert c.stats.repair_rate == 0.0


def test_parses_json_wrapped_in_fences_and_prose():
    c = client('Here is my analysis:\n```json\n{"root_cause": "bin_decline", "confidence": 0.7}\n```\nHope that helps!')
    assert c.structured(system="s", user="u", schema=Diagnosis).value.root_cause == "bin_decline"


def test_repairs_invalid_output_once():
    c = client(
        '{"root_cause": "gateway", "confidence": 4.2}',   # confidence out of range
        '{"root_cause": "gateway", "confidence": 0.42}',
    )
    result = c.structured(system="s", user="u", schema=Diagnosis)
    assert result.repaired
    assert result.value.confidence == 0.42
    assert c.stats.repairs == 1


def test_raises_when_repair_also_fails():
    c = client("not json at all", "still not json")
    with pytest.raises(LLMError):
        c.structured(system="s", user="u", schema=Diagnosis)
    assert c.stats.failure_rate == 1.0


def test_extract_json_handles_braces_inside_strings():
    assert extract_json('prefix {"a": "}{"} suffix') == '{"a": "}{"}'


class BrokenProvider:
    """A provider that is down, rate limited, or otherwise unreachable."""

    name = "broken"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, **_):
        raise self.error


def test_a_dead_provider_raises_llm_error_not_a_transport_error():
    """A rate limit must not escape as a RuntimeError.

    Callers treat LLMError as "no plan came back" and fall through to the
    deterministic path. A provider exception that bypasses that handling takes
    the whole run down, which is exactly what happened on a live 429.
    """
    client = LLMClient(
        provider=BrokenProvider(RuntimeError("429 rate limit")), model="m"
    )
    with pytest.raises(LLMError, match="unavailable"):
        client.structured(system="s", user="u", schema=Diagnosis)
    assert client.stats.failures == 1


def test_a_provider_that_dies_during_repair_also_raises_llm_error():
    class DiesOnSecondCall:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def complete(self, **_):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text="not json at all", model="m")
            raise RuntimeError("429 rate limit")

    client = LLMClient(provider=DiesOnSecondCall(), model="m")
    with pytest.raises(LLMError, match="unavailable during repair"):
        client.structured(system="s", user="u", schema=Diagnosis)


def test_diagnosis_degrades_to_no_answer_when_the_provider_is_down():
    """The pipeline's safe default: an undiagnosed cluster, not a crash."""
    from backstop.diagnose.diagnoser import Diagnoser

    class Bundle:
        cluster_id = "cluster_001"

        def render(self):
            return "evidence"

    client = LLMClient(provider=BrokenProvider(RuntimeError("503")), model="m")
    result = Diagnoser(client).diagnose(Bundle())
    assert not result.ok
    assert result.diagnosis is None
    assert "unavailable" in result.error
