"""The reasoning layer must never hand malformed data to the pipeline."""

import pytest
from pydantic import BaseModel, Field

from backstop.llm import LLMClient, LLMError, ScriptedProvider
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
