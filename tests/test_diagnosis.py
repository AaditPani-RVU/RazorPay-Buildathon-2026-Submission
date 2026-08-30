"""Diagnosis behaviour, tested offline against scripted model output."""

import json

import pytest

from backstop.detect.correlate import correlate
from backstop.detect.multires import MultiResolutionDetector
from backstop.diagnose.diagnoser import TAXONOMY, Diagnoser
from backstop.diagnose.evidence import EvidenceBuilder, SegmentObservation
from backstop.detect.detector import SegmentKey
from backstop.domain.declines import DeclineCode, RootCause
from backstop.evaluation import diagnosis_score as ds
from backstop.llm import LLMClient, ScriptedProvider
from backstop.simulate.generator import SimConfig, generate


def diagnoser(*responses: str) -> Diagnoser:
    return Diagnoser(LLMClient(provider=ScriptedProvider(list(responses)), model="scripted"))


def reply(cause: str, confidence: float = 0.9, **extra) -> str:
    body = {
        "root_cause": cause,
        "confidence": confidence,
        "locus": "rail=card",
        "key_evidence": ["otp_abandoned 82% of failures"],
        **extra,
    }
    return json.dumps(body)


@pytest.fixture(scope="module")
def bundles():
    scenario = generate(SimConfig(seed=7, days=2, orders_per_day=20000))
    attempts = [a for o in scenario.orders for a in o.attempts]
    clusters = correlate(MultiResolutionDetector().run(attempts))
    builder = EvidenceBuilder(attempts)
    return scenario, clusters, [builder.build(c) for c in clusters]


# -- evidence ------------------------------------------------------------


def test_degradation_renders_as_a_negative_delta():
    """A collapse shown as a positive number invites the exact misreading the
    evidence exists to prevent."""
    obs = SegmentObservation(SegmentKey("rail", "card"), attempts=100, successes=60,
                             reference_rate=0.9)
    assert obs.delta_pp == pytest.approx(-30.0)
    assert obs.drop_pp == pytest.approx(30.0)
    assert "-30.0pp" in obs.line()


def test_improvement_renders_as_a_positive_delta():
    obs = SegmentObservation(SegmentKey("rail", "upi"), attempts=100, successes=95,
                             reference_rate=0.9)
    assert obs.delta_pp == pytest.approx(5.0)
    assert obs.drop_pp == 0.0


@pytest.mark.slow
def test_bundle_carries_healthy_peers(bundles):
    """Healthy peers are what separate a routing fault from an issuer outage.
    Without them a model is guessing."""
    _, clusters, built = bundles
    acq = next(b for b in built if b.primary.segment.dimension == "acquirer")
    peers = acq.peers_by_dimension["acquirer"]
    assert len(peers) >= 3
    assert any(p.drop_pp < 5 for p in peers), "at least one clean peer must be shown"
    assert any(p.drop_pp > 15 for p in peers), "and the failing one"


@pytest.mark.slow
def test_bundle_reference_rate_is_within_segment(bundles):
    """Rails have genuinely different baselines; comparing across them would
    invent effects that are not there."""
    _, _, built = bundles
    for b in built:
        for obs in b.peers_by_dimension.get("rail", []):
            assert 0.5 < obs.reference_rate < 1.0


# -- diagnoser -----------------------------------------------------------


def test_taxonomy_covers_every_root_cause():
    """A cause the prompt never describes cannot be chosen deliberately."""
    assert set(TAXONOMY) == set(RootCause)


def test_valid_response_is_parsed(bundles=None):
    d = diagnoser(reply("authentication_dropoff"))
    result = d.diagnose(_stub_bundle())
    assert result.ok
    assert result.diagnosis.root_cause is RootCause.AUTHENTICATION_DROPOFF


def test_free_text_cause_is_rejected_then_repaired():
    """The closed enum is what stops a weaker model's prose reaching the planner."""
    d = diagnoser(reply("HDFC issuer system outage"), reply("issuer_outage"))
    result = d.diagnose(_stub_bundle())
    assert result.ok and result.repaired
    assert result.diagnosis.root_cause is RootCause.ISSUER_OUTAGE


def test_unusable_output_yields_no_diagnosis_rather_than_a_guess():
    """No cause means no intervention, which is the safe default."""
    d = diagnoser("I cannot determine this", "still not json")
    result = d.diagnose(_stub_bundle())
    assert not result.ok
    assert result.diagnosis is None and result.error


def _stub_bundle():
    from datetime import UTC, datetime

    from backstop.diagnose.evidence import EvidenceBundle
    from backstop.domain.money import Money

    obs = SegmentObservation(
        SegmentKey("rail", "card"), attempts=400, successes=260, reference_rate=0.87,
        declines={DeclineCode.OTP_ABANDONED: 115, DeclineCode.INSUFFICIENT_FUNDS: 25},
    )
    return EvidenceBundle(
        cluster_id="c1", starts_at=datetime(2026, 8, 1, tzinfo=UTC),
        ends_at=datetime(2026, 8, 1, 3, tzinfo=UTC), primary=obs,
        peers_by_dimension={"rail": [obs]}, money_at_risk=Money.rupees(50000),
        total_failures=140, duration_minutes=180,
    )


# -- scoring -------------------------------------------------------------


def test_retry_class_errors_are_flagged_as_dangerous():
    """Calling an auth drop-off an outage turns 'bring the customer back' into
    'retry the charge', which cannot work and burns issuer goodwill."""
    assert ds.is_dangerous(RootCause.AUTHENTICATION_DROPOFF, RootCause.PSP_OR_RAIL_OUTAGE)
    assert ds.is_dangerous(RootCause.ISSUER_OUTAGE, RootCause.FRAUD_PRESSURE)


def test_confusions_within_a_class_are_not_dangerous():
    """Both still mean 'retry when it recovers'; the label is wrong, not the plan."""
    assert not ds.is_dangerous(RootCause.ISSUER_OUTAGE, RootCause.PSP_OR_RAIL_OUTAGE)


def test_calibration_gap_detects_uninformative_confidence():
    rep = ds.DiagnosisReport()
    rep.confidence_when_right = [0.95, 0.95]
    rep.confidence_when_wrong = [0.95]
    assert rep.calibration_gap == pytest.approx(0.0)
