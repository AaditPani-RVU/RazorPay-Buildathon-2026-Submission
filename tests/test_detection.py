"""Detection behaviour that must not regress."""

import pytest

from backstop.detect.correlate import correlate, covers
from backstop.detect.detector import Detector, DetectorConfig, SegmentKey, segments_for
from backstop.detect.multires import MultiResolutionDetector
from backstop.evaluation.bench import scope_is_exact
from backstop.evaluation.detection_score import score
from backstop.simulate.generator import SimConfig, generate

SEG = SegmentKey


@pytest.fixture(scope="module")
def detected():
    scenario = generate(SimConfig(seed=7))
    attempts = [a for o in scenario.orders for a in o.attempts]
    signals = MultiResolutionDetector().run(attempts)
    clusters = correlate(signals)
    return scenario, signals, clusters, score(scenario, [c.primary for c in clusters])


def test_hierarchy_is_well_formed():
    assert covers(SEG("global", "all"), SEG("rail", "card"))
    assert covers(SEG("rail", "card"), SEG("rail_issuer", "card:HDFC"))
    assert covers(SEG("issuer", "HDFC"), SEG("issuer_bin", "HDFC:591000"))
    assert not covers(SEG("issuer", "SBI"), SEG("issuer_bin", "HDFC:591000"))
    assert not covers(SEG("rail", "card"), SEG("rail", "card")), "nothing covers itself"
    assert not covers(SEG("acquirer", "acq_beta"), SEG("rail", "upi")), "acquirers are orthogonal"


@pytest.mark.slow
def test_every_attempt_lands_in_a_global_segment(detected):
    scenario, *_ = detected
    attempt = next(a for o in scenario.orders for a in o.attempts)
    assert SEG("global", "all") in segments_for(attempt)


@pytest.mark.slow
def test_single_resolution_misses_thin_segments(detected):
    """The reason multi-resolution exists, pinned as a fact rather than a claim."""
    scenario, *_ = detected
    attempts = [a for o in scenario.orders for a in o.attempts]
    single = score(scenario, Detector(DetectorConfig()).run(attempts))
    assert single.recall < 0.75


@pytest.mark.slow
def test_multi_resolution_finds_everything_on_this_seed(detected):
    _, _, _, report = detected
    assert report.recall == 1.0
    assert report.precision == 1.0


@pytest.mark.slow
def test_correlation_collapses_signals_to_incident_count(detected):
    scenario, signals, clusters, _ = detected
    assert len(signals) > 3 * len(scenario.incidents), "many slices should breach"
    assert len(clusters) == len(scenario.incidents)


@pytest.mark.slow
def test_correlation_elects_the_correct_scope(detected):
    """Dilution, not breadth, decides scope. A localized fault must not be
    reported globally, and a global one must not be pinned on a slice."""
    scenario, _, _, report = detected
    assert all(scope_is_exact(i, report) for i in scenario.incidents)


@pytest.mark.slow
def test_localized_fault_beats_the_global_reading(detected):
    """The acquirer incident must be named on the acquirer, despite the global
    segment covering strictly more traffic."""
    scenario, _, clusters, _ = detected
    acq = next(i for i in scenario.incidents if i.segment.acquirer)
    hit = next(c for c in clusters if c.primary.segment.value == acq.segment.acquirer)
    assert hit.primary.segment.dimension == "acquirer"
    assert hit.supporting, "the global and rail views should survive as evidence"


@pytest.mark.slow
def test_all_traffic_incident_is_not_pinned_to_a_slice(detected):
    scenario, _, clusters, _ = detected
    nsf = next(i for i in scenario.incidents if i.root_cause.value == "insufficient_funds_cluster")
    hit = next(c for c in clusters if c.primary.dominant_decline == nsf.dominant_decline)
    assert hit.primary.segment.dimension == "global"


@pytest.mark.slow
def test_cluster_money_does_not_double_count(detected):
    """Failed attempts appear in every slice they belong to; summing children
    would inflate the headline number several-fold."""
    _, _, clusters, _ = detected
    for c in clusters:
        assert c.money_at_risk == c.primary.money_at_risk


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 3, 5, 9])
def test_detection_holds_up_across_seeds(seed):
    """Guards against thresholds tuned to one scenario."""
    scenario = generate(SimConfig(seed=seed))
    attempts = [a for o in scenario.orders for a in o.attempts]
    report = score(scenario, [c.primary for c in correlate(MultiResolutionDetector().run(attempts))])
    assert report.recall >= 0.80
    assert report.precision >= 0.70
