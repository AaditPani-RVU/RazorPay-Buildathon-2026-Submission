"""The backtest's own guarantees.

The headline claim is "recovers more than the naive baseline with zero policy
violations". These tests are what stop that claim quietly becoming false --
the violation count is checked as an invariant across seeds rather than
observed once and written into a slide.

Everything here runs offline. The tail playbook drives the Backstop arm, which
is enough to exercise the enforcement path; the LLM planner is scored
separately by the demo and the backtest's own reporting.
"""

import pytest

from backstop.domain.entities import utc
from backstop.evaluation.backtest import run
from backstop.ledger.ledger import Surface
from backstop.policy.engine import Disposition
from backstop.simulate.generator import SimConfig, generate

SEEDS = [1, 2, 3]


@pytest.fixture(scope="module")
def runs():
    out = {}
    for seed in SEEDS:
        scenario = generate(SimConfig(seed=seed, days=3, orders_per_day=4000))
        arms, prop = run(scenario, offline=True, model=None)
        out[seed] = (scenario, {a.name: a for a in arms}, prop)
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_the_policed_arm_never_executes_a_violation(seed, runs):
    """The invariant the whole design exists to provide."""
    _, arms, _ = runs[seed]
    assert arms["backstop"].violations == []


@pytest.mark.parametrize("seed", SEEDS)
def test_the_naive_baseline_does_break_rules(seed, runs):
    """If the baseline were already compliant there would be nothing to prove."""
    _, arms, _ = runs[seed]
    assert len(arms["naive-retry"].violations) > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_doing_nothing_recovers_nothing(seed, runs):
    _, arms, _ = runs[seed]
    led = arms["do-nothing"].ledger
    assert not led.recovered
    assert not led.cost
    assert led.proposed == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_recovery_never_exceeds_the_money_that_was_at_risk(seed, runs):
    """Bounded per surface, not in aggregate.

    The surfaces are denominated differently -- a payment is one amount that
    did not land, a mandate is a year of billing that stopped -- so a single
    combined ceiling would be satisfied by a number that means nothing. Each
    surface is held to its own.
    """
    scenario, arms, _ = runs[seed]
    payments_at_risk = sum(o.amount_at_risk.paise for o in scenario.orders)
    recurring_at_risk = scenario.recurring_at_risk.paise
    for arm in arms.values():
        assert arm.ledger.on(Surface.PAYMENT).recovered.paise <= payments_at_risk
        assert arm.ledger.on(Surface.RECURRING).recovered.paise <= recurring_at_risk


@pytest.mark.parametrize("seed", SEEDS)
def test_no_order_is_recovered_more_than_once(seed, runs):
    _, arms, _ = runs[seed]
    for arm in arms.values():
        recovered = [
            e.action.subject_id for e in arm.ledger.executed
            if e.execution.is_recovery
        ]
        assert len(recovered) == len(set(recovered))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_policed_arm_respects_the_contact_cap(seed, runs):
    _, arms, _ = runs[seed]
    from backstop.policy.engine import PolicyConfig

    cap = PolicyConfig().max_contacts_per_subject
    assert arms["backstop"].ledger.worst_contact_burst <= cap


@pytest.mark.parametrize("seed", SEEDS)
def test_enforcement_only_ever_removes_actions(seed, runs):
    """Policing is subtractive. It may veto or delay, never invent work."""
    _, arms, _ = runs[seed]
    policed, loose = arms["backstop"], arms["planner-unpoliced"]
    assert len(policed.ledger.executed) <= len(loose.ledger.executed)
    assert policed.ledger.contacts_sent <= loose.ledger.contacts_sent


@pytest.mark.parametrize("seed", SEEDS)
def test_both_planner_arms_propose_the_same_actions(seed, runs):
    """The controlled comparison is only controlled if the input is identical."""
    _, arms, _ = runs[seed]
    assert arms["backstop"].ledger.proposed == arms["planner-unpoliced"].ledger.proposed


@pytest.mark.parametrize("seed", SEEDS)
def test_the_policed_arm_beats_the_naive_baseline_on_keepable_revenue(seed, runs):
    """Gross recovery is the wrong comparison and this test used to make it.

    The naive arm recovers money by chasing revoked mandates and contacting
    people it may not contact. A merchant cannot keep that revenue, so
    crediting it would score the baseline for the exact behaviour the policy
    engine exists to stop. The comparison is on what survives the rules.
    """
    _, arms, _ = runs[seed]
    assert arms["backstop"].ledger.compliant_net > arms["naive-retry"].ledger.compliant_net


@pytest.mark.parametrize("seed", SEEDS)
def test_the_policed_arm_keeps_everything_it_recovers(seed, runs):
    _, arms, _ = runs[seed]
    led = arms["backstop"].ledger
    assert not led.recovered_in_violation
    assert led.compliant_recovered == led.recovered


@pytest.mark.parametrize("seed", SEEDS)
def test_the_naive_baseline_recovers_money_it_cannot_keep(seed, runs):
    _, arms, _ = runs[seed]
    assert arms["naive-retry"].ledger.recovered_in_violation


@pytest.mark.parametrize("seed", SEEDS)
def test_the_policed_arm_wastes_less_effort(seed, runs):
    _, arms, _ = runs[seed]
    assert arms["backstop"].ledger.wasted_actions < arms["naive-retry"].ledger.wasted_actions


@pytest.mark.parametrize("seed", SEEDS)
def test_every_executed_action_is_recorded_with_a_ruling(seed, runs):
    """An audit trail with gaps is not an audit trail."""
    _, arms, _ = runs[seed]
    for name in ("backstop", "planner-unpoliced", "naive-retry"):
        for entry in arms[name].ledger.executed:
            assert entry.ruling is not None
            assert entry.execution is not None


@pytest.mark.parametrize("seed", SEEDS)
def test_a_rescheduled_action_is_executed_later_rather_than_dropped(seed, runs):
    """A move is not a refusal, and this arm used to treat it as one.

    Three rules exist mainly to shift an action's moment -- quiet hours, retry
    spacing, and the outage hold. The enforced arm skipped everything that was
    not `allowed`, and `allowed` did not include `RESCHEDULE`, so every action
    those three rules touched was silently dropped instead of being executed
    at the hour they chose. That made them denials wearing a different label,
    and understated the policed arm against its own design.
    """
    _, arms, _ = runs[seed]
    policed = arms["backstop"]
    moved = [
        entry for entry in policed.ledger.executed
        if entry.ruling is not None
        and entry.ruling.disposition is Disposition.RESCHEDULE
    ]
    assert moved, "no action was rescheduled in this batch; the test proves nothing"
    for entry in moved:
        # Executed at the moment the rules chose, not the one proposed.
        assert utc(entry.execution.at) == utc(entry.action.scheduled_at)
        assert utc(entry.action.scheduled_at) != utc(entry.ruling.proposed.scheduled_at)
