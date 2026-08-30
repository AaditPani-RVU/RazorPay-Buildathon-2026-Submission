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

from backstop.evaluation.backtest import run
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
    scenario, arms, _ = runs[seed]
    at_risk = sum(o.amount_at_risk.paise for o in scenario.orders)
    for arm in arms.values():
        assert arm.ledger.recovered.paise <= at_risk


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
def test_the_policed_arm_beats_the_naive_baseline_on_net(seed, runs):
    _, arms, _ = runs[seed]
    assert arms["backstop"].ledger.net > arms["naive-retry"].ledger.net


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
