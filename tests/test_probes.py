"""The bench: a fixed set of actions the engine must keep ruling on the same way.

The demo and the console both run this, and both of them only *display* the
result. That is not an assertion, so this is: if a rule change quietly turns a
denial into a permission, the failure should land in the test suite rather than
in a paragraph of terminal output nobody reads closely.

Offline: a generated batch, no model, no network.
"""

import pytest

from backstop.policy.engine import Disposition, PolicyEngine
from backstop.policy.probes import bench, ist, pick
from backstop.simulate.generator import SimConfig, generate

SEEDS = [1, 2, 3]


@pytest.fixture(scope="module")
def scenarios():
    return {seed: generate(SimConfig(seed=seed, days=7, orders_per_day=8000))
            for seed in SEEDS}


@pytest.mark.parametrize("seed", SEEDS)
def test_every_probe_is_ruled_on_as_expected(scenarios, seed):
    engine = PolicyEngine()
    surprises = []
    for probe in bench(scenarios[seed]):
        ruling = engine.evaluate(probe.action, probe.context)
        if ruling.disposition is not probe.expect:
            surprises.append(
                f"{probe.label}: expected {probe.expected}, "
                f"got {ruling.disposition.value.upper()}"
            )
    assert not surprises, "\n".join(surprises)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_bench_covers_both_halves_of_the_argument(scenarios, seed):
    """Things that must never happen, and things permitted on the system's
    terms. A bench of only denials would prove the engine can say no."""
    expected = {p.expect for p in bench(scenarios[seed])}
    assert Disposition.DENY in expected
    assert Disposition.ALLOW in expected
    assert Disposition.RESCHEDULE in expected
    assert Disposition.REQUIRE_APPROVAL in expected


def test_every_probe_names_a_real_subject_from_the_batch(scenarios):
    """A probe against a subject the batch does not contain would be testing
    the rule against a fixture rather than against the world."""
    scenario = scenarios[1]
    known = (
        {o.id for o in scenario.orders}
        | {i.id for i in scenario.invoices}
        | {s.id for s in scenario.subscriptions}
    )
    for probe in bench(scenario):
        assert probe.action.subject_id in known


def test_every_denial_names_the_rule_that_produced_it(scenarios):
    engine = PolicyEngine()
    for probe in bench(scenarios[1]):
        ruling = engine.evaluate(probe.action, probe.context)
        if ruling.disposition is Disposition.DENY:
            assert ruling.blocking_rule


def test_ist_lands_at_the_wall_clock_hour_it_was_asked_for(scenarios):
    from zoneinfo import ZoneInfo

    base = scenarios[1].ends_at
    assert ist(base, 3).astimezone(ZoneInfo("Asia/Kolkata")).hour == 3
    assert ist(base, 21, 30).astimezone(ZoneInfo("Asia/Kolkata")).hour == 21


def test_pick_returns_nothing_rather_than_the_wrong_order(scenarios):
    from backstop.domain.declines import DeclineCode
    from backstop.domain.money import Money

    scenario = scenarios[1]
    found = pick(scenario.failed_orders, DeclineCode.STOLEN_OR_LOST_CARD)
    assert found is None or found.last_decline is DeclineCode.STOLEN_OR_LOST_CARD
    huge = pick(
        scenario.failed_orders, DeclineCode.INSUFFICIENT_FUNDS,
        above=Money.rupees(10_000_000),
    )
    assert huge is None
