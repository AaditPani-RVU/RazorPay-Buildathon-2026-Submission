"""The recurring surface: what recovering a mandate is allowed to claim.

Recurring revenue is the easiest place in the whole system to report a number
that is not true, and these tests are aimed at each of the ways.

The first is unit confusion -- crediting a restored mandate at one charge, or
worse, adding a year of billing to a batch of one-off payments and printing
the sum. The second is the counterfactual: a paused mandate that would have
resumed on its own is not recovery, and an agent that mails those customers
and books the resumption has measured its own postage. The third is the one
the policy engine exists for -- a revoked mandate is a decision, and re-asking
is not a growth tactic.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backstop.decide.planner import mandate_actions, naive_mandate_chase
from backstop.domain.actions import ActionType
from backstop.domain.declines import Rail
from backstop.domain.entities import (
    BILLING_PERIODS_PER_YEAR,
    Channel,
    Customer,
    MandateStatus,
    Subscription,
)
from backstop.domain.money import Money
from backstop.evaluation.backtest import run
from backstop.execute.executor import Outcome, SimulatedExecutor
from backstop.ledger.ledger import Surface
from backstop.policy.engine import Disposition, PolicyContext, PolicyEngine
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.recoverability import MandateRecovery

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def sub(status=MandateStatus.EXPIRED, *, amount=499, cancelled=None, sid="sub_1"):
    return Subscription(
        id=sid,
        customer_id="cust_1",
        amount=Money.rupees(amount),
        rail=Rail.UPI_AUTOPAY,
        mandate_status=status,
        next_charge_at=NOW,
        cancelled_at=cancelled,
    )


def truth(*, would=True, unprompted=False, days=30.0, sid="sub_1"):
    return MandateRecovery(
        subscription_id=sid,
        would_reregister=would,
        resumes_unprompted=unprompted,
        responsive_until=NOW + timedelta(days=days) if would else None,
    )


def executor(s, rec):
    return SimulatedExecutor(
        orders={}, recoverability={},
        subscriptions={s.id: s}, mandate_recovery={s.id: rec},
    )


# -- what a restored mandate is worth --------------------------------------


def test_a_restored_mandate_is_credited_a_year_of_billing():
    """One charge would understate a stream by an order of magnitude."""
    s = sub(amount=499)
    ex = executor(s, truth())
    result = ex.execute(mandate_actions(s, NOW)[0], NOW + timedelta(days=1))
    assert result.outcome is Outcome.RECOVERED
    assert result.recovered == Money.rupees(499) * BILLING_PERIODS_PER_YEAR


def test_the_scan_the_engine_and_the_ledger_price_a_mandate_identically():
    """Three components read the value of a mandate. They must agree, or the
    policy engine is deciding about one number and the ledger reporting another."""
    from backstop.detect.mandates import scan

    s = sub(amount=1200)
    risk = scan([s]).at_risk[0]
    ctx = PolicyContext(now=NOW, subscription=s)
    ex = executor(s, truth())
    recovered = ex.execute(mandate_actions(s, NOW)[0], NOW + timedelta(days=1)).recovered

    assert risk.annual_value == s.annual_value
    assert ctx.subject_amount == s.annual_value
    assert recovered == s.annual_value


# -- the counterfactual ----------------------------------------------------


def test_a_mandate_that_would_have_resumed_on_its_own_is_not_recovery():
    """The customer came back either way. Chasing them recovered a stamp."""
    s = sub(MandateStatus.PAUSED)
    ex = executor(s, truth(unprompted=True))
    result = ex.execute(mandate_actions(s, NOW)[0], NOW + timedelta(days=4))
    assert result.outcome is Outcome.NO_EFFECT
    assert not result.recovered
    assert result.cost, "the contact was still sent, so it still cost money"


def test_asking_after_the_customer_has_gone_cold_recovers_nothing():
    s = sub()
    ex = executor(s, truth(days=5.0))
    assert ex.execute(mandate_actions(s, NOW)[0], NOW + timedelta(days=40)).outcome \
        is Outcome.NO_EFFECT


def test_asking_twice_does_not_give_two_chances():
    """Re-registration is drawn once per customer. An arm that asks five times
    must not get five independent rolls, which is the error that makes
    aggressive dunning look better on paper than it is."""
    s = sub()
    ex = executor(s, truth())
    outcomes = [
        ex.execute(a, NOW + timedelta(days=1)).outcome
        for a in naive_mandate_chase([s], NOW, contacts=5)
    ]
    assert outcomes.count(Outcome.RECOVERED) == 1


def test_a_dead_mandate_cannot_be_charged():
    """There is no authorisation to present against, however good the card."""
    from backstop.domain.actions import Action

    s = sub()
    ex = executor(s, truth())
    charge = Action(
        type=ActionType.RETRY_PAYMENT, subject_id=s.id, scheduled_at=NOW,
        rationale="present against a lapsed mandate",
    )
    assert ex.execute(charge, NOW).outcome is Outcome.NO_EFFECT


# -- what the planner proposes ---------------------------------------------


def test_a_revoked_mandate_is_routed_to_a_person_not_re_asked():
    actions = mandate_actions(sub(MandateStatus.REVOKED), NOW)
    assert [a.type for a in actions] == [ActionType.ESCALATE_TO_HUMAN]


def test_nothing_is_proposed_for_a_cancelled_subscription():
    assert mandate_actions(sub(cancelled=NOW), NOW) == []


def test_nothing_is_proposed_for_a_healthy_mandate():
    assert mandate_actions(sub(MandateStatus.ACTIVE), NOW) == []


def test_a_paused_mandate_is_left_alone_before_it_is_chased():
    """Long enough that self-resumers resolve themselves, short enough that the
    rest are still listening."""
    actions = mandate_actions(sub(MandateStatus.PAUSED), NOW)
    assert len(actions) == 1
    delay = actions[0].scheduled_at - NOW
    assert timedelta(days=2) <= delay <= timedelta(days=5)


@pytest.mark.parametrize(
    "status", [MandateStatus.EXPIRED, MandateStatus.PAUSED, MandateStatus.NOT_REGISTERED]
)
def test_every_proposal_the_planner_makes_survives_the_policy_engine(status):
    """The recurring playbook proposes nothing the rules refuse.

    Not redundancy with the engine -- defence in depth. The engine is the
    guarantee; a planner that constantly proposes vetoed work is a planner
    that would be dangerous the moment somebody ran it unpoliced.
    """
    s = sub(status)
    customer = Customer(
        id="cust_1", email="a@example.test", phone="+919800000000",
        consented_channels={Channel.EMAIL},
    )
    engine = PolicyEngine()
    for action in mandate_actions(s, NOW):
        ruling = engine.evaluate(action, PolicyContext(now=NOW, customer=customer, subscription=s))
        assert ruling.disposition is not Disposition.DENY, ruling.describe()


# -- end to end ------------------------------------------------------------


@pytest.fixture(scope="module")
def arms():
    scenario = generate(SimConfig(seed=7, days=3, orders_per_day=3000))
    result, _ = run(scenario, offline=True, model=None)
    return scenario, {a.name: a for a in result}


def test_the_recurring_surface_is_actually_measured(arms):
    _, by_name = arms
    assert by_name["backstop"].ledger.on(Surface.RECURRING).recovered


def test_the_policed_arm_breaks_no_recurring_rules(arms):
    _, by_name = arms
    recurring = [v for v in by_name["backstop"].violations if v.surface is Surface.RECURRING]
    assert recurring == []


def test_the_naive_arm_does_break_recurring_rules(arms):
    """Chasing every dead mandate re-asks people who revoked on purpose."""
    _, by_name = arms
    reasons = {
        v.rule_id for v in by_name["naive-retry"].violations if v.surface is Surface.RECURRING
    }
    assert "revoked_mandate" in reasons


def test_the_policed_arm_reaches_the_same_book_with_far_fewer_contacts(arms):
    _, by_name = arms
    policed = by_name["backstop"].ledger.on(Surface.RECURRING)
    naive = by_name["naive-retry"].ledger.on(Surface.RECURRING)
    assert policed.contacts_sent < naive.contacts_sent / 2


def test_no_mandate_is_recovered_twice(arms):
    _, by_name = arms
    for arm in by_name.values():
        won = [
            e.action.subject_id for e in arm.ledger.on(Surface.RECURRING).executed
            if e.execution.is_recovery
        ]
        assert len(won) == len(set(won))


def test_policing_the_recurring_surface_costs_no_revenue_at_all(arms):
    """The README's central recurring claim, as an executable one.

    Strip from the naive arm the money it took with actions the rules refuse,
    and what remains is exactly what the policed arm recovered -- not close to
    it, equal to it. Re-registration is drawn once per customer, so asking
    three times buys no extra chances; both arms reach every mandate they are
    allowed to reach, and naive's remaining contacts are pure waste. If this
    ever stops holding, the claim that the leash is free on this surface has
    stopped being true and the README needs to say something else.
    """
    _, by_name = arms
    naive = by_name["naive-retry"].ledger.on(Surface.RECURRING)
    policed = by_name["backstop"].ledger.on(Surface.RECURRING)

    assert naive.compliant_recovered == policed.recovered
    assert naive.orders_recovered > policed.orders_recovered, (
        "naive should still recover more mandates -- the illegal ones"
    )
    assert policed.compliant_net > naive.compliant_net, (
        "identical revenue for fewer contacts means the policed arm nets more"
    )


def test_recurring_is_never_folded_into_the_payments_number(arms):
    """The two surfaces are different units. A reader who takes the payments
    row must not be reading a total that has a year of billing hidden in it."""
    _, by_name = arms
    led = by_name["backstop"].ledger
    payments = led.on(Surface.PAYMENT)
    recurring = led.on(Surface.RECURRING)
    assert payments.recovered + recurring.recovered == led.recovered
    assert recurring.recovered
    assert payments.recovered != led.recovered
