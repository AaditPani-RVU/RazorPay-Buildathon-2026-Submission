"""Subscription recovery: lifecycle risk, mandate rules, mandate planning.

Mandates fail differently from one-off payments. The money is recurring, so a
lapse compounds; the failure is usually a *state* rather than an event, so
there is no anomaly to detect; and one of the states -- revoked -- is a
customer decision that recovery must not try to reverse on its own.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backstop.decide.planner import MANDATE_PLAYBOOK, tail_actions
from backstop.detect.mandates import REREGISTRATION_ODDS, scan
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    Customer,
    MandateStatus,
    Order,
    PaymentAttempt,
    Subscription,
)
from backstop.domain.money import Money
from backstop.policy.engine import Disposition, PolicyContext, PolicyEngine
from backstop.simulate.generator import SimConfig, generate

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def sub(status: MandateStatus, amount=499, cancelled=None) -> Subscription:
    return Subscription(
        id="sub_1", customer_id="cust_1", amount=Money.rupees(amount),
        rail=Rail.EMANDATE_NACH, mandate_status=status, next_charge_at=T0,
        cancelled_at=cancelled,
    )


def mandate_order(code: DeclineCode, amount=499) -> Order:
    amt = Money.rupees(amount)
    o = Order(id="order_m", customer_id="cust_1", amount=amt, created_at=T0)
    o.attempts.append(
        PaymentAttempt(
            id="pay_m", order_id=o.id, customer_id="cust_1", amount=amt,
            rail=Rail.EMANDATE_NACH, at=T0, status=AttemptStatus.FAILED,
            decline_code=code,
        )
    )
    return o


@pytest.fixture(scope="module")
def scenario():
    return generate(SimConfig(seed=1, days=7, orders_per_day=8000))


# -- generation ------------------------------------------------------------


def test_subscriptions_actually_present_charges(scenario):
    """Before this existed, subscriptions were a data model with no pipeline."""
    presentations = [
        a for o in scenario.orders for a in o.attempts
        if a.rail in (Rail.EMANDATE_NACH, Rail.UPI_AUTOPAY)
    ]
    assert len(presentations) == len(scenario.subscriptions)


def test_every_presentation_maps_back_to_its_subscription(scenario):
    for order_id, sub_id in scenario.subscription_by_order.items():
        assert any(s.id == sub_id for s in scenario.subscriptions)
        assert any(o.id == order_id for o in scenario.orders)


def test_a_lapsed_mandate_never_collects(scenario):
    """No rate to model: a revoked mandate does not sometimes work."""
    by_id = {s.id: s for s in scenario.subscriptions}
    for order_id, sub_id in scenario.subscription_by_order.items():
        s = by_id[sub_id]
        if s.mandate_status is MandateStatus.ACTIVE:
            continue
        order = next(o for o in scenario.orders if o.id == order_id)
        assert not order.is_captured


def test_lapsed_mandates_carry_a_matching_decline_code(scenario):
    expected = {
        MandateStatus.EXPIRED: DeclineCode.MANDATE_EXPIRED,
        MandateStatus.REVOKED: DeclineCode.MANDATE_REVOKED,
        MandateStatus.PAUSED: DeclineCode.MANDATE_PAUSED,
        MandateStatus.NOT_REGISTERED: DeclineCode.MANDATE_NOT_REGISTERED,
    }
    by_id = {s.id: s for s in scenario.subscriptions}
    for order_id, sub_id in scenario.subscription_by_order.items():
        s = by_id[sub_id]
        if s.mandate_status is MandateStatus.ACTIVE:
            continue
        order = next(o for o in scenario.orders if o.id == order_id)
        assert order.last_decline is expected[s.mandate_status]


# -- the lifecycle scan ----------------------------------------------------


def test_scan_prices_lapsed_mandates_over_a_year():
    report = scan([sub(MandateStatus.EXPIRED, amount=499)])
    assert len(report.at_risk) == 1
    assert report.at_risk[0].annual_value == Money.rupees(499 * 12)


def test_scan_ignores_active_mandates():
    report = scan([sub(MandateStatus.ACTIVE), sub(MandateStatus.EXPIRED)])
    assert report.active == 1
    assert len(report.at_risk) == 1


def test_scan_ignores_cancelled_subscriptions():
    """A cancellation is not revenue at risk. There is nothing to recover."""
    report = scan([sub(MandateStatus.EXPIRED, cancelled=T0)])
    assert report.at_risk == []


def test_scan_ranks_by_value():
    subs = [sub(MandateStatus.EXPIRED, amount=a) for a in (199, 2999, 799)]
    for i, s in enumerate(subs):
        s.id = f"sub_{i}"
    ranked = scan(subs).at_risk
    assert [r.charge_amount.as_rupees for r in ranked] == [2999, 799, 199]


def test_revoked_mandates_are_scored_as_least_recoverable():
    """Somebody who cancelled a mandate mostly meant to."""
    assert REREGISTRATION_ODDS[MandateStatus.REVOKED] < min(
        v for k, v in REREGISTRATION_ODDS.items() if k is not MandateStatus.REVOKED
    )


def test_scan_finds_real_lapsed_mandates_in_a_batch(scenario):
    report = scan(scenario.subscriptions)
    assert report.at_risk
    assert report.total_annual_value.paise > 0
    assert report.active + len(report.at_risk) == len(scenario.subscriptions)


# -- policy ----------------------------------------------------------------


def test_a_revoked_mandate_is_never_auto_chased():
    engine = PolicyEngine()
    action = Action(type=ActionType.REQUEST_MANDATE_REREGISTRATION,
                    subject_id="sub_1", scheduled_at=T0 + timedelta(hours=4),
                    channel=Channel.EMAIL, rationale="ask them to re-authorise")
    ctx = PolicyContext(now=T0, subscription=sub(MandateStatus.REVOKED),
                        customer=Customer(id="cust_1", email="a@b.test",
                                          consented_channels={Channel.EMAIL}))
    ruling = engine.evaluate(action, ctx)
    assert ruling.disposition is Disposition.DENY
    assert ruling.blocking_rule == "revoked_mandate"


def test_a_revoked_mandate_is_caught_by_decline_code_alone():
    """The rule must hold even without subscription context in scope."""
    engine = PolicyEngine()
    order = mandate_order(DeclineCode.MANDATE_REVOKED, amount=2999)
    action = Action(type=ActionType.SEND_DUNNING, subject_id=order.id,
                    scheduled_at=T0 + timedelta(hours=4), channel=Channel.EMAIL,
                    rationale="chase it")
    ctx = PolicyContext(now=T0, order=order,
                        customer=Customer(id="cust_1", email="a@b.test",
                                          consented_channels={Channel.EMAIL}))
    assert engine.evaluate(action, ctx).blocking_rule == "revoked_mandate"


def test_escalating_a_revoked_mandate_to_a_person_is_allowed():
    """The rule refuses automation, not the decision itself."""
    engine = PolicyEngine()
    action = Action(type=ActionType.ESCALATE_TO_HUMAN, subject_id="sub_1",
                    scheduled_at=T0, rationale="a person should decide")
    ctx = PolicyContext(now=T0, subscription=sub(MandateStatus.REVOKED))
    assert engine.evaluate(action, ctx).allowed


def test_a_cancelled_subscription_is_never_collected_on():
    engine = PolicyEngine()
    action = Action(type=ActionType.RETRY_PAYMENT, subject_id="sub_1",
                    scheduled_at=T0 + timedelta(hours=4), rationale="unpaid")
    ctx = PolicyContext(now=T0, subscription=sub(MandateStatus.ACTIVE, cancelled=T0))
    ruling = engine.evaluate(action, ctx)
    assert ruling.disposition is Disposition.DENY
    assert ruling.blocking_rule == "cancelled_subscription"


@pytest.mark.parametrize(
    "code", [DeclineCode.MANDATE_EXPIRED, DeclineCode.MANDATE_REVOKED,
             DeclineCode.MANDATE_NOT_REGISTERED]
)
def test_lapsed_mandates_are_never_re_presented(code):
    engine = PolicyEngine()
    order = mandate_order(code, amount=2999)
    action = Action(type=ActionType.RETRY_PAYMENT, subject_id=order.id,
                    scheduled_at=T0 + timedelta(days=1), rationale="try again")
    assert not engine.evaluate(action, PolicyContext(now=T0, order=order)).allowed


# -- planning --------------------------------------------------------------


def test_an_expired_mandate_is_asked_to_re_register_not_merely_dunned():
    actions = tail_actions(mandate_order(DeclineCode.MANDATE_EXPIRED))
    assert actions
    assert all(a.type is ActionType.REQUEST_MANDATE_REREGISTRATION for a in actions)


def test_a_revoked_mandate_goes_to_a_person():
    actions = tail_actions(mandate_order(DeclineCode.MANDATE_REVOKED))
    assert actions
    assert all(a.type is ActionType.ESCALATE_TO_HUMAN for a in actions)
    assert not any(a.is_contact for a in actions)


def test_a_paused_mandate_is_waited_out_rather_than_chased():
    actions = tail_actions(mandate_order(DeclineCode.MANDATE_PAUSED))
    assert actions
    assert all(a.type is ActionType.WAIT for a in actions)


def test_the_mandate_playbook_never_proposes_a_charge():
    """Every lapsed state needs the customer or a person, never a re-present."""
    for code in MANDATE_PLAYBOOK:
        for action in tail_actions(mandate_order(code)):
            assert not action.is_charging
