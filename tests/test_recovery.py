"""The execution path: latent outcomes, the executor, the ledger, the planner.

The invariant these protect is that recovery is *measured*, not asserted. If
the executor can be talked into recovering money twice, or the ledger can
undercount what an arm did, every number the backtest reports is worthless.
"""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from backstop.decide.planner import naive_retry, tail_actions
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail, RetryClass
from backstop.domain.entities import AttemptStatus, Channel, Order, PaymentAttempt
from backstop.domain.money import Money
from backstop.execute.executor import Outcome, SimulatedExecutor
from backstop.ledger.ledger import LedgerEntry, RecoveryLedger
from backstop.policy.engine import PolicyContext, PolicyEngine
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.recoverability import Recoverability

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
DEFAULT_AMOUNT = Money.rupees(1000)


def make_order(code: DeclineCode, amount: Money = DEFAULT_AMOUNT) -> Order:
    o = Order(id="order_x", customer_id="cust_x", amount=amount, created_at=T0)
    o.attempts.append(
        PaymentAttempt(
            id="pay_x", order_id=o.id, customer_id="cust_x", amount=amount,
            rail=Rail.CARD, at=T0, status=AttemptStatus.FAILED, decline_code=code,
        )
    )
    return o


def executor_for(order: Order, rec: Recoverability) -> SimulatedExecutor:
    return SimulatedExecutor(orders={order.id: order}, recoverability={order.id: rec})


def retry(at: datetime, subject="order_x") -> Action:
    return Action(type=ActionType.RETRY_PAYMENT, subject_id=subject,
                  scheduled_at=at, rationale="test")


def dun(at: datetime, subject="order_x") -> Action:
    return Action(type=ActionType.SEND_DUNNING, subject_id=subject, scheduled_at=at,
                  channel=Channel.EMAIL, rationale="test")


# -- the latent model ------------------------------------------------------


def test_recoverability_is_deterministic_for_a_seed():
    a = generate(SimConfig(seed=5, days=2, orders_per_day=1500)).recoverability
    b = generate(SimConfig(seed=5, days=2, orders_per_day=1500)).recoverability
    assert a == b


def test_every_failed_order_has_a_latent_outcome():
    s = generate(SimConfig(seed=6, days=2, orders_per_day=1500))
    for order in s.failed_orders:
        assert order.id in s.recoverability


def test_fraud_blocks_are_never_recoverable():
    s = generate(SimConfig(seed=7, days=3, orders_per_day=2000))
    fraud = [
        o for o in s.failed_orders
        if o.last_decline and o.last_decline.retry_class is RetryClass.FRAUD_BLOCK
    ]
    assert fraud, "seed produced no fraud declines to check"
    for o in fraud:
        rec = s.recoverability[o.id]
        assert not rec.retry_would_succeed
        assert not rec.dunning_would_convert


@pytest.mark.parametrize(
    "klass", [RetryClass.USER_ACTION_REQUIRED, RetryClass.HARD_DECLINE]
)
def test_unretryable_classes_never_heal_into_a_retry(klass):
    s = generate(SimConfig(seed=8, days=3, orders_per_day=2000))
    for o in s.failed_orders:
        if o.last_decline and o.last_decline.retry_class is klass:
            assert not s.recoverability[o.id].retry_would_succeed


# -- the executor ----------------------------------------------------------


def test_a_retry_before_the_blocker_clears_does_not_recover():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    rec = Recoverability("order_x", T0 + timedelta(days=1), True, False, None, False)
    ex = executor_for(order, rec)
    early = ex.execute(retry(T0 + timedelta(hours=1)), T0 + timedelta(hours=1))
    assert early.outcome is Outcome.NO_EFFECT
    assert not early.recovered


def test_the_same_retry_after_the_blocker_clears_does_recover():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    rec = Recoverability("order_x", T0 + timedelta(days=1), True, False, None, False)
    ex = executor_for(order, rec)
    at = T0 + timedelta(days=2)
    late = ex.execute(retry(at), at)
    assert late.outcome is Outcome.RECOVERED
    assert late.recovered == order.amount


def test_money_is_never_recovered_twice():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    rec = Recoverability("order_x", T0, True, False, None, False)
    ex = executor_for(order, rec)
    first = ex.execute(retry(T0 + timedelta(hours=1)), T0 + timedelta(hours=1))
    second = ex.execute(retry(T0 + timedelta(hours=2)), T0 + timedelta(hours=2))
    assert first.outcome is Outcome.RECOVERED
    assert second.outcome is Outcome.NO_EFFECT
    assert not second.recovered


def test_chasing_a_settled_order_still_costs_money():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    rec = Recoverability("order_x", T0, True, False, None, False)
    ex = executor_for(order, rec)
    ex.execute(retry(T0 + timedelta(hours=1)), T0 + timedelta(hours=1))
    wasted = ex.execute(dun(T0 + timedelta(hours=2)), T0 + timedelta(hours=2))
    assert wasted.outcome is Outcome.NO_EFFECT
    assert wasted.cost


def test_late_dunning_misses_a_customer_who_moved_on():
    order = make_order(DeclineCode.CARD_EXPIRED)
    deadline = T0 + timedelta(hours=12)
    rec = Recoverability("order_x", None, False, True, deadline, False)
    ex = executor_for(order, rec)
    late = ex.execute(dun(T0 + timedelta(hours=48)), T0 + timedelta(hours=48))
    assert late.outcome is Outcome.NO_EFFECT


def test_prompt_dunning_reaches_the_same_customer():
    order = make_order(DeclineCode.CARD_EXPIRED)
    deadline = T0 + timedelta(hours=12)
    rec = Recoverability("order_x", None, False, True, deadline, False)
    ex = executor_for(order, rec)
    at = T0 + timedelta(hours=2)
    assert ex.execute(dun(at), at).outcome is Outcome.RECOVERED


def test_inert_actions_cost_nothing_and_recover_nothing():
    order = make_order(DeclineCode.STOLEN_OR_LOST_CARD)
    rec = Recoverability("order_x", None, False, False, None, False)
    ex = executor_for(order, rec)
    action = Action(type=ActionType.ESCALATE_TO_RISK, subject_id="order_x",
                    scheduled_at=T0, rationale="test")
    result = ex.execute(action, T0)
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.cost and not result.recovered


# -- the ledger ------------------------------------------------------------


def test_ledger_arithmetic_holds():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    rec = Recoverability("order_x", T0, True, False, None, False)
    ex = executor_for(order, rec)
    led = RecoveryLedger(arm="t")
    for hours in (1, 2, 3):
        a = retry(T0 + timedelta(hours=hours))
        led.record(LedgerEntry(action=a, execution=ex.execute(a, a.scheduled_at)))
    assert led.orders_recovered == 1
    assert led.recovered == order.amount
    assert led.net == led.recovered - led.cost
    assert led.wasted_actions == 2


def test_ledger_counts_contacts_per_subject():
    order = make_order(DeclineCode.CARD_EXPIRED)
    rec = Recoverability("order_x", None, False, False, None, False)
    ex = executor_for(order, rec)
    led = RecoveryLedger(arm="t")
    for hours in (1, 2, 3, 4):
        a = dun(T0 + timedelta(hours=hours))
        led.record(LedgerEntry(action=a, execution=ex.execute(a, a.scheduled_at)))
    assert led.contacts_sent == 4
    assert led.worst_contact_burst == 4


def test_audit_finds_actions_that_ran_but_should_not_have():
    order = make_order(DeclineCode.STOLEN_OR_LOST_CARD)
    rec = Recoverability("order_x", None, False, False, None, False)
    ex = executor_for(order, rec)
    led = RecoveryLedger(arm="unpoliced")
    a = retry(T0 + timedelta(hours=1))
    led.record(LedgerEntry(action=a, execution=ex.execute(a, a.scheduled_at)))

    violations = led.audit(
        PolicyEngine(), lambda action: PolicyContext(now=T0, order=order)
    )
    assert len(violations) == 1
    assert violations[0].rule_id == "fraud_block"


# -- the planner -----------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [c for c in DeclineCode if not c.spec.is_retryable and c.retry_class
     is not RetryClass.FRAUD_BLOCK],
)
def test_tail_playbook_never_retries_what_cannot_be_retried(code):
    order = make_order(code)
    for action in tail_actions(order):
        assert not action.is_charging


def test_tail_playbook_sends_fraud_to_risk_and_never_contacts():
    order = make_order(DeclineCode.STOLEN_OR_LOST_CARD)
    actions = tail_actions(order)
    assert actions and all(a.type is ActionType.ESCALATE_TO_RISK for a in actions)
    assert not any(a.is_contact for a in actions)


def test_tail_playbook_spaces_repeat_attempts_apart():
    order = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    actions = sorted(tail_actions(order), key=lambda a: a.scheduled_at)
    assert len(actions) >= 2
    gaps = [
        (b.scheduled_at - a.scheduled_at).total_seconds()
        for a, b in pairwise(actions)
    ]
    assert all(g > 0 for g in gaps)


def test_naive_baseline_really_is_indiscriminate():
    """The baseline has to be the obvious implementation, or beating it is
    meaningless. It must retry things that cannot be retried."""
    order = make_order(DeclineCode.CARD_EXPIRED)
    actions = naive_retry([order])
    assert any(a.is_charging for a in actions)
    assert any(a.is_contact for a in actions)
