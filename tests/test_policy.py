"""The policy engine is the safety boundary. These tests are the proof.

Every test here corresponds to something the system must never do, regardless
of what the detector, diagnoser or planner concluded.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail, RootCause
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    ContactRecord,
    Customer,
    Invoice,
    Order,
    PaymentAttempt,
    PromiseToPay,
)
from backstop.domain.money import Money
from backstop.policy.engine import (
    DEFAULT_RULES,
    Disposition,
    PolicyConfig,
    PolicyContext,
    PolicyEngine,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)  # 17:30 IST, comfortably awake
ENGINE = PolicyEngine()


def make_order(code, amount=2500, recoveries=0):
    o = Order(id="o", customer_id="c1", amount=Money.rupees(amount), created_at=NOW)
    o.attempts.append(PaymentAttempt(
        id="a", order_id="o", customer_id="c1", amount=Money.rupees(amount), rail=Rail.CARD,
        at=NOW, status=AttemptStatus.FAILED, decline_code=code, issuer="HDFC"))
    for i in range(recoveries):
        o.attempts.append(PaymentAttempt(
            id=f"r{i}", order_id="o", customer_id="c1", amount=Money.rupees(amount),
            rail=Rail.CARD, at=NOW + timedelta(days=i + 1), status=AttemptStatus.FAILED,
            decline_code=code, issuer="HDFC", is_recovery_attempt=True))
    return o


def customer(**kw):
    base = {"id": "c1", "email": "a@b.test", "phone": "+9198",
                "consented_channels": {Channel.EMAIL, Channel.SMS}}
    return Customer(**{**base, **kw})


def retry(at=NOW):
    return Action(type=ActionType.RETRY_PAYMENT, subject_id="o", scheduled_at=at, rationale="r")


def dunning(at=NOW, channel=Channel.SMS, subject="o"):
    return Action(type=ActionType.SEND_DUNNING, subject_id=subject, scheduled_at=at,
                  channel=channel, rationale="r")


def ctx(**kw):
    kw.setdefault("customer", customer())
    return PolicyContext(now=NOW, **kw)


# -- things that must never happen ---------------------------------------


def test_a_stolen_card_is_never_retried():
    r = ENGINE.evaluate(retry(), ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
    assert r.disposition is Disposition.DENY
    assert r.blocking_rule == "fraud_block"


def test_a_stolen_card_victim_is_never_contacted():
    """Dunning here is a harm, not a recovery."""
    r = ENGINE.evaluate(dunning(), ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
    assert r.disposition is Disposition.DENY


@pytest.mark.parametrize("code", [
    DeclineCode.CARD_EXPIRED, DeclineCode.INVALID_CARD_NUMBER,
    DeclineCode.RESTRICTED_CARD, DeclineCode.MANDATE_REVOKED, DeclineCode.INVALID_VPA,
])
def test_unrecoverable_declines_are_never_represented(code):
    assert ENGINE.evaluate(retry(), ctx(order=make_order(code))).disposition is Disposition.DENY


def test_a_disputed_invoice_is_never_chased():
    inv = Invoice(id="i", buyer_id="b", amount=Money.rupees(180000),
                  issued_at=NOW - timedelta(days=60), due_at=NOW - timedelta(days=30),
                  disputed_at=NOW - timedelta(days=5))
    r = ENGINE.evaluate(dunning(channel=Channel.EMAIL, subject="i"), ctx(invoice=inv))
    assert r.disposition is Disposition.DENY
    assert r.blocking_rule == "dispute_freeze"


def test_a_live_promise_to_pay_suspends_chasing():
    inv = Invoice(id="i", buyer_id="b", amount=Money.rupees(90000),
                  issued_at=NOW - timedelta(days=60), due_at=NOW - timedelta(days=30),
                  promise=PromiseToPay(promised_at=NOW, promised_for=NOW + timedelta(days=7),
                                       amount=Money.rupees(90000)))
    r = ENGINE.evaluate(dunning(channel=Channel.EMAIL, subject="i"), ctx(invoice=inv))
    assert r.blocking_rule == "promise_to_pay"


def test_dnd_registration_blocks_sms_but_not_email():
    c = customer(dnd_registered=True, consented_channels={Channel.SMS, Channel.EMAIL})
    o = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    assert ENGINE.evaluate(dunning(), PolicyContext(now=NOW, customer=c, order=o)).disposition \
        is Disposition.DENY
    assert ENGINE.evaluate(dunning(channel=Channel.EMAIL),
                           PolicyContext(now=NOW, customer=c, order=o)).allowed


def test_an_opt_out_blocks_every_channel():
    c = customer(opted_out_at=NOW - timedelta(days=1))
    for channel in (Channel.EMAIL, Channel.SMS, Channel.WHATSAPP):
        r = ENGINE.evaluate(dunning(channel=channel),
                            PolicyContext(now=NOW, customer=c,
                                          order=make_order(DeclineCode.INSUFFICIENT_FUNDS)))
        assert r.disposition is Disposition.DENY


# -- stopping rules ------------------------------------------------------


def test_retry_budget_is_a_hard_ceiling():
    spec = DeclineCode.INSUFFICIENT_FUNDS.spec
    o = make_order(DeclineCode.INSUFFICIENT_FUNDS, recoveries=spec.max_retries)
    r = ENGINE.evaluate(retry(NOW + timedelta(days=9)), ctx(order=o))
    assert r.blocking_rule == "retry_budget"


def test_contact_frequency_cap_stops_harassment():
    cfg = PolicyConfig()
    contacts = [ContactRecord(id=f"c{i}", customer_id="c1", channel=Channel.SMS,
                              at=NOW - timedelta(days=i + 1), subject_ref="o")
                for i in range(cfg.max_contacts_per_subject)]
    r = ENGINE.evaluate(dunning(), ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS),
                                       contacts=contacts))
    assert r.blocking_rule == "contact_frequency"


def test_chasing_below_the_economic_floor_is_refused():
    """Three messages to recover five rupees destroys value."""
    r = ENGINE.evaluate(dunning(), ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS, amount=5)))
    assert r.blocking_rule == "cost_of_recovery"


def test_diagnosis_can_forbid_retrying_a_technically_retryable_failure():
    """The instrument is fine; nobody is at the OTP screen. Only re-engagement
    can work, so charging must be blocked even where a budget remains."""
    o = make_order(DeclineCode.GATEWAY_TIMEOUT)
    r = ENGINE.evaluate(retry(NOW + timedelta(hours=2)),
                        ctx(order=o, diagnosis=RootCause.AUTHENTICATION_DROPOFF))
    assert r.disposition is Disposition.DENY
    assert r.blocking_rule == "diagnosis_gate"


# -- rescheduling --------------------------------------------------------


def test_retry_spacing_is_enforced_not_merely_suggested():
    r = ENGINE.evaluate(retry(), ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS)))
    assert r.disposition is Disposition.RESCHEDULE
    spec = DeclineCode.INSUFFICIENT_FUNDS.spec
    assert r.final.scheduled_at >= NOW + timedelta(seconds=spec.min_retry_delay_s)


def test_quiet_hours_move_an_sms_out_of_the_night():
    night = datetime(2026, 8, 30, 21, 30, tzinfo=UTC)  # 03:00 IST
    r = ENGINE.evaluate(dunning(at=night), ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS)))
    assert r.disposition is Disposition.RESCHEDULE
    from backstop.policy.engine import IST
    assert r.final.scheduled_at.astimezone(IST).hour == PolicyConfig().quiet_hours_end_ist


def test_email_is_exempt_from_quiet_hours():
    """An inbox waits; a phone lights up."""
    night = datetime(2026, 8, 30, 21, 30, tzinfo=UTC)
    r = ENGINE.evaluate(dunning(at=night, channel=Channel.EMAIL),
                        ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS)))
    assert r.disposition is not Disposition.RESCHEDULE


def test_reschedules_compose_to_satisfy_every_constraint():
    """A delay for spacing can land inside quiet hours; both must hold."""
    o = make_order(DeclineCode.INSUFFICIENT_FUNDS)
    outage_end = NOW + timedelta(days=1, hours=9)  # lands at 02:30 IST
    r = ENGINE.evaluate(dunning(), ctx(order=o, outage_until=outage_end))
    if r.disposition is Disposition.RESCHEDULE:
        from backstop.policy.engine import IST
        hour = r.final.scheduled_at.astimezone(IST).hour
        cfg = PolicyConfig()
        assert cfg.quiet_hours_end_ist <= hour < cfg.quiet_hours_start_ist


def test_outage_hold_defers_retries_past_the_outage():
    until = NOW + timedelta(hours=6)
    r = ENGINE.evaluate(retry(NOW + timedelta(days=2)),
                        ctx(order=make_order(DeclineCode.GATEWAY_TIMEOUT), outage_until=until))
    assert r.allowed


# -- engine invariants ---------------------------------------------------


def test_high_value_requires_approval_rather_than_silent_execution():
    r = ENGINE.evaluate(retry(NOW + timedelta(days=2)),
                        ctx(order=make_order(DeclineCode.INSUFFICIENT_FUNDS, amount=60000)))
    assert r.disposition is Disposition.REQUIRE_APPROVAL
    assert r.allowed, "approval is a gate, not a refusal"


def test_denial_is_final_whatever_else_voted():
    """Adding a rule can only ever make the system more conservative."""
    r = ENGINE.evaluate(retry(), ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
    assert r.disposition is Disposition.DENY
    assert r.final is None


def test_every_ruling_names_the_rules_that_shaped_it():
    """'Zero policy violations' must be a claim about a log."""
    r = ENGINE.evaluate(retry(), ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
    assert r.verdicts
    assert all(v.rule_id for v in r.verdicts)
    assert r.blocking_rule in {rule.id for rule in DEFAULT_RULES}


def test_inert_actions_are_always_permitted():
    """Doing nothing, waiting and escalating must never be blocked, or the
    engine could leave the planner with no legal move at all."""
    for kind in (ActionType.DO_NOTHING, ActionType.WAIT, ActionType.ESCALATE_TO_HUMAN,
                 ActionType.ESCALATE_TO_RISK):
        a = Action(type=kind, subject_id="o", scheduled_at=NOW, rationale="r")
        r = ENGINE.evaluate(a, ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
        assert r.allowed, f"{kind} must always remain available"


def test_rule_ids_are_unique():
    ids = [r.id for r in DEFAULT_RULES]
    assert len(ids) == len(set(ids))


def test_a_reschedule_is_permission_not_refusal():
    """`allowed` means "not denied", and a caller that reads it as "dispatch
    now" would drop every action the three timing rules touch."""
    r = ENGINE.evaluate(retry(NOW + timedelta(hours=1)),
                        ctx(order=make_order(DeclineCode.GATEWAY_TIMEOUT),
                            outage_until=NOW + timedelta(hours=6)))
    assert r.disposition is Disposition.RESCHEDULE
    assert r.allowed, "a moved action is permitted, just later"
    assert r.final is not None
    assert r.moving_rule == "outage_hold"


def test_only_a_denial_is_a_refusal():
    r = ENGINE.evaluate(retry(), ctx(order=make_order(DeclineCode.STOLEN_OR_LOST_CARD)))
    assert not r.allowed
    assert r.moving_rule is None
