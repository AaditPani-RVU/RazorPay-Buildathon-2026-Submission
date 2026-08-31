"""The approval queue: what a human yes buys, and what it does not.

`REQUIRE_APPROVAL` used to be a disposition with nowhere to go -- the live
walkthrough held those actions back and dropped them, which is a gate with a
deletion behind it. These tests pin down the four properties that make the
queue behind it worth having, all of which cost the recovery number something:

*   **An approval is not an exemption.** The rules run again at release, in the
    world that holds then. If the queue could dispatch whatever a reviewer said
    yes to on Tuesday, every guarantee the engine makes would carry a silent
    asterisk exactly the width of this class.
*   **Silence is not consent.** Unanswered requests expire and are recorded as
    expired, so a merchant who does not staff the desk sees the revenue they
    gave up rather than quietly booking it.
*   **A decision has a person on it.** No auto-approve, no anonymous yes.
*   **One yes, one dispatch.** A released request cannot release again.

Offline, no model, no network.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backstop.approve import (
    ApprovalQueue,
    ApprovalState,
    ReleaseOutcome,
    StandingApproval,
)
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    ContactRecord,
    Customer,
    Order,
    PaymentAttempt,
)
from backstop.domain.money import Money
from backstop.ledger.ledger import Surface
from backstop.policy.engine import (
    Disposition,
    PolicyContext,
    PolicyEngine,
    Ruling,
    Verdict,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
BIG = 60000  # comfortably over the ₹25,000 approval threshold


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def order(oid="o", amount=BIG, code=DeclineCode.INSUFFICIENT_FUNDS):
    o = Order(id=oid, customer_id="c1", amount=Money.rupees(amount), created_at=NOW)
    o.attempts.append(PaymentAttempt(
        id="a", order_id=oid, customer_id="c1", amount=o.amount, rail=Rail.CARD,
        at=NOW, status=AttemptStatus.FAILED, decline_code=code, issuer="HDFC"))
    return o


def customer(**kw):
    base = {"id": "c1", "email": "a@b.test", "phone": "+9198",
            "consented_channels": {Channel.EMAIL, Channel.SMS}}
    return Customer(**{**base, **kw})


def dunning(at=NOW, channel=Channel.EMAIL, subject="o"):
    return Action(type=ActionType.SEND_DUNNING, subject_id=subject,
                  scheduled_at=at, channel=channel, rationale="r")


def ctx(**kw):
    kw.setdefault("customer", customer())
    kw.setdefault("order", order())
    return PolicyContext(now=NOW, **kw)


def contacts(n, *, at=NOW, subject="o"):
    return [ContactRecord(id=f"k{i}", customer_id="c1", channel=Channel.EMAIL,
                          at=at - timedelta(days=i), subject_ref=subject)
            for i in range(n)]


def queued(engine=None, action=None, context=None, *, at=NOW, ttl=None):
    """Put one action through the engine and into the queue, as the pipeline does."""
    engine = engine or PolicyEngine()
    action = action or dunning()
    ruling = engine.evaluate(action, context or ctx())
    assert ruling.disposition is Disposition.REQUIRE_APPROVAL, ruling.describe()
    q = ApprovalQueue()
    request = q.submit(action, ruling, surface=Surface.PAYMENT, at=at, ttl=ttl)
    return q, request, engine


def always(context):
    return lambda action: context


# --------------------------------------------------------------------------
# what may enter
# --------------------------------------------------------------------------


def test_only_an_approval_ruling_may_be_queued():
    """A queue that accepts a DENY is a way around the DENY."""
    engine = PolicyEngine()
    action = dunning()
    denied = Ruling(
        proposed=action, disposition=Disposition.DENY,
        verdicts=[Verdict("fraud_block", Disposition.DENY, "no")],
    )
    with pytest.raises(ValueError, match="only require_approval"):
        ApprovalQueue().submit(action, denied, at=NOW)
    assert engine  # engine unused here on purpose; the guard is structural


def test_the_queue_records_which_rule_asked_for_a_person():
    _, request, _ = queued()
    assert request.asking_rule == "high_value_approval"
    assert "approval threshold" in request.reason


def test_resubmitting_the_same_action_is_not_a_second_ask():
    """A replayed plan must not put the same decision in front of a person
    twice. Identity is the dispatch fingerprint, so this holds across runs."""
    q, first, engine = queued()
    again = q.submit(dunning(), engine.evaluate(dunning(), ctx()), at=NOW)
    assert again is first
    assert len(q.requests) == 1


def test_rewording_the_rationale_does_not_buy_a_fresh_ask():
    """`rationale` is excluded from identity for the same reason the policy
    engine ignores it: a model must not talk its way into a second decision."""
    q, first, engine = queued()
    reworded = dunning().model_copy(update={"rationale": "a much better reason"})
    again = q.submit(reworded, engine.evaluate(reworded, ctx()), at=NOW)
    assert again is first


def test_a_genuinely_different_action_is_a_different_ask():
    q, _, engine = queued()
    later = dunning(at=NOW + timedelta(days=3))
    q.submit(later, engine.evaluate(later, ctx()), at=NOW)
    assert len(q.requests) == 2


# --------------------------------------------------------------------------
# a decision has a person on it, and is made once
# --------------------------------------------------------------------------


def test_an_approval_must_name_the_person_who_gave_it():
    q, request, _ = queued()
    with pytest.raises(ValueError, match="name the person"):
        q.approve(request.id, by="", at=NOW)


def test_a_decision_is_made_once():
    q, request, _ = queued()
    q.approve(request.id, by="ops@merchant.test", at=NOW)
    with pytest.raises(ValueError, match="already approved"):
        q.reject(request.id, by="ops@merchant.test", at=NOW)


def test_the_decision_is_recorded_with_who_and_when():
    q, request, _ = queued()
    at = NOW + timedelta(hours=3)
    q.approve(request.id, by="finance@merchant.test", at=at, note="known good buyer")
    assert request.state is ApprovalState.APPROVED
    assert request.decided_by == "finance@merchant.test"
    assert request.decided_at == at
    assert request.note == "known good buyer"


def test_deciding_an_unknown_request_is_an_error_not_a_silent_no_op():
    with pytest.raises(KeyError):
        ApprovalQueue().approve("bkstp_nope", by="ops", at=NOW)


# --------------------------------------------------------------------------
# silence is not consent
# --------------------------------------------------------------------------


def test_an_unanswered_request_expires_rather_than_firing():
    q, request, _ = queued(ttl=timedelta(days=2))
    expired = q.expire_due(NOW + timedelta(days=3))
    assert expired == [request]
    assert request.state is ApprovalState.EXPIRED
    assert request.decided_at == request.expires_at


def test_an_expired_request_is_no_longer_pending():
    q, request, _ = queued(ttl=timedelta(days=2))
    assert q.pending(NOW) == [request]
    assert q.pending(NOW + timedelta(days=3)) == []


def test_a_yes_that_arrives_too_late_is_recorded_as_expiry_not_approval():
    """The honest record is that nobody got to it in time."""
    q, request, _ = queued(ttl=timedelta(days=2))
    q.approve(request.id, by="ops", at=NOW + timedelta(days=5))
    assert request.state is ApprovalState.EXPIRED
    assert request.decided_by == ""


def test_expiry_is_visible_as_revenue_given_up():
    """A merchant who does not staff the desk should be able to count what it
    cost them, which is the point of recording expiry rather than dropping."""
    q, request, _ = queued(ttl=timedelta(hours=6))
    q.expire_due(NOW + timedelta(days=1))
    assert q.expired == [request]
    assert q.counts()[ApprovalState.EXPIRED] == 1


# --------------------------------------------------------------------------
# an approval is not an exemption -- the load-bearing property
# --------------------------------------------------------------------------


def test_release_re_evaluates_and_still_refuses_a_now_deniable_action():
    """The single most important test in this file.

    The reviewer said yes on Tuesday. By the time the action would go out, the
    customer's fortnight contact budget is full. The rules must win.
    """
    q, request, engine = queued()
    q.approve(request.id, by="ops", at=NOW)

    later = NOW + timedelta(hours=6)
    full = PolicyContext(now=later, customer=customer(), order=order(),
                         contacts=contacts(3), customer_contacts=contacts(3))
    releases = q.release(engine, always(full), at=later)

    assert len(releases) == 1
    assert releases[0].outcome is ReleaseOutcome.REFUSED
    assert not releases[0].dispatchable
    assert releases[0].blocking_rule == "contact_frequency"


def test_release_names_the_rule_that_overtook_the_reviewer():
    q, request, engine = queued()
    q.approve(request.id, by="ops", at=NOW)
    opted_out = PolicyContext(now=NOW, customer=customer(opted_out_at=NOW - timedelta(days=1)), order=order())
    release = q.release(engine, always(opted_out), at=NOW)[0]
    assert release.outcome is ReleaseOutcome.REFUSED
    assert release.blocking_rule == "contact_consent"


def test_a_human_yes_discharges_the_approval_verdict_and_nothing_else():
    """Otherwise every approved action is immortal in the queue: the
    high_value rule fires again on the re-evaluation, forever."""
    q, request, engine = queued()
    q.approve(request.id, by="ops", at=NOW)
    release = q.release(engine, always(ctx()), at=NOW)[0]
    assert release.outcome is ReleaseOutcome.RELEASED
    assert release.dispatchable
    assert release.ruling.disposition is Disposition.REQUIRE_APPROVAL


def test_a_reschedule_at_release_moves_the_action_rather_than_dropping_it():
    night = datetime(2026, 3, 1, 22, 30, tzinfo=UTC)
    late = dunning(at=night, channel=Channel.SMS)
    q, request, engine = queued(
        action=late, context=PolicyContext(now=night, customer=customer(), order=order()))
    q.approve(request.id, by="ops", at=night)
    release = q.release(engine, always(PolicyContext(now=night, customer=customer(),
                                                     order=order())), at=night)[0]
    assert release.outcome is ReleaseOutcome.RESCHEDULED
    assert release.action.scheduled_at > night


def test_a_rejected_request_never_releases():
    q, request, engine = queued()
    q.reject(request.id, by="ops", at=NOW, note="customer already called in")
    assert q.release(engine, always(ctx()), at=NOW) == []


def test_a_pending_request_never_releases():
    q, _, engine = queued()
    assert q.release(engine, always(ctx()), at=NOW) == []


def test_a_subject_that_left_the_batch_is_refused_not_dispatched_blind():
    q, request, engine = queued()
    q.approve(request.id, by="ops", at=NOW)
    release = q.release(engine, lambda action: None, at=NOW)[0]
    assert release.outcome is ReleaseOutcome.REFUSED
    assert not release.dispatchable


def test_one_yes_is_one_dispatch():
    """A queue that can release twice can contact somebody twice for one
    approval, which is precisely what the contact rules exist to stop."""
    q, request, engine = queued()
    q.approve(request.id, by="ops", at=NOW)
    assert len(q.release(engine, always(ctx()), at=NOW)) == 1
    assert q.release(engine, always(ctx()), at=NOW) == []


# --------------------------------------------------------------------------
# the backtest's assumption, made visible
# --------------------------------------------------------------------------


def test_a_standing_approval_signs_off_after_its_stated_latency():
    q, request, _ = queued()
    desk = StandingApproval(latency=timedelta(hours=4))
    assert q.staff(desk, at=NOW + timedelta(hours=1)) == []
    assert q.staff(desk, at=NOW + timedelta(hours=5)) == [request]
    assert request.state is ApprovalState.APPROVED
    assert request.decided_at == NOW + timedelta(hours=4)


def test_a_standing_approval_names_itself_as_an_assumption():
    """Reading the ledger should tell you an assumption was made, and by whom."""
    q, request, _ = queued()
    q.staff(StandingApproval(), at=NOW + timedelta(days=1))
    assert request.decided_by.startswith("backtest:")
    assert "assum" in request.note


def test_a_desk_slower_than_the_shelf_life_approves_nothing():
    q, request, _ = queued(ttl=timedelta(hours=2))
    q.staff(StandingApproval(latency=timedelta(days=1)), at=NOW + timedelta(days=2))
    assert request.state is ApprovalState.PENDING
    q.expire_due(NOW + timedelta(days=2))
    assert request.state is ApprovalState.EXPIRED


def test_a_partly_staffed_desk_is_deterministic():
    """The same batch must produce the same staffing outcome on every arm and
    every re-run, or the comparison is measuring the sampler."""
    engine = PolicyEngine()
    outcomes = []
    for _ in range(2):
        q = ApprovalQueue()
        for i in range(40):
            action = dunning(at=NOW + timedelta(hours=i))
            q.submit(action, engine.evaluate(action, ctx()), at=NOW)
        q.staff(StandingApproval(rate=0.5), at=NOW + timedelta(days=1))
        outcomes.append(sorted(r.id for r in q.in_state(ApprovalState.APPROVED)))
    assert outcomes[0] == outcomes[1]
    assert 0 < len(outcomes[0]) < 40


def test_an_unstaffed_desk_expires_everything_it_never_reached():
    engine = PolicyEngine()
    q = ApprovalQueue()
    for i in range(20):
        action = dunning(at=NOW + timedelta(hours=i))
        q.submit(action, engine.evaluate(action, ctx()), at=NOW, ttl=timedelta(days=2))
    q.staff(StandingApproval(rate=0.0), at=NOW + timedelta(days=1))
    q.expire_due(NOW + timedelta(days=3))
    assert len(q.expired) == 20
    assert q.counts()[ApprovalState.APPROVED] == 0


# --------------------------------------------------------------------------
# reading the queue
# --------------------------------------------------------------------------


def test_pending_is_ordered_oldest_first():
    engine = PolicyEngine()
    q = ApprovalQueue()
    for i in (3, 1, 2):
        action = dunning(at=NOW + timedelta(hours=i))
        q.submit(action, engine.evaluate(action, ctx()), at=NOW + timedelta(hours=i))
    assert [r.submitted_at for r in q.pending(NOW + timedelta(hours=4))] == [
        NOW + timedelta(hours=1), NOW + timedelta(hours=2), NOW + timedelta(hours=3),
    ]


def test_a_surface_view_carries_only_that_surface():
    engine = PolicyEngine()
    q = ApprovalQueue()
    pay = dunning()
    inv = dunning(at=NOW + timedelta(hours=1))
    q.submit(pay, engine.evaluate(pay, ctx()), surface=Surface.PAYMENT, at=NOW)
    q.submit(inv, engine.evaluate(inv, ctx()), surface=Surface.RECEIVABLE, at=NOW)
    assert len(q.on(Surface.RECEIVABLE).requests) == 1
    assert len(q.on(Surface.PAYMENT).requests) == 1
