"""The scheduler: nothing fires before its moment, and lateness is not free.

`RESCHEDULE` was a real disposition with no live consequence. The engine moved
a 22:30 SMS to 09:00 and the adapter sent it at 22:30, which is a log entry
claiming a protection that did not happen. These tests pin down the four
properties that make the disposition mean something outside the simulator:

*   **Nothing fires early.** The single guarantee. Everything else exists to
    keep it honest when a deployment misbehaves.
*   **The rules run again at fire time.** The world the action was judged in is
    not the world it lands in, so a denial still denies at the later clock.
*   **A re-held action is not immortal.** A blocker that never clears has to
    end in the action being abandoned, not in an unbounded retry loop.
*   **A missed window is not a licence to fire late.** A process that was down
    comes back holding actions whose moment has passed.

Offline: a recording executor, no model, no network.
"""

from datetime import UTC, datetime, timedelta

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
from backstop.execute.executor import ExecutionResult, Outcome
from backstop.ledger.ledger import Surface
from backstop.policy.engine import PolicyContext, PolicyEngine
from backstop.schedule import Fate, Scheduler, SchedulerState

NOW = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)  # a Monday morning, IST daytime


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


class Recorder:
    """An executor that records what it was asked to do, and when."""

    name = "recorder"

    def __init__(self):
        self.calls: list[tuple[Action, datetime]] = []

    def execute(self, action: Action, at: datetime) -> ExecutionResult:
        self.calls.append((action, at))
        return ExecutionResult(action, Outcome.NO_EFFECT, at, detail="recorded")


def order(oid="o", amount=2500, code=DeclineCode.INSUFFICIENT_FUNDS):
    o = Order(id=oid, customer_id="c1", amount=Money.rupees(amount), created_at=NOW)
    o.attempts.append(PaymentAttempt(
        id="a", order_id=oid, customer_id="c1", amount=o.amount, rail=Rail.CARD,
        at=NOW - timedelta(days=2), status=AttemptStatus.FAILED,
        decline_code=code, issuer="HDFC"))
    return o


def customer(**kw):
    base = {"id": "c1", "email": "a@b.test", "phone": "+9198",
            "consented_channels": {Channel.EMAIL, Channel.SMS}}
    return Customer(**{**base, **kw})


def dunning(at=NOW, channel=Channel.EMAIL, subject="o"):
    return Action(type=ActionType.SEND_DUNNING, subject_id=subject,
                  scheduled_at=at, channel=channel, rationale="r")


def ctx(now=NOW, **kw):
    kw.setdefault("customer", customer())
    kw.setdefault("order", order())
    return PolicyContext(now=now, **kw)


def contacts(n, *, at=NOW, subject="o"):
    return [ContactRecord(id=f"k{i}", customer_id="c1", channel=Channel.EMAIL,
                          at=at - timedelta(days=i), subject_ref=subject)
            for i in range(n)]


def always(context):
    return lambda action: context


# --------------------------------------------------------------------------
# nothing fires before its moment
# --------------------------------------------------------------------------


def test_an_action_scheduled_for_later_does_not_fire_now():
    """The guarantee. Without it, RESCHEDULE is a log entry, not a protection."""
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW + timedelta(hours=6)), at=NOW)
    assert s.run_due(ex, engine, always(ctx()), at=NOW) == []
    assert ex.calls == []


def test_it_fires_once_its_moment_arrives():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW + timedelta(hours=6)), at=NOW)
    later = NOW + timedelta(hours=6)
    firings = s.run_due(ex, engine, always(ctx(now=later)), at=later)
    assert [f.fate for f in firings] == [Fate.DISPATCHED]
    assert len(ex.calls) == 1


def test_a_quiet_hours_reschedule_is_actually_honoured():
    """The case the whole component exists for.

    22:30 IST is inside quiet hours, so the engine moves the SMS. Before this
    scheduler the adapter sent it at 22:30 anyway.
    """
    night = datetime(2026, 3, 2, 17, 30, tzinfo=UTC)  # 23:00 IST
    engine, s, ex = PolicyEngine(), Scheduler(), Recorder()
    action = dunning(at=night, channel=Channel.SMS)
    ruling = engine.evaluate(action, ctx(now=night))
    s.submit(ruling.final or action, at=night)

    assert s.run_due(ex, engine, always(ctx(now=night)), at=night) == []
    assert ex.calls == [], "it must not go out during the night"

    morning = s.next_due
    assert morning > night
    fired = s.run_due(ex, engine, always(ctx(now=morning)), at=morning)
    assert [f.fate for f in fired] == [Fate.DISPATCHED]


def test_there_is_no_way_to_fire_something_early():
    """No flush(), no force. A method that skipped the wait would remove the
    only guarantee this class makes."""
    assert not hasattr(Scheduler, "flush")
    assert not hasattr(Scheduler, "force")


def test_due_actions_fire_in_due_time_order_not_submission_order():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    for hours in (5, 1, 3):
        s.submit(dunning(at=NOW + timedelta(hours=hours)), at=NOW)
    later = NOW + timedelta(hours=6)
    s.run_due(ex, engine, always(ctx(now=later)), at=later)
    assert [a.scheduled_at for a, _ in ex.calls] == [
        NOW + timedelta(hours=1), NOW + timedelta(hours=3), NOW + timedelta(hours=5),
    ]


def test_resubmitting_the_same_action_does_not_queue_it_twice():
    """A replayed plan must not become two dispatches."""
    s = Scheduler()
    first = s.submit(dunning(at=NOW + timedelta(hours=2)), at=NOW)
    again = s.submit(dunning(at=NOW + timedelta(hours=2)), at=NOW)
    assert again is first
    assert len(s.entries) == 1


# --------------------------------------------------------------------------
# the rules run again at fire time
# --------------------------------------------------------------------------


def test_an_action_denied_by_the_time_it_comes_due_is_not_sent():
    """The world moved while it waited. A ruling from yesterday is not a
    licence to act today."""
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW + timedelta(hours=4)), at=NOW)
    later = NOW + timedelta(hours=4)
    full = PolicyContext(now=later, customer=customer(), order=order(),
                         contacts=contacts(3), customer_contacts=contacts(3))
    firings = s.run_due(ex, engine, always(full), at=later)
    assert [f.fate for f in firings] == [Fate.REFUSED]
    assert firings[0].blocking_rule == "contact_frequency"
    assert ex.calls == []


def test_a_refusal_at_fire_time_names_the_rule():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW + timedelta(hours=1)), at=NOW)
    later = NOW + timedelta(hours=1)
    gone = PolicyContext(now=later, order=order(),
                         customer=customer(opted_out_at=NOW))
    firing = s.run_due(ex, engine, always(gone), at=later)[0]
    assert firing.blocking_rule == "contact_consent"
    assert firing.scheduled.state is SchedulerState.REFUSED


def test_each_entry_is_judged_against_its_own_action():
    """Sounds obvious; it is exactly what went wrong when this was first wired
    into the walkthrough. A caller that hands back one shared context for
    everything is re-ruling each action against some other subject, and the
    refusals it produces name rules that have nothing to do with the clock."""
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    for i in range(3):
        s.submit(dunning(at=NOW + timedelta(hours=i), subject=f"o{i}"), at=NOW)

    seen: list[str] = []

    def context_for(action):
        seen.append(action.subject_id)
        return ctx(now=NOW + timedelta(hours=3), order=order(action.subject_id))

    later = NOW + timedelta(hours=3)
    s.run_due(ex, engine, context_for, at=later)
    assert seen == ["o0", "o1", "o2"]
    assert [a.subject_id for a, _ in ex.calls] == ["o0", "o1", "o2"]


def test_a_subject_that_left_the_batch_is_not_fired_blind():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    firing = s.run_due(ex, engine, lambda action: None, at=NOW)[0]
    assert firing.fate is Fate.REFUSED
    assert ex.calls == []


def test_the_action_executed_is_the_one_the_rules_left():
    """Not the one submitted -- a fire-time reschedule that lands in the past
    still rewrites the action, and the executor must see that version."""
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    s.run_due(ex, engine, always(ctx()), at=NOW)
    fired, when = ex.calls[0]
    assert when == NOW
    assert fired.type is ActionType.SEND_DUNNING


# --------------------------------------------------------------------------
# a re-held action is not immortal
# --------------------------------------------------------------------------


def test_an_action_the_rules_push_again_is_re_held_not_fired():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    outage = PolicyContext(now=NOW, customer=customer(), order=order(),
                           outage_until=NOW + timedelta(hours=3))
    # A charging action is what outage_hold moves, so schedule one.
    s.entries.clear()
    retry = Action(type=ActionType.RETRY_PAYMENT, subject_id="o",
                   scheduled_at=NOW, rationale="r")
    entry = s.submit(retry, at=NOW)
    firing = s.run_due(ex, engine, always(outage), at=NOW)[0]

    assert firing.fate is Fate.DEFERRED
    assert ex.calls == [], "a deferral is not a dispatch"
    assert entry.state is SchedulerState.WAITING
    assert entry.due_at == NOW + timedelta(hours=3)
    assert entry.deferrals == 1


def test_a_deferred_action_fires_once_the_blocker_clears():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    retry = Action(type=ActionType.RETRY_PAYMENT, subject_id="o",
                   scheduled_at=NOW, rationale="r")
    s.submit(retry, at=NOW)
    outage_until = NOW + timedelta(hours=3)
    s.run_due(ex, engine, always(PolicyContext(
        now=NOW, customer=customer(), order=order(), outage_until=outage_until)), at=NOW)

    clear = PolicyContext(now=outage_until, customer=customer(), order=order())
    firings = s.run_due(ex, engine, always(clear), at=outage_until)
    assert [f.fate for f in firings] == [Fate.DISPATCHED]
    assert len(ex.calls) == 1


def test_an_action_pushed_past_the_limit_is_abandoned():
    """A blocker that never clears has to end somewhere. An unbounded retry
    loop is the same failure as an unbounded contact loop."""
    s, ex, engine = Scheduler(max_deferrals=2), Recorder(), PolicyEngine()
    retry = Action(type=ActionType.RETRY_PAYMENT, subject_id="o",
                   scheduled_at=NOW, rationale="r")
    entry = s.submit(retry, at=NOW)

    at = NOW
    fates = []
    for _ in range(4):
        never_clears = PolicyContext(now=at, customer=customer(), order=order(),
                                     outage_until=at + timedelta(hours=2))
        firings = s.run_due(ex, engine, always(never_clears), at=at)
        if not firings:
            break
        fates.append(firings[0].fate)
        at = entry.due_at

    assert fates[-1] is Fate.ABANDONED
    assert entry.state is SchedulerState.ABANDONED
    assert ex.calls == []


def test_every_move_is_recorded():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    retry = Action(type=ActionType.RETRY_PAYMENT, subject_id="o",
                   scheduled_at=NOW, rationale="r")
    entry = s.submit(retry, at=NOW)
    s.run_due(ex, engine, always(PolicyContext(
        now=NOW, customer=customer(), order=order(),
        outage_until=NOW + timedelta(hours=5))), at=NOW)
    assert entry.history == [NOW, NOW + timedelta(hours=5)]


# --------------------------------------------------------------------------
# a missed window is not a licence to fire late
# --------------------------------------------------------------------------


def test_an_action_long_past_due_is_dropped_rather_than_fired_late():
    """The process was down for two days. A retry timed for the hour after a
    failure is a different act two days later."""
    s = Scheduler(max_lateness=timedelta(hours=24))
    ex, engine = Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    much_later = NOW + timedelta(days=2)
    firings = s.run_due(ex, engine, always(ctx(now=much_later)), at=much_later)
    assert [f.fate for f in firings] == [Fate.STALE]
    assert ex.calls == []
    assert s.in_state(SchedulerState.STALE)


def test_an_action_slightly_late_still_fires():
    """Lateness is a spectrum, not a cliff at zero. A deployment that ticks
    every few minutes is always slightly late."""
    s = Scheduler(max_lateness=timedelta(hours=24))
    ex, engine = Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    bit_late = NOW + timedelta(minutes=20)
    firings = s.run_due(ex, engine, always(ctx(now=bit_late)), at=bit_late)
    assert [f.fate for f in firings] == [Fate.DISPATCHED]


def test_staleness_is_recorded_rather_than_silently_dropped():
    s = Scheduler(max_lateness=timedelta(hours=6))
    ex, engine = Recorder(), PolicyEngine()
    entry = s.submit(dunning(at=NOW), at=NOW)
    late = NOW + timedelta(days=1)
    s.run_due(ex, engine, always(ctx(now=late)), at=late)
    assert entry.state is SchedulerState.STALE
    assert entry.settled_at == late
    assert "not fired late" in entry.note


def test_a_stale_action_is_not_reconsidered_on_the_next_tick():
    s = Scheduler(max_lateness=timedelta(hours=6))
    ex, engine = Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    late = NOW + timedelta(days=1)
    s.run_due(ex, engine, always(ctx(now=late)), at=late)
    assert s.run_due(ex, engine, always(ctx(now=late)), at=late) == []


# --------------------------------------------------------------------------
# reading the schedule
# --------------------------------------------------------------------------


def test_next_due_reports_when_there_is_something_to_do():
    s = Scheduler()
    assert s.next_due is None
    s.submit(dunning(at=NOW + timedelta(hours=5)), at=NOW)
    s.submit(dunning(at=NOW + timedelta(hours=2), subject="o2"), at=NOW)
    assert s.next_due == NOW + timedelta(hours=2)


def test_an_action_fires_once():
    s, ex, engine = Scheduler(), Recorder(), PolicyEngine()
    s.submit(dunning(at=NOW), at=NOW)
    s.run_due(ex, engine, always(ctx()), at=NOW)
    s.run_due(ex, engine, always(ctx()), at=NOW)
    assert len(ex.calls) == 1


def test_counts_and_surface_views_report_the_same_entries():
    s = Scheduler()
    s.submit(dunning(at=NOW), surface=Surface.PAYMENT, at=NOW)
    s.submit(dunning(at=NOW, subject="inv_1"), surface=Surface.RECEIVABLE, at=NOW)
    assert s.counts()[SchedulerState.WAITING] == 2
    assert len(s.on(Surface.RECEIVABLE).entries) == 1


# --------------------------------------------------------------------------
# a deferral names its author
# --------------------------------------------------------------------------


def test_a_deferral_names_the_rule_that_moved_it():
    """Every other disposition in this system names the rule behind it. A
    deferral that could only say "a rule" would be the exception, and the
    audit trail is the product."""
    scheduler = Scheduler()
    until = NOW + timedelta(hours=6)
    action = Action(type=ActionType.RETRY_PAYMENT, subject_id="o",
                    scheduled_at=NOW, rationale="r")
    scheduler.submit(action, at=NOW)

    [firing] = scheduler.run_due(
        Recorder(), PolicyEngine(), always(ctx(outage_until=until)), at=NOW
    )

    assert firing.fate is Fate.DEFERRED
    assert firing.ruling.moving_rule == "outage_hold"
    assert "outage_hold" in firing.detail
    assert "outage_hold" in firing.scheduled.note
