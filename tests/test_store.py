"""What survives a restart, and why losing it would be a safety failure.

A process holding recovery state in memory is fine right up until it stops,
and then it is three different problems at once:

*   **Work that quietly disappears.** The scheduler's promise is that a held
    action gets its moment. A queue that dies with the process keeps the half
    of that promise nobody wants -- it never fires early, because it never
    fires.
*   **A second contact for one yes.** `released` is what makes a reviewer's
    approval spendable exactly once. Emptied by a restart, one approval
    dispatches again.
*   **The same money booked twice.** `reconciled` is what makes a settlement
    creditable once. Emptied, the next redelivered `payment_link.paid` credits
    a payment that was already credited -- and Razorpay redelivers by design.

So these tests are less about serialisation than about those three, and the
mechanical ones underneath them: a torn write costs one update rather than an
entity, and the restored state is the state that was saved rather than a
plausible reconstruction of it.

Everything runs offline against a temporary file.
"""

import json
from datetime import UTC, datetime, timedelta

from backstop.approve import ApprovalQueue, ApprovalState, ReleaseOutcome
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    ContactRecord,
    Customer,
    Invoice,
    MandateStatus,
    Order,
    PaymentAttempt,
    Subscription,
)
from backstop.domain.money import Money
from backstop.execute.executor import ExecutionResult, Outcome
from backstop.execute.razorpay import (
    ApiResponse,
    RazorpayExecutor,
    RecordedTransport,
    reference_for,
)
from backstop.execute.webhook import Verdict as WebhookVerdict
from backstop.execute.webhook import WebhookReceiver
from backstop.ledger.ledger import Surface
from backstop.policy.engine import (
    Disposition,
    PolicyContext,
    PolicyEngine,
)
from backstop.schedule import Fate, Scheduler, SchedulerState
from backstop.store import APPROVAL, DISPATCH, SCHEDULED, Journal

NOW = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)  # a Monday morning, IST daytime
BIG = 60000  # over the approval threshold, so the engine asks for a person


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


def journal(tmp_path, name="live.jsonl") -> Journal:
    return Journal(tmp_path / name)


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
# the file itself
# --------------------------------------------------------------------------


def test_a_journal_that_does_not_exist_yet_replays_as_nothing(tmp_path):
    """The ordinary state of a fresh deployment, not an error."""
    replay = journal(tmp_path).replay()
    assert replay.records == []
    assert replay.damaged == 0


def test_the_last_snapshot_of_an_entity_wins(tmp_path):
    j = journal(tmp_path)
    j.append(SCHEDULED, "a", {"state": "waiting"})
    j.append(SCHEDULED, "a", {"state": "fired"})
    assert j.replay().latest(SCHEDULED)["a"]["state"] == "fired"


def test_every_snapshot_is_kept_as_the_audit_trail(tmp_path):
    j = journal(tmp_path)
    j.append(SCHEDULED, "a", {"state": "waiting"})
    j.append(SCHEDULED, "a", {"state": "deferred"})
    j.append(SCHEDULED, "a", {"state": "fired"})
    history = j.replay().history(SCHEDULED, "a")
    assert [r.body["state"] for r in history] == ["waiting", "deferred", "fired"]


def test_kinds_share_one_file_without_colliding(tmp_path):
    j = journal(tmp_path)
    j.append(SCHEDULED, "same_id", {"which": "scheduled"})
    j.append(APPROVAL, "same_id", {"which": "approval"})
    j.append(DISPATCH, "same_id", {"which": "dispatch"})
    replay = j.replay()
    assert replay.latest(SCHEDULED)["same_id"]["which"] == "scheduled"
    assert replay.latest(APPROVAL)["same_id"]["which"] == "approval"
    assert replay.latest(DISPATCH)["same_id"]["which"] == "dispatch"


def test_a_torn_last_write_is_skipped_and_counted(tmp_path):
    """A process killed mid-append leaves half a line. Two properties: the
    other records still load, and the loss is reported rather than hidden."""
    j = journal(tmp_path)
    j.append(SCHEDULED, "a", {"state": "waiting"})
    with open(j.path, "a") as fh:
        fh.write('{"kind":"scheduled","id":"b","at":"2026-03-02T09:00:00+00:00","bo')

    replay = j.replay()
    assert replay.damaged == 1
    assert set(replay.latest(SCHEDULED)) == {"a"}


def test_a_damaged_record_costs_one_update_not_the_entity(tmp_path):
    """The whole reason snapshots are written instead of deltas.

    A torn record in a delta log leaves the entity in the wrong state forever.
    Here the previous snapshot is still in the file, so the entity falls back
    to something stale and true rather than to something invented.
    """
    j = journal(tmp_path)
    j.append(SCHEDULED, "a", {"state": "waiting", "deferrals": 1})
    lines = j.path.read_text().splitlines()
    lines.append('{"kind":"scheduled","id":"a","at":"nope","body":{"state":"fir')
    j.path.write_text("\n".join(lines) + "\n")

    replay = j.replay()
    assert replay.damaged == 1
    assert replay.latest(SCHEDULED)["a"] == {"state": "waiting", "deferrals": 1}


def test_compaction_keeps_the_state_and_drops_the_history(tmp_path):
    j = journal(tmp_path)
    for state in ("waiting", "deferred", "fired"):
        j.append(SCHEDULED, "a", {"state": state})
    j.append(SCHEDULED, "b", {"state": "waiting"})

    dropped = j.compact()

    assert dropped == 2
    assert len(j.replay().records) == 2
    assert j.replay().latest(SCHEDULED)["a"]["state"] == "fired"
    assert j.replay().latest(SCHEDULED)["b"]["state"] == "waiting"


def test_compaction_leaves_a_file_that_is_still_appendable(tmp_path):
    j = journal(tmp_path)
    j.append(SCHEDULED, "a", {"state": "waiting"})
    j.append(SCHEDULED, "a", {"state": "fired"})
    j.compact()
    j.append(SCHEDULED, "b", {"state": "waiting"})

    replay = j.replay()
    assert replay.damaged == 0
    assert set(replay.latest(SCHEDULED)) == {"a", "b"}


def test_a_record_is_on_disk_before_append_returns(tmp_path):
    """Written synchronously, because the caller is about to act on it."""
    j = journal(tmp_path)
    j.append(DISPATCH, "ref", {"reference": "ref"})
    raw = json.loads(j.path.read_text().splitlines()[0])
    assert raw["kind"] == DISPATCH
    assert raw["id"] == "ref"


# --------------------------------------------------------------------------
# the scheduler across a restart
# --------------------------------------------------------------------------


def restart_scheduler(j, **kw) -> Scheduler:
    """A new process, reading the same file. Nothing is carried in memory."""
    return Scheduler(journal=j, **kw)


def test_a_held_action_survives_the_process_that_held_it(tmp_path):
    j = journal(tmp_path)
    before = Scheduler(journal=j)
    tomorrow = NOW + timedelta(days=1)
    before.submit(dunning(at=tomorrow), at=NOW)
    del before

    after = restart_scheduler(j)
    [entry] = after.waiting()
    assert entry.due_at == tomorrow
    assert entry.action.type is ActionType.SEND_DUNNING
    assert after.next_due == tomorrow


def test_a_restored_action_still_does_not_fire_early(tmp_path):
    """Persistence restores the queue, it does not relax the guarantee."""
    j = journal(tmp_path)
    Scheduler(journal=j).submit(dunning(at=NOW + timedelta(days=1)), at=NOW)

    recorder = Recorder()
    firings = restart_scheduler(j).run_due(
        recorder, PolicyEngine(), always(ctx()), at=NOW + timedelta(hours=2)
    )
    assert firings == []
    assert recorder.calls == []


def test_a_restored_action_fires_when_its_moment_arrives(tmp_path):
    j = journal(tmp_path)
    later = NOW + timedelta(hours=6)
    Scheduler(journal=j).submit(dunning(at=later), at=NOW)

    recorder = Recorder()
    [firing] = restart_scheduler(j).run_due(
        recorder, PolicyEngine(), always(ctx(now=later)), at=later
    )
    assert firing.fate is Fate.DISPATCHED
    assert len(recorder.calls) == 1


def test_a_window_that_passed_while_the_process_was_down_is_not_fired_late(tmp_path):
    """The restart does not launder lateness. Coming back holding an action
    whose moment was two days ago is exactly the case `max_lateness` is for."""
    j = journal(tmp_path)
    Scheduler(journal=j).submit(dunning(at=NOW), at=NOW)

    back_up = NOW + timedelta(days=2)
    recorder = Recorder()
    [firing] = restart_scheduler(j).run_due(
        recorder, PolicyEngine(), always(ctx(now=back_up)), at=back_up
    )
    assert firing.fate is Fate.STALE
    assert recorder.calls == []


def test_a_fired_action_does_not_fire_again_after_a_restart(tmp_path):
    """The terminal state is the record. A restart that forgot it would send
    a second message about the same thing to the same person."""
    j = journal(tmp_path)
    live = Scheduler(journal=j)
    live.submit(dunning(at=NOW), at=NOW)
    live.run_due(Recorder(), PolicyEngine(), always(ctx()), at=NOW)

    after = restart_scheduler(j)
    recorder = Recorder()
    assert after.waiting() == []
    assert after.run_due(
        recorder, PolicyEngine(), always(ctx()), at=NOW + timedelta(hours=1)
    ) == []
    assert recorder.calls == []
    assert after.counts()[SchedulerState.FIRED] == 1


def test_the_deferral_bound_does_not_reset_on_restart(tmp_path):
    """Otherwise a crash loop is an unbounded retry loop: an action pushed
    three times, restarted, and pushed three times again, forever."""
    j = journal(tmp_path)
    night = datetime(2026, 3, 2, 17, 30, tzinfo=UTC)  # 23:00 IST, quiet hours
    live = Scheduler(journal=j)
    # SMS, not email: a message that makes a phone light up is the one quiet
    # hours moves, and an email waiting in an inbox is exempt.
    live.submit(dunning(at=night, channel=Channel.SMS), at=NOW)
    live.run_due(Recorder(), PolicyEngine(), always(ctx(now=night)), at=night)

    [entry] = restart_scheduler(j).waiting()
    assert entry.deferrals == 1
    assert len(entry.history) == 2


def test_a_deferral_is_written_down_before_the_process_can_lose_it(tmp_path):
    """A re-held action restored on its *old* time would fire inside the quiet
    hours the deferral moved it out of, which is the protection undone."""
    j = journal(tmp_path)
    night = datetime(2026, 3, 2, 17, 30, tzinfo=UTC)  # 23:00 IST
    live = Scheduler(journal=j)
    live.submit(dunning(at=night, channel=Channel.SMS), at=NOW)
    [firing] = live.run_due(Recorder(), PolicyEngine(), always(ctx(now=night)), at=night)
    assert firing.fate is Fate.DEFERRED

    [entry] = restart_scheduler(j).waiting()
    assert entry.due_at > night


def test_a_surface_view_does_not_write_to_the_journal(tmp_path):
    j = journal(tmp_path)
    live = Scheduler(journal=j)
    live.submit(dunning(at=NOW), surface=Surface.PAYMENT, at=NOW)
    before = j.writes

    view = live.on(Surface.PAYMENT)
    view.submit(dunning(at=NOW + timedelta(days=1), subject="other"), at=NOW)

    assert view.journal is None
    assert j.writes == before


def test_without_a_journal_nothing_is_written_and_nothing_comes_back(tmp_path):
    """The default is unchanged: the backtest runs a batch to completion in one
    process and has no use for a file."""
    j = journal(tmp_path)
    Scheduler().submit(dunning(at=NOW), at=NOW)
    assert not j.path.exists()
    assert Scheduler().waiting() == []


# --------------------------------------------------------------------------
# the approval queue across a restart
# --------------------------------------------------------------------------


def approval_ctx(now=NOW):
    return ctx(now=now, order=order(amount=BIG))


def queued_for_approval(j, at=NOW):
    engine = PolicyEngine()
    action = dunning(at=at)
    ruling = engine.evaluate(action, approval_ctx())
    assert ruling.disposition is Disposition.REQUIRE_APPROVAL, ruling.describe()
    queue = ApprovalQueue(journal=j)
    request = queue.submit(action, ruling, surface=Surface.PAYMENT, at=at)
    return queue, request, engine


def test_a_request_waiting_for_a_person_survives_a_restart(tmp_path):
    j = journal(tmp_path)
    _, request, _ = queued_for_approval(j)

    [restored] = ApprovalQueue(journal=j).pending(NOW)
    assert restored.id == request.id
    assert restored.state is ApprovalState.PENDING
    assert restored.surface is Surface.PAYMENT


def test_the_reviewer_still_sees_why_they_are_being_asked(tmp_path):
    """The ruling travels with the request. A restored ask that could not say
    which rule raised it is a reviewer being asked to guess."""
    j = journal(tmp_path)
    _, request, _ = queued_for_approval(j)

    [restored] = ApprovalQueue(journal=j).pending(NOW)
    assert restored.asking_rule == request.asking_rule
    assert restored.reason == request.reason
    assert restored.describe() == request.describe()


def test_a_decision_and_the_person_who_made_it_survive(tmp_path):
    j = journal(tmp_path)
    queue, request, _ = queued_for_approval(j)
    later = NOW + timedelta(hours=4)
    queue.approve(request.id, by="ops@merchant.test", at=later, note="checked by hand")

    [restored] = ApprovalQueue(journal=j).in_state(ApprovalState.APPROVED)
    assert restored.decided_by == "ops@merchant.test"
    assert restored.decided_at == later
    assert restored.note == "checked by hand"


def test_one_yes_cannot_be_spent_twice_across_a_restart(tmp_path):
    """The property the whole file exists for. `released` emptied by a restart
    is a second message to a customer off a single approval."""
    j = journal(tmp_path)
    queue, request, engine = queued_for_approval(j)
    later = NOW + timedelta(hours=4)
    queue.approve(request.id, by="ops@merchant.test", at=later)
    [release] = queue.release(engine, always(approval_ctx(later)), at=later)
    assert release.outcome is ReleaseOutcome.RELEASED

    after = ApprovalQueue(journal=j)
    assert request.id in after.released
    assert after.release(engine, always(approval_ctx(later)), at=later) == []


def test_an_expired_request_does_not_come_back_answerable(tmp_path):
    """Silence is not consent, and a restart is not a second chance to say yes
    to something whose shelf life ran out."""
    j = journal(tmp_path)
    queue, request, _ = queued_for_approval(j)
    queue.expire_due(NOW + timedelta(days=3))

    after = ApprovalQueue(journal=j)
    assert after.pending(NOW + timedelta(days=3)) == []
    assert [r.id for r in after.expired] == [request.id]


def test_a_restored_approval_is_still_re_ruled_before_it_dispatches(tmp_path):
    """An approval is permission from a person, not an exemption from the
    rules -- and coming back from a restart does not change that."""
    j = journal(tmp_path)
    queue, request, engine = queued_for_approval(j)
    later = NOW + timedelta(hours=4)
    queue.approve(request.id, by="ops@merchant.test", at=later)

    # By release time the customer's fortnight contact budget has filled.
    full = ctx(now=later, order=order(amount=BIG), customer_contacts=contacts(6, at=later))
    [release] = ApprovalQueue(journal=j).release(engine, always(full), at=later)

    assert release.outcome is ReleaseOutcome.REFUSED
    assert release.blocking_rule is not None
    assert not release.dispatchable


# --------------------------------------------------------------------------
# what was dispatched, across a restart
# --------------------------------------------------------------------------


LINK = "POST /payment_links"
LOOKUP = "GET /orders"
LINK_LOOKUP = "GET /payment_links"


def transport(**routes):
    base = {
        LINK: [ApiResponse(status=200, body={
            "id": "plink_A", "short_url": "https://rzp.io/rzp/A"})],
        LOOKUP: [ApiResponse(status=200, body={"count": 0, "items": []})],
        LINK_LOOKUP: [ApiResponse(status=200, body={"count": 0, "items": []})],
    }
    base.update(routes)
    return RecordedTransport(routes=base)


def adapter(j, t=None):
    return RazorpayExecutor(
        transport=t or transport(),
        orders={"o": order()},
        invoices={"inv_1": Invoice(
            id="inv_1", buyer_id="c1", amount=Money.rupees(5000),
            issued_at=NOW - timedelta(days=70), due_at=NOW - timedelta(days=40))},
        subscriptions={"sub_1": Subscription(
            id="sub_1", customer_id="c1", amount=Money.rupees(499),
            rail=Rail.EMANDATE_NACH, mandate_status=MandateStatus.EXPIRED,
            next_charge_at=NOW)},
        customers={"c1": customer()},
        journal=j,
    )


def test_a_dispatched_link_is_still_ours_after_a_restart(tmp_path):
    """Without this the link is an orphan: nothing to poll, and nothing for a
    webhook to match, so a payment that lands on it is never credited."""
    j = journal(tmp_path)
    result = adapter(j).execute(dunning(), NOW)
    assert result.outcome is Outcome.DISPATCHED

    after = adapter(j, transport())
    dispatch = after.dispatch_for("plink_A")
    assert dispatch is not None
    assert dispatch.action.subject_id == "o"
    assert dispatch.amount == Money.rupees(2500)


def test_a_webhook_after_a_restart_credits_the_dispatch(tmp_path):
    """The loop closes across the restart: Razorpay pushes, and the new
    process knows what the id refers to and what it is worth."""
    j = journal(tmp_path)
    adapter(j).execute(dunning(), NOW)

    after = adapter(j, transport())
    receiver = WebhookReceiver(executor=after, secret="whsec")
    receipt = _deliver(receiver, "plink_A", Money.rupees(2500).paise, "evt_1")

    assert receipt.verdict is WebhookVerdict.RECOVERED
    assert receipt.result.recovered == Money.rupees(2500)


def test_a_settlement_is_not_credited_twice_across_a_restart(tmp_path):
    """Razorpay redelivers by design, and `reconciled` is what makes that
    harmless. Lost on restart, the redelivery books the same money again."""
    j = journal(tmp_path)
    live = adapter(j)
    live.execute(dunning(), NOW)
    receiver = WebhookReceiver(executor=live, secret="whsec")
    first = _deliver(receiver, "plink_A", Money.rupees(2500).paise, "evt_1")
    assert first.verdict is WebhookVerdict.RECOVERED

    # A new process, and a redelivery it has never seen the event id of. The
    # dispatch-level dedupe is the layer that has to catch this.
    after = adapter(j, transport())
    again = _deliver(
        WebhookReceiver(executor=after, secret="whsec"),
        "plink_A", Money.rupees(2500).paise, "evt_2",
    )
    assert again.verdict is WebhookVerdict.DUPLICATE
    assert again.result is None


def test_a_replayed_action_after_a_restart_is_not_a_second_send(tmp_path):
    j = journal(tmp_path)
    adapter(j).execute(dunning(), NOW)

    t = transport()
    result = adapter(j, t).execute(dunning(), NOW)

    assert result.outcome is Outcome.NO_EFFECT
    assert "already dispatched" in result.detail
    assert t.payloads(LINK) == []


def test_an_uncredited_dispatch_comes_back_uncredited(tmp_path):
    """Restoring is restoring, not settling. A dispatch nobody has paid is
    still pending after a restart, and still worth nothing."""
    j = journal(tmp_path)
    adapter(j).execute(dunning(), NOW)

    after = adapter(j, transport())
    assert [d.reference for d in after.pending] == [reference_for(dunning())]
    assert after.reconciled == set()


def test_an_event_for_something_we_never_dispatched_is_still_not_ours(tmp_path):
    """Restoring dispatches must not widen what counts as recovery."""
    j = journal(tmp_path)
    adapter(j).execute(dunning(), NOW)

    receiver = WebhookReceiver(executor=adapter(j, transport()), secret="whsec")
    receipt = _deliver(receiver, "plink_somebody_elses", 100000, "evt_x")
    assert receipt.verdict is WebhookVerdict.UNMATCHED


def _deliver(receiver, entity_id, amount_paise, event_id):
    import hashlib
    import hmac

    body = json.dumps({
        "entity": "event",
        "event": "payment_link.paid",
        "contains": ["payment_link"],
        "payload": {"payment_link": {"entity": {
            "id": entity_id, "status": "paid", "amount_paid": amount_paise}}},
        "created_at": int(NOW.timestamp()),
    }).encode()
    signature = hmac.new(b"whsec", body, hashlib.sha256).hexdigest()
    return receiver.receive(body, signature, event_id=event_id, at=NOW)
