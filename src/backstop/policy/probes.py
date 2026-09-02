"""A fixed bench of actions the engine must rule on the same way every time.

Two callers need this and for the same reason. The walkthrough needs to show
the safety boundary doing something, and a stream of actions drawn from the
planner does not show it: a good planner does not propose retrying a stolen
card, so a demo built only from what the planner proposes reports fourteen
`ALLOW`s and proves nothing. The console needs the same thing for the same
reason, on screen instead of in a terminal.

So the bench is written down once, here, rather than twice.

Each probe carries the disposition it is *expected* to receive. That is what
makes it a bench rather than an exhibit: an engine change that quietly turns a
denial into a permission shows up as a probe that no longer matches, in the
demo and in the console both, instead of as a line of output nobody reads
closely. The expectation is the assertion; the printout is incidental.

Every probe names a real order, invoice or customer out of the generated
batch, because a probe against a synthetic subject would be testing the rule
against a fixture rather than against the world the rest of the system runs
on. A subject the batch does not happen to contain yields no probe rather than
a fabricated one -- a bench that invents its own evidence when the seed is
inconvenient is not measuring anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, RootCause
from backstop.domain.entities import Channel, ContactRecord, Order
from backstop.domain.money import Money
from backstop.policy.engine import Disposition, PolicyContext


@dataclass(frozen=True)
class Probe:
    """One action, the context it is judged in, and what should happen to it."""

    label: str
    expect: Disposition
    action: Action
    context: PolicyContext
    note: str = ""
    """What the expectation means in words, where the disposition alone is
    not the whole story -- "rescheduled to 09:00 IST" says more than
    "rescheduled"."""

    @property
    def expected(self) -> str:
        return self.expect.value.upper() + (f" ({self.note})" if self.note else "")


def ist(base: datetime, hour: int, minute: int = 0) -> datetime:
    """A UTC instant that lands at the given IST wall-clock time."""
    return base.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        hours=hour - 5, minutes=minute - 30
    )


def pick(orders: list[Order], code: DeclineCode, **bounds) -> Order | None:
    """The first failed order with this decline code, inside optional bounds."""
    lo, hi = bounds.get("above"), bounds.get("below")
    for o in orders:
        if o.last_decline is not code:
            continue
        if lo is not None and o.amount < lo:
            continue
        if hi is not None and o.amount >= hi:
            continue
        return o
    return None


def bench(scenario) -> list[Probe]:
    """Build the bench against one generated batch.

    Ordered deliberately: the things that must never happen first, then the
    things that are permitted only on the system's terms. A reader who stops
    halfway has still seen the argument.
    """
    failed = scenario.failed_orders
    now = scenario.ends_at
    day = now - timedelta(days=1)
    cust = scenario.customers
    probes: list[Probe] = []

    def add(label, expect, action, ctx, note=""):
        probes.append(Probe(label, expect, action, ctx, note))

    # -- things that must never happen -------------------------------------
    stolen = pick(failed, DeclineCode.STOLEN_OR_LOST_CARD)
    if stolen:
        add(
            "Retry a card reported stolen",
            Disposition.DENY,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=stolen.id,
                   scheduled_at=now + timedelta(hours=1),
                   rationale="the amount is material and the order is unpaid"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )
        add(
            "Dun the customer whose card was stolen",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=stolen.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="ask them to complete payment with another method"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )

    expired = pick(failed, DeclineCode.CARD_EXPIRED)
    if expired:
        add(
            "Re-present an expired card",
            Disposition.DENY,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=expired.id,
                   scheduled_at=now + timedelta(hours=6),
                   rationale="maybe it works on a second attempt"),
            PolicyContext(now=now, order=expired, customer=cust.get(expired.customer_id)),
        )

    # The buyer travels with the invoice. Without one the consent rule denies
    # for a missing customer and the probe stops demonstrating the rule it was
    # written for -- a denial for the wrong reason is not the same evidence.
    disputed = next(
        (i for i in scenario.invoices
         if i.disputed_at and (c := cust.get(i.buyer_id)) and c.reachable_on(Channel.EMAIL)),
        None,
    )
    if disputed:
        add(
            "Chase a disputed invoice",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=disputed.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="it is overdue and unpaid"),
            PolicyContext(now=now, invoice=disputed, customer=cust.get(disputed.buyer_id)),
        )

    promised = next(
        (i for i in scenario.invoices
         if i.promise and i.promise.is_live(now)
         and (c := cust.get(i.buyer_id)) and c.reachable_on(Channel.EMAIL)),
        None,
    )
    if promised:
        add(
            "Chase a buyer who has committed to a payment date",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=promised.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="the balance is still outstanding"),
            PolicyContext(now=now, invoice=promised, customer=cust.get(promised.buyer_id)),
        )

    revoked = next(
        (s for s in scenario.lapsed_subscriptions
         if s.mandate_status.value == "revoked"),
        None,
    )
    if revoked:
        add(
            "Re-ask a customer who revoked their mandate",
            Disposition.DENY,
            Action(type=ActionType.REQUEST_MANDATE_REREGISTRATION, subject_id=revoked.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="the mandate is lapsed and the revenue is at risk"),
            PolicyContext(now=now, subscription=revoked,
                          customer=cust.get(revoked.customer_id)),
        )

    dnd_order = next(
        (o for o in failed
         if (c := cust.get(o.customer_id)) and c.dnd_registered
         and Channel.SMS in c.consented_channels and o.amount >= Money.rupees(500)),
        None,
    )
    if dnd_order:
        add(
            "SMS a customer on the DND registry",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=dnd_order.id,
                   scheduled_at=ist(day, 11), channel=Channel.SMS,
                   rationale="SMS gets read faster than email"),
            PolicyContext(now=now, order=dnd_order, customer=cust.get(dnd_order.customer_id)),
        )

    tiny = min(
        (o for o in failed
         if o.last_decline and o.last_decline.spec.max_retries == 0),
        key=lambda o: o.amount, default=None,
    )
    if tiny:
        add(
            f"Chase a balance of {tiny.amount}",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=tiny.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="every rupee counts"),
            PolicyContext(now=now, order=tiny, customer=cust.get(tiny.customer_id)),
        )

    nsf = pick(failed, DeclineCode.INSUFFICIENT_FUNDS, above=Money.rupees(500))
    if nsf:
        add(
            "Retry, when diagnosis says customers are abandoning at OTP",
            Disposition.DENY,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=nsf.id,
                   scheduled_at=nsf.last_attempt.at + timedelta(days=2),
                   rationale="the decline code is technically retryable"),
            PolicyContext(now=now, order=nsf, customer=cust.get(nsf.customer_id),
                          diagnosis=RootCause.AUTHENTICATION_DROPOFF),
        )
        add(
            "A fourth contact on the same order inside 14 days",
            Disposition.DENY,
            Action(type=ActionType.SEND_DUNNING, subject_id=nsf.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="they have not responded yet"),
            PolicyContext(
                now=now, order=nsf, customer=cust.get(nsf.customer_id),
                contacts=[
                    ContactRecord(id=f"c{n}", customer_id=nsf.customer_id,
                                  channel=Channel.EMAIL, at=now - timedelta(days=3 * n + 1),
                                  subject_ref=nsf.id)
                    for n in range(3)
                ],
            ),
        )

    # -- things that are permitted, but only on the system's terms ----------
    if nsf:
        add(
            "Retry an NSF decline 30 seconds later",
            Disposition.RESCHEDULE,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=nsf.id,
                   scheduled_at=nsf.last_attempt.at + timedelta(seconds=30),
                   rationale="try again immediately"),
            PolicyContext(now=now, order=nsf, customer=cust.get(nsf.customer_id)),
            note="to +24h",
        )

    night = next(
        (o for o in failed
         if (c := cust.get(o.customer_id)) and not c.dnd_registered
         and Channel.SMS in c.consented_channels and o.amount >= Money.rupees(500)),
        None,
    )
    if night:
        add(
            "SMS at 03:00 IST",
            Disposition.RESCHEDULE,
            Action(type=ActionType.SEND_DUNNING, subject_id=night.id,
                   scheduled_at=ist(day, 3), channel=Channel.SMS,
                   rationale="send the reminder now"),
            PolicyContext(now=now, order=night, customer=cust.get(night.customer_id)),
            note="to 09:00 IST",
        )

    timeout = pick(failed, DeclineCode.GATEWAY_TIMEOUT, above=Money.rupees(500))
    if timeout:
        add(
            "Retry into an outage that is still open",
            Disposition.RESCHEDULE,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=timeout.id,
                   scheduled_at=timeout.last_attempt.at + timedelta(minutes=10),
                   rationale="the instrument is fine, it was a timeout"),
            PolicyContext(now=now, order=timeout, customer=cust.get(timeout.customer_id),
                          outage_until=timeout.last_attempt.at + timedelta(hours=2)),
            note="past the outage",
        )
        add(
            "Retry a gateway timeout once the outage has cleared",
            Disposition.ALLOW,
            Action(type=ActionType.RETRY_PAYMENT, subject_id=timeout.id,
                   scheduled_at=timeout.last_attempt.at + timedelta(minutes=10),
                   rationale="transient infrastructure failure, instrument is healthy"),
            PolicyContext(now=now, order=timeout, customer=cust.get(timeout.customer_id)),
        )

    big = max(
        (i for i in scenario.invoices
         if i.is_chaseable(now) and (c := cust.get(i.buyer_id))
         and c.reachable_on(Channel.EMAIL)),
        key=lambda i: i.outstanding, default=None,
    )
    if big:
        add(
            f"Chase a {big.outstanding} receivable",
            Disposition.REQUIRE_APPROVAL,
            Action(type=ActionType.SEND_DUNNING, subject_id=big.id,
                   scheduled_at=ist(day, 11), channel=Channel.EMAIL,
                   rationale="materially overdue, no dispute on record"),
            PolicyContext(now=now, invoice=big, customer=cust.get(big.buyer_id)),
        )

    if stolen:
        add(
            "Escalate the stolen card to the risk team",
            Disposition.ALLOW,
            Action(type=ActionType.ESCALATE_TO_RISK, subject_id=stolen.id,
                   scheduled_at=now, rationale="fraud signal, recovery must not touch this"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )

    return probes
