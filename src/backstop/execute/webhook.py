"""Closing the loop the other way round: Razorpay tells us, we do not ask.

`reconcile()` polls. It works, and it is the wrong shape for the job -- a
dispatch settles when a human being opens a link, which is minutes or days
later and out of band, so a poll either runs constantly and mostly learns
nothing, or runs rarely and reports recovery long after it happened. Razorpay
already pushes `payment_link.paid`, `order.paid` and `invoice.paid`. This is
the receiver for them.

It is a pure function of bytes in, verdict out. No server, no framework, no
network: whatever a deployment fronts this with -- a Flask route, a Lambda, a
Cloud Run handler -- hands it the raw request body and the signature header
and gets back a receipt. That is also what makes it testable offline, which
the rest of this suite is and this had to stay.

Five things it refuses to do, and each one is a way a naive receiver books
revenue it did not earn.

**It verifies before it parses.** The signature is checked against the raw
bytes, first, always. Parsing first would feed unauthenticated attacker JSON
to the parser; re-serialising a parsed body to check the signature would break
on whitespace and key order and quietly tempt somebody to skip the check. So
`receive` takes `bytes` and there is no overload that takes a dict.

**A bad signature is a refusal, not a shrug.** It is reported distinctly from
an event that is merely uninteresting, because the two mean completely
different things about the sender and only one of them should ever page
anybody.

**An event about something Backstop did not dispatch is ignored and credited
to nobody.** A merchant's own dashboard-created payment link gets paid all the
time. That is real revenue and it is not *recovered* revenue, and a receiver
that credited every `payment_link.paid` on the account would make exactly the
error the receivables surface exists to warn about: measuring the world's
ordinary behaviour and booking it as the agent's work. Only an entity this
process dispatched, matched by the id the API returned, is a recovery.

**A retry is not a second recovery.** Razorpay redelivers on non-2xx and can
redeliver on a timeout it caused itself, so the same payment arrives more than
once, and `order.paid` and `payment_link.paid` can both describe one
settlement. Deduplication is therefore two-layered: on the delivery, by event
id, and underneath it on the *dispatch*, which is the thing that can only be
recovered once. The second layer is the load-bearing one -- the first is only
an optimisation, because two different events can describe one settlement.

**It does not decide what money is worth.** Settlement value goes through
`RazorpayExecutor.credit`, the same call the poll uses, so a re-registered
mandate is a year of billing whichever way the news arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from backstop.execute.executor import ExecutionResult
from backstop.execute.razorpay import RazorpayExecutor

#: The header Razorpay signs its deliveries with.
SIGNATURE_HEADER = "X-Razorpay-Signature"

#: Events that can mean money landed on something recovery dispatched. Each
#: names the key its entity lives under in the payload. Nothing else is
#: actionable: `payment_link.expired` and friends are real events about real
#: entities, and none of them is a recovery.
SETTLING_EVENTS: dict[str, str] = {
    "payment_link.paid": "payment_link",
    "payment_link.partially_paid": "payment_link",
    "order.paid": "order",
    "invoice.paid": "invoice",
}


class Verdict(StrEnum):
    RECOVERED = "recovered"
    """Ours, settled, credited once. The only verdict that moves the ledger."""

    DUPLICATE = "duplicate"
    """Seen before, or the dispatch behind it was already credited."""

    UNMATCHED = "unmatched"
    """A real event about an entity Backstop did not dispatch. Not ours."""

    IGNORED = "ignored"
    """Authentic and uninteresting -- an event type that is not a settlement."""

    UNSETTLED = "unsettled"
    """Ours, but the entity is not paid. A partially-paid link with nothing on
    it yet, or a status the settlement rules do not count."""

    REJECTED = "rejected"
    """Signature did not verify, or the body was not an event. Never trusted."""


@dataclass(frozen=True)
class WebhookEvent:
    id: str
    """Razorpay's delivery id, from `x-razorpay-event-id`. May be absent on
    older deliveries, in which case the dispatch-level dedupe carries it."""
    name: str
    entity_kind: str
    entity: dict[str, Any]
    at: datetime

    @property
    def entity_id(self) -> str:
        return str(self.entity.get("id") or "")


@dataclass(frozen=True)
class Receipt:
    """What the receiver did with one delivery, and why.

    Carries an HTTP status because the answer matters to the sender: Razorpay
    retries anything that is not 2xx. An unmatched or duplicate event must
    therefore return 200 -- it was handled correctly and redelivering it will
    produce the same answer forever. Only a signature failure is a 400.
    """

    verdict: Verdict
    detail: str
    event: WebhookEvent | None = None
    result: ExecutionResult | None = None

    @property
    def status(self) -> int:
        return 400 if self.verdict is Verdict.REJECTED else 200

    @property
    def is_recovery(self) -> bool:
        return self.verdict is Verdict.RECOVERED

    def describe(self) -> str:
        name = self.event.name if self.event else "unparsed"
        return f"{name}: {self.verdict.value} -- {self.detail}"


def verify(body: bytes, signature: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 over the raw body, exactly as Razorpay signs.

    An empty secret returns False rather than skipping the check. A receiver
    that treats "no secret configured" as "everything is authentic" is worse
    than one that has no signature check at all, because it looks like it has
    one.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


def parse(body: bytes, *, event_id: str = "") -> WebhookEvent | None:
    """Pull the settling entity out of a delivery, or None if there is none.

    Returns None for malformed bodies and for events this receiver has no
    business acting on, and the caller distinguishes those two by the verdict
    it assigns -- not here, because "unparseable" and "uninteresting" look
    identical at this level and only the caller knows the signature passed.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("event") or "")
    kind = SETTLING_EVENTS.get(name)
    if kind is None:
        return None
    entity = (payload.get("payload") or {}).get(kind) or {}
    entity = entity.get("entity") if isinstance(entity, dict) else None
    if not isinstance(entity, dict):
        return None
    created = payload.get("created_at")
    at = (
        datetime.fromtimestamp(created, UTC)
        if isinstance(created, int | float)
        else datetime.now(UTC)
    )
    return WebhookEvent(
        id=event_id or str(payload.get("id") or ""),
        name=name,
        entity_kind=kind,
        entity=entity,
        at=at,
    )


@dataclass
class WebhookReceiver:
    """Verifies a delivery and credits it, or explains why it did not.

    Holds the executor rather than the ledger: what a settlement is *worth*
    is the adapter's answer, and routing the result onward is the caller's
    job. Keeping those apart is what lets a deployment put this behind any
    web framework without the framework learning about revenue.
    """

    executor: RazorpayExecutor
    secret: str
    seen: set[str] = field(default_factory=set, init=False)
    """Event ids already handled. The cheap layer of the two."""

    def receive(
        self, body: bytes, signature: str, *, event_id: str = "", at: datetime | None = None
    ) -> Receipt:
        now = at or datetime.now(UTC)

        if not verify(body, signature, self.secret):
            # Nothing below this line may look at the body.
            return Receipt(Verdict.REJECTED, "signature did not verify")

        event = parse(body, event_id=event_id)
        if event is None:
            return Receipt(Verdict.IGNORED, "not a settlement event this receiver acts on")

        if event.id and event.id in self.seen:
            return Receipt(
                Verdict.DUPLICATE, f"{event.id} was already handled", event=event
            )
        if event.id:
            self.seen.add(event.id)

        dispatch = self.executor.dispatch_for(event.entity_id)
        if dispatch is None:
            return Receipt(
                Verdict.UNMATCHED,
                f"{event.entity_kind} {event.entity_id} was not dispatched by recovery; "
                "this is the merchant's own revenue, not recovered revenue",
                event=event,
            )
        if dispatch.ref.entity != event.entity_kind:
            # The id matched something we dispatched and the event describes a
            # different kind of thing. Razorpay's ids are prefixed and unique,
            # so in practice this cannot happen -- which is exactly why it must
            # be refused rather than trusted. `_settlement` reads a different
            # field per entity kind, and letting the two disagree would credit
            # an auth link a year of billing on the strength of a payment
            # link's `amount_paid`.
            return Receipt(
                Verdict.UNMATCHED,
                f"{event.name} describes a {event.entity_kind}, but "
                f"{event.entity_id} was dispatched as a {dispatch.ref.entity}",
                event=event,
            )

        already = dispatch.reference in self.executor.reconciled
        result = self.executor.credit(dispatch, event.entity, at=now)
        if result is None:
            if already:
                return Receipt(
                    Verdict.DUPLICATE,
                    "this dispatch was already credited, by an earlier event or a poll",
                    event=event,
                )
            return Receipt(
                Verdict.UNSETTLED,
                f"{event.entity_kind} {event.entity_id} has not settled",
                event=event,
            )

        return Receipt(
            Verdict.RECOVERED, result.detail, event=event, result=result
        )
