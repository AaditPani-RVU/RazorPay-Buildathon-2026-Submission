"""Razorpay test-mode adapter: the same `Executor` protocol, real API calls.

The simulator answers "would this action have worked?". This answers a
narrower and more literal question: "can this action be *made to happen* on
Razorpay's rails, and what did the rails say?". Those are different jobs and
conflating them would quietly corrupt the measurement, so the seam between
them is drawn explicitly here rather than left to a reader's good faith.

What the adapter maps
---------------------

Every action in the catalog resolves to one API call or to nothing:

    retry_payment / switch_route      POST /orders
    send_dunning                      POST /payment_links  (+ notify_by)
    regenerate_payment_link           POST /payment_links
    offer_part_payment                POST /payment_links, accept_partial
    request_mandate_reregistration    POST /subscription_registration/auth_links
    wait / do_nothing / escalate_*    no call at all

The three inert verbs make no network call on purpose. An escalation is a
handoff inside the merchant, not a message to a customer, and an adapter that
pinged an API to record one would be inventing traffic.

What test mode genuinely cannot do
----------------------------------

Stated plainly, because the gap is the interesting part and burying it would
make the rest of this file look better than it is.

*   **A re-presentment is not really executable.** Charging a customer again
    without asking them needs a saved token or a live mandate, and a test
    account has neither. `POST /orders` creates the *intent* a merchant's own
    checkout would then satisfy -- it is the honest floor, not a charge, and
    the result says so.
*   **Nothing settles synchronously.** A payment link is paid when a human
    opens it, which is minutes or days later and out of band. So a live
    dispatch reports `Outcome.DISPATCHED`, never `RECOVERED`, and `reconcile`
    is what closes the loop afterwards by re-reading the entity.
*   **Therefore this adapter cannot run the backtest.** The four-arm
    comparison needs a counterfactual -- what would have happened had nobody
    acted -- and reality does not offer one. Swapping this in would not make
    the measurement more real, it would make it unmeasurable. The simulator
    stays the measurement backend; this is the execution backend.

Probed rather than trusted
--------------------------

Endpoint behaviour was checked against the live test API on 2026-08-30, the
same way model availability was, because the docs and a given account's
enabled products are not the same thing:

*   `POST /payment_links` enforces **uniqueness on `reference_id`**, and
    `subscription_registration/auth_links` enforces it on `receipt`. Both
    return a 400 naming the field, in different words and with no distinct
    error code. That is free server-side idempotency and the adapter leans on
    it: a duplicate dispatch is reported as a no-op rather than sending a
    second message to the same person about the same thing.
*   `POST /orders` does **not** enforce `receipt` uniqueness -- a repeat
    silently creates a second order, which for a *charging* action is the
    worst thing here could do. `GET /orders?receipt=` does filter, so charging
    actions look before they leap and cost one extra round trip for it.
*   `POST /invoices/:id/notify/:medium` rejects secret-key auth on this
    account. `POST /payment_links/:id/notify_by/:medium` does not. Receivables
    therefore dun through a payment link carrying the invoice reference, which
    also gives part-payment offers somewhere real to live.
*   `GET /plans` returns 401: the Subscriptions product is not enabled. It is
    not needed. `subscription_registration/auth_links` works without it, which
    is the endpoint mandate re-registration actually wants -- emandate accepts
    a zero-amount authorisation, UPI and card need the ₹1 one.

Two guardrails
--------------

**Live keys are refused outright.** This code dispatches payment links and
charges at people. The difference between `rzp_test_` and `rzp_live_` is one
underscore-separated token in a `.env` file, and the failure mode is messaging
real customers. There is deliberately no flag to override it.

**Notifications are off unless asked for.** Razorpay really does send in test
mode -- a probe of `notify_by/email` returned `{"success": true}`. So links
are created silently by default and `notify=True` is a decision someone makes
on purpose, with real contact details in front of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import Rail
from backstop.domain.entities import (
    BILLING_PERIODS_PER_YEAR,
    Channel,
    Customer,
    Invoice,
    Order,
    Subscription,
    utc,
)
from backstop.domain.money import Money
from backstop.execute.executor import (
    ExecutionCosts,
    ExecutionResult,
    ExternalRef,
    Outcome,
)
from backstop.store.codec import (
    dump_action,
    dump_money,
    dump_ref,
    dump_time,
    load_action,
    load_money,
    load_ref,
    load_time,
)
from backstop.store.journal import DISPATCH, Journal

RAZORPAY_API = "https://api.razorpay.com/v1"

#: Razorpay caps both fields at 40 characters.
MAX_REFERENCE_LEN = 40

#: A payment link, and the authorisation transaction behind a card or UPI
#: mandate, must both be worth at least one rupee.
MIN_CHARGEABLE_PAISE = 100

#: Ceiling Razorpay applies to an e-mandate's `max_amount`: ₹1 crore.
MAX_MANDATE_PAISE = 1_000_000_00

#: How long a re-authorisation link stays open. Long enough that a customer
#: who reads mail weekly still gets there, short enough that a stale mandate
#: ask does not sit around for a quarter.
AUTH_LINK_VALID_DAYS = 30

#: Domain channel -> the key Razorpay's notify payloads use. Voice is absent
#: because Razorpay has no voice channel, and mapping it onto email would be
#: the adapter silently substituting its own judgement for the planner's.
NOTIFY_CHANNEL = {
    Channel.EMAIL: "email",
    Channel.SMS: "sms",
    Channel.WHATSAPP: "whatsapp",
}

#: Rail -> the registration method its mandate is authorised through.
MANDATE_METHOD = {
    Rail.EMANDATE_NACH: "emandate",
    Rail.UPI_AUTOPAY: "upi",
    Rail.UPI: "upi",
    Rail.CARD: "card",
    Rail.NETBANKING: "emandate",
    Rail.WALLET: "upi",
}


class RazorpayError(RuntimeError):
    """The dispatch did not happen, and not because a customer ignored it.

    Reserved for transport and API faults. A *capability* gap -- an action
    Razorpay has no verb for -- is recorded as a result instead, because that
    is a fact about the action worth keeping in the ledger rather than an
    exception worth unwinding a run for.
    """


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: dict[str, Any]

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def error(self) -> str:
        err = self.body.get("error")
        if isinstance(err, dict):
            return str(err.get("description") or err.get("code") or err)
        if err:
            return str(err)
        return f"HTTP {self.status}"

    @property
    def is_duplicate_reference(self) -> bool:
        """Razorpay's way of saying "you already sent this one".

        Two different endpoints say it two different ways, and both were found
        by re-running a dispatch against the live test API rather than by
        reading about it -- payment links complain about `reference_id`, auth
        links about `receipt`. There is no distinct error code for either, so
        this matches on the description.

        Deliberately narrow. Each arm requires two independent tokens, so an
        unrelated 400 is never mistaken for a successful de-duplication and
        quietly swallowed as "already sent".
        """
        if self.status != 400:
            return False
        text = self.error.lower()
        return (
            ("reference_id" in text and "already exists" in text)
            or ("receipt" in text and "unique" in text)
        )


class Transport(Protocol):
    """The whole surface the adapter needs. Swappable so tests run offline."""

    name: str

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> ApiResponse: ...


@dataclass
class HttpTransport:
    """Basic-auth JSON over httpx, with bounded retry on transient failures.

    Retries 429 and 5xx and connection faults; never a 4xx, because a
    malformed payload does not improve on the second attempt and a duplicate
    reference is an answer rather than an error.
    """

    key_id: str
    key_secret: str
    base_url: str = RAZORPAY_API
    timeout: float = 20.0
    max_retries: int = 3
    name: str = "razorpay-http"
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.key_id or not self.key_secret:
            raise RazorpayError(
                "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are not set. Put them in "
                ".env, or use RecordedTransport for offline runs."
            )
        if not self.key_id.startswith("rzp_test_"):
            raise RazorpayError(
                f"refusing to run against key {self.key_id[:12]}...: this adapter "
                "dispatches payment links and charges at customers, and is only "
                "validated in test mode. There is no override flag on purpose."
            )

    @classmethod
    def from_settings(cls, settings: Any, **kw: Any) -> HttpTransport:
        return cls(
            key_id=settings.razorpay_key_id or "",
            key_secret=settings.razorpay_key_secret or "",
            **kw,
        )

    @property
    def client(self) -> Any:  # lazily imported, mirroring the LLM providers
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self.base_url,
                auth=(self.key_id, self.key_secret),
                timeout=self.timeout,
            )
        return self._client

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> ApiResponse:
        import httpx

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.request(method, path, json=payload)
                try:
                    body = resp.json()
                except ValueError:
                    body = {"error": {"description": resp.text[:300]}}
                if not isinstance(body, dict):
                    body = {"items": body}
                out = ApiResponse(status=resp.status_code, body=body)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = RazorpayError(f"{method} {path}: {out.error}")
                else:
                    return out
            except httpx.HTTPError as err:
                last = err
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)
        raise RazorpayError(
            f"{method} {path} failed after {self.max_retries} attempts: {last}"
        ) from last

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass
class RecordedTransport:
    """Canned responses keyed by route. Makes the adapter testable offline.

    Keyed rather than ordered because the adapter's call sequence depends on
    the action mix, and a test that breaks when an unrelated action is added
    to the batch is testing the wrong thing. A route with several queued
    responses consumes them in order and then repeats the last one.
    """

    routes: dict[str, list[ApiResponse]] = field(default_factory=dict)
    name: str = "recorded"
    calls: list[tuple[str, str, dict[str, Any] | None]] = field(
        default_factory=list, init=False
    )

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> ApiResponse:
        self.calls.append((method, path, payload))
        key = f"{method} {path}"
        queued = self.routes.get(key) or self.routes.get(f"{method} {_route_shape(path)}")
        if not queued:
            raise RazorpayError(f"RecordedTransport has no response for {key}")
        return queued.pop(0) if len(queued) > 1 else queued[0]

    def payloads(self, route: str) -> list[dict[str, Any]]:
        """Every body posted to one route, for assertions."""
        return [
            p or {}
            for m, path, p in self.calls
            if f"{m} {path}" == route or f"{m} {_route_shape(path)}" == route
        ]


def _route_shape(path: str) -> str:
    """Collapse a path to the route it belongs to.

    `/payment_links/plink_x/notify_by/email` -> `/payment_links/:id/notify_by/email`,
    and `/orders?receipt=bkstp_x` -> `/orders`. Lets a recorded route match
    without knowing the id the API happened to mint, or the reference the
    action happened to hash to.
    """
    parts = path.split("?", 1)[0].strip("/").split("/")
    return "/" + "/".join(
        ":id" if "_" in p and p.split("_", 1)[0] in _ENTITY_PREFIXES else p for p in parts
    )


_ENTITY_PREFIXES = {"plink", "order", "inv", "pay", "cust", "sub"}


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def reference_for(action: Action) -> str:
    """A stable id for one action, so re-running a plan does not re-send it.

    Derived from what makes an action *the same action* -- its verb, its
    subject and the moment it was scheduled for -- and not from anything the
    model wrote. Two runs of the same plan produce the same reference; a
    genuine second chase, scheduled for a different day, produces a different
    one. `rationale` is excluded for the same reason the policy engine ignores
    it: a model rewording its own justification must not be able to buy an
    extra message to a customer.
    """
    seed = "|".join(
        [
            action.type.value,
            action.subject_id,
            utc(action.scheduled_at).isoformat(timespec="minutes"),
            action.channel.value if action.channel else "",
            str(action.amount_paise or ""),
        ]
    )
    # sha1 is a naming device here, not a security control.
    digest = hashlib.sha1(seed.encode()).hexdigest()[:24]
    return f"bkstp_{digest}"[:MAX_REFERENCE_LEN]


@dataclass(frozen=True)
class Dispatch:
    """What Razorpay created for one action, and what it was worth."""

    reference: str
    ref: ExternalRef
    action: Action
    amount: Money
    at: datetime

    @property
    def poll_path(self) -> str:
        return f"/{self.ref.entity}s/{self.ref.id}"

    def to_record(self, *, reconciled: bool) -> dict[str, Any]:
        """Everything needed to credit this dispatch in a later process.

        The entity id and its kind are the load-bearing part: a webhook
        arrives naming an id, and without a record tying that id back to the
        action that created it the receiver correctly refuses to credit a
        stranger. The amount travels too, because what a settlement is worth
        is decided by the action -- a re-registered mandate is a year of
        billing -- and a later process must not have to guess at it.
        """
        return {
            "reference": self.reference,
            "ref": dump_ref(self.ref),
            "action": dump_action(self.action),
            "amount": dump_money(self.amount),
            "at": dump_time(self.at),
            "reconciled": reconciled,
        }

    @classmethod
    def from_record(cls, body: dict[str, Any]) -> Dispatch:
        return cls(
            reference=str(body["reference"]),
            ref=load_ref(body["ref"]),
            action=load_action(body["action"]),
            amount=load_money(body["amount"]) or Money.zero(),
            at=load_time(body["at"]),
        )


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


@dataclass
class RazorpayExecutor:
    """Executes permitted actions against the Razorpay test API.

    Holds the same subject directories the simulator does, for the same
    reason: `Executor.execute` is handed an `Action` and nothing else, so
    whatever resolves a subject id into an amount and a person to contact has
    to live in the backend.

    What it does *not* hold is any notion of recoverability. That asymmetry is
    the point -- the simulator knows how the story ends and this one cannot,
    which is exactly the difference between measuring recovery and doing it.
    """

    transport: Transport
    orders: dict[str, Order] = field(default_factory=dict)
    subscriptions: dict[str, Subscription] = field(default_factory=dict)
    invoices: dict[str, Invoice] = field(default_factory=dict)
    customers: dict[str, Customer] = field(default_factory=dict)
    subscription_by_order: dict[str, str] = field(default_factory=dict)
    """Which mandate a failed order was being billed against.

    A mandate-rail payment fails as an *order*, and the playbook's answer is
    to re-authorise -- so the action arrives naming the order while the thing
    that needs re-registering is the subscription behind it. Without this map
    the adapter would refuse a perfectly good action for naming the subject
    the planner naturally names.
    """
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    notify: bool = False
    """Whether to actually send. Off by default; test mode really delivers."""
    name: str = "razorpay-test"
    journal: Journal | None = None
    """Where dispatches are written down, if anywhere.

    `dispatched` is the only handle this system has on something that has
    already left the building, and `reconciled` is what stops one settlement
    being booked as recovery twice. Held in memory alone, a restart turns a
    sent payment link into an orphan -- nothing to poll and nothing for a
    webhook to match, so a payment that lands on it can never be credited --
    and turns an already-credited dispatch back into an uncredited one, so
    the next `payment_link.paid` books the same money again. Given a journal,
    the constructor restores both, and every dispatch and every crediting is
    written as it happens.
    """
    dispatched: dict[str, Dispatch] = field(default_factory=dict, init=False)
    reconciled: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.journal is None:
            return
        for body in self.journal.replay().latest(DISPATCH).values():
            dispatch = Dispatch.from_record(body)
            self.dispatched[dispatch.reference] = dispatch
            if body.get("reconciled"):
                self.reconciled.add(dispatch.reference)

    def _remember(self, dispatch: Dispatch) -> None:
        if self.journal is not None:
            self.journal.append(
                DISPATCH, dispatch.reference,
                dispatch.to_record(reconciled=dispatch.reference in self.reconciled),
            )

    # -- the protocol ------------------------------------------------------

    def execute(self, action: Action, at: datetime) -> ExecutionResult:
        at = utc(at)
        if action.is_inert:
            return ExecutionResult(
                action, Outcome.NOT_APPLICABLE, at,
                detail=f"{action.type.value} moves no money and calls no API",
            )

        reference = reference_for(action)
        if reference in self.dispatched:
            prior = self.dispatched[reference]
            # No cost: nothing left the building, so charging the merchant for
            # it would overstate what recovery spent.
            return ExecutionResult(
                action, Outcome.NO_EFFECT, at, external=prior.ref,
                detail="already dispatched; this action was a replay, not a second contact",
            )

        subject = self._subject(action)
        if subject is None:
            return ExecutionResult(
                action, Outcome.NO_EFFECT, at,
                detail="no such subject in this batch; nothing to act on",
            )
        kind, amount, customer = subject

        gap = self._capability_gap(action, kind, customer)
        if gap:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, detail=gap)

        if action.is_charging:
            ref = self._create_order(action, amount, reference)
        elif action.type is ActionType.REQUEST_MANDATE_REREGISTRATION:
            ref = self._create_auth_link(action, reference, customer)
        else:
            ref = self._create_payment_link(action, kind, amount, reference, customer)

        if ref is None:
            # Already sent, by an earlier run or an earlier process. Recover
            # the entity it created rather than merely declining to send
            # again: without this the dispatch is unreconcilable forever, and
            # the only handle on the thing that went out is the reference. A
            # journal restores this too, and better -- it recovers the auth
            # links this lookup cannot -- but the two are not redundant: this
            # path is what an adapter running without a store, or against a
            # journal somebody deleted, still has.
            existing = self._find_dispatched(action, reference)
            if existing is None:
                return ExecutionResult(
                    action, Outcome.NO_EFFECT, at,
                    detail="Razorpay already holds this reference; not sent twice",
                )
            recovered = Dispatch(reference, existing, action, amount, at)
            self.dispatched[reference] = recovered
            self._remember(recovered)
            return ExecutionResult(
                action, Outcome.NO_EFFECT, at, external=existing,
                detail=(
                    f"already sent as {existing.id}; not sent twice, and now "
                    "reconcilable again"
                ),
            )

        cost = self.costs.for_action(action)
        dispatch = Dispatch(reference, ref, action, amount, at)
        self.dispatched[reference] = dispatch
        # Recorded before the notification goes out, so a crash between the two
        # leaves a dispatch that is known and reconcilable rather than an
        # orphan. The journal is not a transaction across the network and does
        # not pretend to be: what makes a crash *before* this line safe is that
        # `reference` is unique at Razorpay, so the retry is refused there and
        # `_find_dispatched` re-registers what already exists.
        self._remember(dispatch)
        sent = self._notify(action, ref)
        return ExecutionResult(
            action, Outcome.DISPATCHED, at, cost=cost, external=ref,
            detail=self._dispatch_detail(action, ref, sent),
        )

    def reconcile(self, *, at: datetime | None = None) -> list[ExecutionResult]:
        """Re-read everything dispatched and report what has since been paid.

        This is the half of a live run the simulator does not need. Each
        dispatch is polled once it is known to have settled and then left
        alone, so calling this repeatedly is cheap and never double-counts.

        A re-registered mandate is credited its annual value here, exactly as
        the simulator and the ledger price it. The adapter deciding for itself
        that a restored mandate is worth one billing period would put a second
        opinion about the unit of recurring revenue into the system.
        """
        now = utc(at) if at else datetime.now(UTC)
        out: list[ExecutionResult] = []
        for reference, dispatch in self.dispatched.items():
            if reference in self.reconciled:
                continue
            resp = self.transport.request("GET", dispatch.poll_path)
            if not resp.ok:
                continue
            result = self.credit(dispatch, resp.body, at=now)
            if result is not None:
                out.append(result)
        return out

    def credit(
        self, dispatch: Dispatch, body: dict[str, Any], *, at: datetime
    ) -> ExecutionResult | None:
        """Turn "the rails said something" into "this is recovered", or not.

        The single path from an API body to a recovery, used by both the poll
        above and the webhook receiver. Deliberately one function: if the two
        arrived at their own answers, a re-registered mandate could be worth a
        year down one path and one billing period down the other, and the
        ledger would be reporting whichever route happened to fire first.

        Returns None when the entity has not settled, or when this dispatch
        has already been credited. Both are ordinary, not errors -- Razorpay
        retries webhooks, and a poll can race one.
        """
        if dispatch.reference in self.reconciled:
            return None
        paid, detail = self._settlement(dispatch, body)
        if paid is None:
            return None
        self.reconciled.add(dispatch.reference)
        self._remember(dispatch)
        return ExecutionResult(
            dispatch.action, Outcome.RECOVERED, utc(at),
            recovered=paid, external=dispatch.ref, detail=detail,
        )

    def dispatch_for(self, entity_id: str) -> Dispatch | None:
        """The dispatch that created a given Razorpay entity, if we made it.

        Returns None for anything this process did not dispatch, and the
        caller must treat that as "not ours" rather than as a lookup failure.
        A merchant's own dashboard-created payment link being paid is real
        revenue and is not *recovered* revenue.
        """
        for dispatch in self.dispatched.values():
            if dispatch.ref.id == entity_id:
                return dispatch
        return None

    @property
    def pending(self) -> list[Dispatch]:
        return [d for r, d in self.dispatched.items() if r not in self.reconciled]

    # -- subjects ----------------------------------------------------------

    def _subject(self, action: Action) -> tuple[str, Money, Customer | None] | None:
        sid = action.subject_id
        if sid in self.subscriptions:
            sub = self.subscriptions[sid]
            return "mandate", sub.amount, self.customers.get(sub.customer_id)
        if sid in self.invoices:
            inv = self.invoices[sid]
            return "invoice", inv.outstanding, self.customers.get(inv.buyer_id)
        if sid in self.orders:
            order = self.orders[sid]
            return "order", order.amount, self.customers.get(order.customer_id)
        return None

    def _mandate_for(self, action: Action) -> Subscription | None:
        """The mandate an action is really about, named directly or by order."""
        sid = action.subject_id
        if sid in self.subscriptions:
            return self.subscriptions[sid]
        linked = self.subscription_by_order.get(sid)
        return self.subscriptions.get(linked) if linked else None

    def _capability_gap(
        self, action: Action, kind: str, customer: Customer | None
    ) -> str:
        """Reasons this action cannot be expressed on Razorpay's rails.

        These mirror the simulator's refusals rather than inventing new ones,
        because a merchant reading two ledgers side by side should not have to
        learn two vocabularies for the same impossibility.
        """
        if action.is_charging and kind == "mandate":
            return "mandate is not active; there is nothing to present"
        if action.is_charging and kind == "invoice":
            return "an invoice has no instrument to re-present"
        if (
            action.type is ActionType.REQUEST_MANDATE_REREGISTRATION
            and self._mandate_for(action) is None
        ):
            return "re-registration needs a mandate; no mandate stands behind this subject"
        if (
            action.is_contact
            and kind == "mandate"
            and action.type is not ActionType.REQUEST_MANDATE_REREGISTRATION
        ):
            # Nothing in the planner proposes this, but the action schema
            # permits it and a model could. A payment link against a dead
            # mandate asks for one billing period and leaves the
            # authorisation dead, which is not what recovering a stream
            # means -- the simulator credits re-registration and nothing
            # else, and the adapter must not disagree with it.
            return "a lapsed mandate is restored by re-authorisation, not by a payment link"
        if action.is_contact and action.channel is Channel.VOICE:
            return "razorpay has no voice channel; this needs a different transport"
        if action.is_contact and customer is None:
            return "no contact details on file for this subject"
        return ""

    # -- dispatch ----------------------------------------------------------

    def _create_order(
        self, action: Action, amount: Money, reference: str
    ) -> ExternalRef | None:
        """A re-presentment, as far as test mode can express one.

        `POST /orders` records the intent and the amount; it does not move
        money, because moving money without asking the customer needs a token
        this account does not have. The route the planner chose is kept in
        `notes` rather than dropped -- an acquirer preference is not
        expressible through this endpoint, and losing it silently would make
        `switch_route` and `retry_payment` indistinguishable in the dashboard.

        The lookup first is the price of correctness. Unlike payment links and
        auth links, `/orders` does **not** enforce a unique receipt -- a repeat
        silently creates a second order -- so a re-run of the same plan in a
        fresh process would quietly duplicate a charging action, which is the
        single worst thing this adapter could do. One GET per retry buys
        idempotency that survives a restart instead of only surviving a loop.
        """
        found = self.transport.request("GET", f"/orders?receipt={reference}")
        if found.ok and (found.body.get("items") or []):
            return None

        payload = {
            "amount": max(amount.paise, MIN_CHARGEABLE_PAISE),
            "currency": "INR",
            "receipt": reference,
            "notes": self._notes(action, extra={"route": action.route or "unchanged"}),
        }
        resp = self.transport.request("POST", "/orders", payload)
        if resp.is_duplicate_reference:
            return None
        if not resp.ok:
            raise RazorpayError(f"POST /orders: {resp.error}")
        return ExternalRef(entity="order", id=str(resp.body["id"]))

    def _find_dispatched(self, action: Action, reference: str) -> ExternalRef | None:
        """Recover the entity a previous run created for this reference.

        Probed rather than assumed, on 2026-08-31: `GET /payment_links` takes
        a `reference_id` filter and returns the one link, and `GET /orders`
        takes `receipt` -- the same lookup the charging path already pays for
        on the way in. Both were checked against the live test API.

        Auth links are the honest gap. They are invoices of type `link`, they
        do not come back from `GET /invoices`, and there is no documented
        filter for the `receipt` they were created with -- so a
        re-registration dispatched by a process that has since died cannot be
        recovered here and returns None. Saying so is better than a lookup
        that quietly matches the wrong invoice.
        """
        if action.is_charging:
            resp = self.transport.request("GET", f"/orders?receipt={reference}")
            items = resp.body.get("items") or [] if resp.ok else []
            if items:
                return ExternalRef(entity="order", id=str(items[0]["id"]))
            return None
        if action.type is ActionType.REQUEST_MANDATE_REREGISTRATION:
            return None
        resp = self.transport.request(
            "GET", f"/payment_links?reference_id={reference}"
        )
        if not resp.ok:
            return None
        items = resp.body.get("payment_links") or resp.body.get("items") or []
        if not items:
            return None
        return ExternalRef(
            entity="payment_link",
            id=str(items[0]["id"]),
            url=str(items[0].get("short_url") or ""),
        )

    def _create_payment_link(
        self,
        action: Action,
        kind: str,
        amount: Money,
        reference: str,
        customer: Customer | None,
    ) -> ExternalRef | None:
        """A link the customer can pay, for dunning and for part payment.

        A part-payment offer is the same link with `accept_partial` set and
        `first_min_partial_amount` naming the instalment the planner decided
        to ask for. That is the whole difference, and it matters that it is
        the whole difference: the offer is not a softer reminder, it is a
        smaller ask, and the amount is the planner's decision to make.

        Returns None when Razorpay reports the reference already exists --
        someone, possibly a previous run of this process, has already sent it.
        """
        payload: dict[str, Any] = {
            "amount": max(amount.paise, MIN_CHARGEABLE_PAISE),
            "currency": "INR",
            "description": self._describe(action, kind),
            "reference_id": reference,
            "reminder_enable": False,
            "notify": self._notify_flags(action),
            "notes": self._notes(action),
        }
        if customer:
            payload["customer"] = self._customer_payload(customer)
        if action.type is ActionType.OFFER_PART_PAYMENT:
            instalment = min(
                max(action.amount_paise or 0, MIN_CHARGEABLE_PAISE), amount.paise
            )
            payload["accept_partial"] = True
            payload["first_min_partial_amount"] = instalment

        resp = self.transport.request("POST", "/payment_links", payload)
        if resp.is_duplicate_reference:
            return None
        if not resp.ok:
            raise RazorpayError(f"POST /payment_links: {resp.error}")
        return ExternalRef(
            entity="payment_link",
            id=str(resp.body["id"]),
            url=str(resp.body.get("short_url") or ""),
        )

    def _create_auth_link(
        self, action: Action, reference: str, customer: Customer | None
    ) -> ExternalRef | None:
        """An authorisation link, to replace a mandate that stopped collecting.

        This is not dunning with different words. A lapsed mandate needs a
        fresh authorisation, which is a different ask with a different success
        rate, and Razorpay agrees: it is a different endpoint. It also works
        on an account with the Subscriptions product switched off, which is
        the case here.

        `max_amount` is the ceiling the customer authorises, and it is set to
        the mandate's own periodic amount rather than to something roomier.
        Asking for headroom the mandate does not need is asking the customer
        to sign a larger permission than the merchant is owed.
        """
        sub = self._mandate_for(action)
        assert sub is not None  # _capability_gap has already refused the alternative
        method = MANDATE_METHOD.get(sub.rail, "emandate")
        # e-mandate authorises at zero; UPI and card need a ₹1 transaction,
        # which Razorpay refunds. Probed -- zero is rejected on both.
        auth_amount = 0 if method == "emandate" else MIN_CHARGEABLE_PAISE
        expire_at = int(
            (datetime.now(UTC) + timedelta(days=AUTH_LINK_VALID_DAYS)).timestamp()
        )
        payload: dict[str, Any] = {
            "type": "link",
            "amount": auth_amount,
            "currency": "INR",
            "description": self._describe(action, "mandate"),
            "subscription_registration": {
                "method": method,
                "max_amount": min(max(sub.amount.paise, MIN_CHARGEABLE_PAISE),
                                  MAX_MANDATE_PAISE),
                "expire_at": expire_at,
            },
            "receipt": reference,
            "email_notify": 0,
            "sms_notify": 0,
            "notes": self._notes(action, extra={"rail": sub.rail.value}),
        }
        if customer:
            payload["customer"] = self._customer_payload(customer, name_key="name")
        flags = self._notify_flags(action)
        payload["email_notify"] = int(flags.get("email", False))
        payload["sms_notify"] = int(flags.get("sms", False))

        resp = self.transport.request(
            "POST", "/subscription_registration/auth_links", payload
        )
        if resp.is_duplicate_reference:
            return None
        if not resp.ok:
            raise RazorpayError(f"POST /subscription_registration/auth_links: {resp.error}")
        return ExternalRef(
            entity="invoice",  # Razorpay returns an invoice entity for auth links
            id=str(resp.body["id"]),
            url=str(resp.body.get("short_url") or ""),
        )

    def _notify(self, action: Action, ref: ExternalRef) -> bool:
        """Send the link, if sending was asked for and the rails can do it.

        Only payment links have a re-notify endpoint that accepts secret-key
        auth; an auth link carries its notification flags at creation. Orders
        are never notified -- an order is not a message.
        """
        if not self.notify or ref.entity != "payment_link":
            return False
        medium = NOTIFY_CHANNEL.get(action.channel) if action.channel else None
        if medium is None:
            return False
        resp = self.transport.request(
            "POST", f"/payment_links/{ref.id}/notify_by/{medium}"
        )
        return bool(resp.ok and resp.body.get("success", True))

    # -- payload helpers ---------------------------------------------------

    def _notify_flags(self, action: Action) -> dict[str, bool]:
        medium = NOTIFY_CHANNEL.get(action.channel) if action.channel else None
        flags = {"email": False, "sms": False}
        if self.notify and medium in flags:
            flags[medium] = True
        return flags

    def _customer_payload(
        self, customer: Customer, *, name_key: str = "name"
    ) -> dict[str, str]:
        out: dict[str, str] = {name_key: customer.id}
        if customer.email:
            out["email"] = customer.email
        if customer.phone:
            out["contact"] = customer.phone
        return out

    def _notes(self, action: Action, extra: dict[str, str] | None = None) -> dict[str, str]:
        """The audit trail, carried into the dashboard.

        Every entity Backstop creates says which action created it, on what
        subject, for when. A merchant looking at an unexplained payment link
        in their dashboard can trace it back to a ledger entry, which is the
        difference between an agent that acts and one that can be held to
        account for acting.
        """
        notes = {
            "backstop_action": action.type.value,
            "backstop_subject": action.subject_id,
            "backstop_scheduled": utc(action.scheduled_at).isoformat(timespec="minutes"),
        }
        if extra:
            notes.update(extra)
        return notes

    def _describe(self, action: Action, kind: str) -> str:
        subject = {
            "order": "order", "invoice": "invoice", "mandate": "subscription"
        }[kind]
        verb = {
            ActionType.SEND_DUNNING: "Payment due on",
            ActionType.REGENERATE_PAYMENT_LINK: "New payment link for",
            ActionType.OFFER_PART_PAYMENT: "Part payment for",
            ActionType.REQUEST_MANDATE_REREGISTRATION: "Re-authorise the mandate for",
        }.get(action.type, "Payment for")
        return f"{verb} {subject} {action.subject_id}"[:255]

    def _dispatch_detail(self, action: Action, ref: ExternalRef, sent: bool) -> str:
        if action.is_charging:
            return (
                f"re-presentment intent recorded as {ref.id}; test mode has no saved "
                "instrument, so nothing charges until the customer pays"
            )
        if not self.notify:
            return f"{ref.entity} {ref.id} created, not sent (notifications are off)"
        if sent:
            channel = action.channel.value if action.channel else "link"
            return f"{ref.entity} {ref.id} sent by {channel}"
        return f"{ref.entity} {ref.id} created; the notification carried with it"

    # -- reconciliation ----------------------------------------------------

    def _settlement(
        self, dispatch: Dispatch, body: dict[str, Any]
    ) -> tuple[Money | None, str]:
        """What this entity is now worth, or None if it has not settled."""
        status = str(body.get("status") or "")
        paid = int(body.get("amount_paid") or 0)

        if dispatch.ref.entity == "invoice":  # an auth link
            if status != "paid":
                return None, ""
            sub = self._mandate_for(dispatch.action)
            annual = (
                sub.annual_value if sub
                else dispatch.amount * BILLING_PERIODS_PER_YEAR
            )
            return annual, "customer re-authorised the mandate"

        if not paid:
            return None, ""
        if dispatch.ref.entity == "payment_link" and status == "partially_paid":
            share = paid / dispatch.amount.paise if dispatch.amount.paise else 0.0
            return Money(paid), f"buyer paid {share:.0%} of the balance"
        return Money(paid), "customer paid"


# --------------------------------------------------------------------------
# Capability probe
# --------------------------------------------------------------------------

#: Read-only endpoints, one per product family the adapter depends on. GETs
#: only: a probe that created entities to find out whether it could would be
#: a probe nobody wants to run twice.
PROBES: list[tuple[str, str, str]] = [
    ("payments", "/payments", "read the payment book"),
    ("orders", "/orders", "record a re-presentment intent"),
    ("payment_links", "/payment_links", "dun, and offer part payment"),
    ("invoices", "/invoices", "read auth links, which are invoices"),
    ("plans", "/plans", "Subscriptions product -- not required"),
]


def capabilities(transport: Transport) -> list[tuple[str, bool, str, str]]:
    """Which endpoint families this key can actually reach.

    Worth having because an account's enabled products are not the docs. On
    the account this was built against, `/plans` returns 401 and everything
    the adapter needs returns 200 -- mandate re-registration goes through
    `subscription_registration`, which does not need Subscriptions enabled.
    """
    out: list[tuple[str, bool, str, str]] = []
    for name, path, why in PROBES:
        try:
            resp = transport.request("GET", f"{path}?count=1")
            out.append((name, resp.ok, why, "" if resp.ok else resp.error))
        except RazorpayError as err:
            out.append((name, False, why, str(err)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe the Razorpay test-mode adapter's reach with the configured keys."
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    from backstop.config import Settings

    settings = Settings.load()
    if not settings.has_razorpay:
        raise SystemExit("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are not set in .env")

    transport = HttpTransport.from_settings(settings)
    rows = capabilities(transport)
    transport.close()

    if args.json:
        print(json.dumps([dict(zip(("name", "ok", "why", "error"), r)) for r in rows], indent=2))
        return
    print(f"\nrazorpay {settings.razorpay_key_id[:14]}...  test mode\n")
    for name, ok, why, err in rows:
        mark = "ok  " if ok else "no  "
        print(f"  {mark}{name:<15} {why}{'' if ok else f'  [{err}]'}")
    print()


if __name__ == "__main__":
    main()
