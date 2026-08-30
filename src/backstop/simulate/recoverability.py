"""Ground truth about what would actually recover each failed order, each
lapsed mandate and each overdue invoice.

This is the crux of the whole measurement claim, so it is worth being blunt
about what it is. The backtest reports money recovered; that number is only
meaningful if every arm faces the same world. So the outcome of any recovery
attempt is decided **here, at generation time**, drawn once per order and
stored -- not improvised by the executor when an arm happens to act.

Two consequences follow, and both matter:

*   Arms are comparable. Whether an order is recoverable, and from when, does
    not depend on which strategy is being tested. A naive arm and a policed arm
    that both retry the same order at the same moment get the same answer.
*   The absolute rates are configuration, not findings. The probabilities below
    are stated approximations of how payment recovery behaves; they are not
    measured from Razorpay traffic, because no such data is available here. The
    honest reading of a backtest result is "this policy beats that one under
    this stated world model", not "this recovers 34% of failed payments".

The agent never sees any of this. Only `execute/` may read it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from backstop.domain.declines import DeclineCode, RetryClass
from backstop.domain.entities import Invoice, MandateStatus, Order, Subscription

#: Probability a retry succeeds *once the underlying blocker has cleared*, and
#: probability a customer acts on a contact. Keyed by what actually failed.
#:
#: Transient faults are highly recoverable: the instrument was always fine.
#: Funds failures are a coin-flip that improves with time. User-action failures
#: cannot be retried at all -- the only route is bringing the customer back.
#: Fraud blocks are unrecoverable by design; trying is the harm.
RETRY_SUCCESS: dict[RetryClass, float] = {
    RetryClass.TRANSIENT: 0.85,
    RetryClass.SOFT_RETRYABLE: 0.45,
    RetryClass.USER_ACTION_REQUIRED: 0.0,
    RetryClass.HARD_DECLINE: 0.0,
    RetryClass.FRAUD_BLOCK: 0.0,
}

#: Published dunning benchmarks put recovery of involuntary churn somewhere
#: around 15-30% of a failed cohort over a full sequence. These sit inside that
#: band. Drawn once per order, so contacting somebody five times does not give
#: five independent chances to convert -- which is exactly the error that makes
#: naive dunning look better on paper than it is.
DUNNING_CONVERSION: dict[RetryClass, float] = {
    RetryClass.TRANSIENT: 0.05,
    RetryClass.SOFT_RETRYABLE: 0.18,
    RetryClass.USER_ACTION_REQUIRED: 0.22,
    RetryClass.HARD_DECLINE: 0.12,
    RetryClass.FRAUD_BLOCK: 0.0,
}

#: How long before the blocker clears, for failures not caused by an incident.
#: Incident-caused failures heal when their incident ends, which is exact.
HEAL_DELAY_HOURS: dict[RetryClass, tuple[float, float]] = {
    RetryClass.TRANSIENT: (0.2, 1.5),
    RetryClass.SOFT_RETRYABLE: (18.0, 96.0),
}

#: How long a customer stays responsive after a failed payment. Contact them
#: inside this window and a convertible customer acts; contact them after it
#: and they have moved on -- re-bought elsewhere, forgotten, or given up.
#:
#: This is what makes dunning *timing* matter in both directions. A model where
#: any late contact still converts would reward procrastination without limit,
#: and an arm that waited a week would score best. It also creates the real
#: tension the policy engine has to navigate: prompt contact wins, but quiet
#: hours and frequency caps mean you cannot simply mail everyone immediately.
DUNNING_DEADLINE_HOURS: tuple[float, float] = (6.0, 120.0)


@dataclass(frozen=True)
class Recoverability:
    """What would recover one failed order. Drawn once; identical for all arms."""

    order_id: str
    heals_at: datetime | None
    """When a retry could start working. None means no retry ever succeeds."""
    retry_would_succeed: bool
    """Whether a retry *after* `heals_at` lands. Already accounts for the odds."""
    dunning_would_convert: bool
    """Whether the customer would act on a contact, if permitted to receive one."""
    dunning_deadline: datetime | None
    """Last moment a contact still lands. After this the customer has moved on."""
    route_switch_helps: bool
    """True only where the fault was in one route, so another route works now."""

    def retry_succeeds_at(self, at: datetime) -> bool:
        if not self.retry_would_succeed or self.heals_at is None:
            return False
        return at >= self.heals_at

    def dunning_succeeds_at(self, at: datetime) -> bool:
        """A convertible customer acts only if reached before they move on."""
        if not self.dunning_would_convert or self.dunning_deadline is None:
            return False
        return at <= self.dunning_deadline


def build(
    orders: list[Order],
    incident_end_by_order: dict[str, datetime],
    routing_caused: set[str],
    seed: int,
) -> dict[str, Recoverability]:
    """Assign latent recoverability to every failed order.

    Uses its own RNG stream, deliberately: drawing from the generator's stream
    would shift every downstream random draw and silently change the detection
    numbers that were measured before this existed.
    """
    rng = random.Random(seed ^ 0x5EED)
    out: dict[str, Recoverability] = {}

    for order in orders:
        code: DeclineCode | None = order.last_decline
        if code is None or order.is_captured:
            continue
        last = order.last_attempt
        if last is None:
            continue
        klass = code.retry_class

        # When the blocker clears. An incident-caused failure heals when the
        # incident does -- that is what makes "wait for the outage to pass"
        # a strategy the backtest can actually reward.
        heals_at: datetime | None = None
        if klass in (RetryClass.TRANSIENT, RetryClass.SOFT_RETRYABLE):
            incident_end = incident_end_by_order.get(order.id)
            if incident_end is not None:
                heals_at = incident_end
            else:
                lo, hi = HEAL_DELAY_HOURS[klass]
                heals_at = last.at + timedelta(hours=rng.uniform(lo, hi))

        retry_ok = heals_at is not None and rng.random() < RETRY_SUCCESS[klass]
        dun_ok = rng.random() < DUNNING_CONVERSION[klass]
        patience = timedelta(hours=rng.uniform(*DUNNING_DEADLINE_HOURS))

        out[order.id] = Recoverability(
            order_id=order.id,
            heals_at=heals_at,
            retry_would_succeed=retry_ok,
            dunning_would_convert=dun_ok,
            dunning_deadline=last.at + patience if dun_ok else None,
            # Switching route only helps where the route was the problem.
            route_switch_helps=order.id in routing_caused,
        )
    return out


# --------------------------------------------------------------------------
# Recurring revenue
# --------------------------------------------------------------------------

#: Whether a customer would re-authorise a lapsed mandate if asked.
#:
#: These are deliberately *not* `detect.mandates.REREGISTRATION_ODDS`. That
#: table is the agent's published estimate, the number it reasons and reports
#: with. This one is what the world actually does. Wiring the agent's belief
#: into the ground truth would make the eval a tautology -- the agent would be
#: graded against its own assumption and could never be wrong about it.
REREGISTRATION_TRUTH: dict[MandateStatus, float] = {
    MandateStatus.EXPIRED: 0.38,
    MandateStatus.PAUSED: 0.51,
    MandateStatus.NOT_REGISTERED: 0.26,
    MandateStatus.REVOKED: 0.06,
}

#: Share that come back with no prompting whatsoever.
#:
#: This is the counterfactual, and leaving it out is how recurring recovery
#: gets overstated everywhere. A paused mandate frequently resumes on its own
#: -- the customer paused for a month and always meant to return. An agent
#: that mails them and then claims the resumption has recovered nothing; it
#: has taken credit for something that was going to happen anyway.
UNPROMPTED_RESUME: dict[MandateStatus, float] = {
    MandateStatus.EXPIRED: 0.04,
    MandateStatus.PAUSED: 0.34,
    MandateStatus.NOT_REGISTERED: 0.0,
    MandateStatus.REVOKED: 0.0,
}

#: How long a lapsed-mandate customer stays reachable, in days.
#:
#: Wider than the payment dunning window by an order of magnitude, and that
#: asymmetry is the point: a failed checkout is urgent and goes cold in hours,
#: whereas somebody whose autopay lapsed has no particular deadline. Recurring
#: recovery is a slower, more patient problem than payment recovery, and a
#: policy tuned for one is wrong for the other.
REREGISTRATION_WINDOW_DAYS: tuple[float, float] = (3.0, 45.0)


@dataclass(frozen=True)
class MandateRecovery:
    """What would restore one lapsed mandate. Drawn once; identical for all arms."""

    subscription_id: str
    would_reregister: bool
    """Whether the customer would re-authorise if asked, once, ever."""
    resumes_unprompted: bool
    """Whether they would have come back with no action taken at all."""
    responsive_until: datetime | None
    """Last moment a re-registration request still lands."""

    @property
    def is_incremental(self) -> bool:
        """Whether chasing this mandate recovers anything a merchant did not
        already have coming. Recovery is a delta, not a total."""
        return self.would_reregister and not self.resumes_unprompted

    def reregisters_at(self, at: datetime) -> bool:
        if not self.is_incremental or self.responsive_until is None:
            return False
        return at <= self.responsive_until


def build_mandates(
    subscriptions: list[Subscription], reference: datetime, seed: int
) -> dict[str, MandateRecovery]:
    """Assign latent recoverability to every mandate that cannot collect.

    `reference` is the moment the book is scanned; responsiveness is measured
    from there rather than from each mandate's own lapse date, because that is
    when an agent could first have acted on any of them.

    Its own RNG stream, for the same reason `build` has one: drawing from the
    generator's stream would shift every downstream draw and silently move
    detection numbers that were measured before this existed.
    """
    rng = random.Random(seed ^ 0x3A11ED)
    out: dict[str, MandateRecovery] = {}

    for sub in subscriptions:
        if sub.cancelled_at is not None or sub.mandate_status is MandateStatus.ACTIVE:
            continue
        status = sub.mandate_status
        would = rng.random() < REREGISTRATION_TRUTH.get(status, 0.2)
        unprompted = rng.random() < UNPROMPTED_RESUME.get(status, 0.0)
        patience = timedelta(days=rng.uniform(*REREGISTRATION_WINDOW_DAYS))
        out[sub.id] = MandateRecovery(
            subscription_id=sub.id,
            would_reregister=would,
            resumes_unprompted=unprompted,
            responsive_until=reference + patience if would else None,
        )
    return out


# --------------------------------------------------------------------------
# Receivables
# --------------------------------------------------------------------------
#
# The counterfactual matters more on this surface than on either of the other
# two, and getting it wrong is how every collections tool on the market
# reports a number nobody should believe.
#
# Most overdue B2B invoices are paid whether or not anybody chases them. A
# buyer's accounts-payable department runs on a cycle; an invoice that is
# twelve days late is usually not a collections problem, it is a Tuesday. An
# agent that mails those buyers and then books the payment has measured the
# AP cycle and called it recovery.
#
# So the incremental value of collections is not the overdue balance. It is
# the far smaller slice that would *not* have arrived on its own -- and that
# slice has a shape worth knowing: it is thin among recent invoices, which
# mostly pay themselves, thin again among ancient ones, which mostly never
# pay, and thickest in the middle where a reminder actually changes the
# outcome.

#: Share of overdue invoices in each bucket that get paid if somebody chases.
#:
#: Deliberately *not* `detect.receivables.COLLECTION_ODDS`. That table is what
#: Backstop publishes and reasons with; this one is what the world does.
#: Grading the agent against its own estimate would make the result a
#: tautology.
COLLECTION_TRUTH: dict[str, float] = {
    "1-30": 0.91,
    "31-60": 0.68,
    "61-90": 0.47,
    "90+": 0.19,
}

#: Share that get paid with no contact at all -- the AP cycle turning over.
#:
#: Nested inside COLLECTION_TRUTH by construction below: a buyer who would
#: have paid unprompted would also have paid if asked, so self-curing invoices
#: are a subset of collectable ones rather than an independent draw. Anything
#: else would let an invoice be "would not pay if chased, but pays if left
#: alone", which is not a thing.
SELF_CURE_TRUTH: dict[str, float] = {
    "1-30": 0.79,
    "31-60": 0.44,
    "61-90": 0.21,
    "90+": 0.05,
}

#: Share of collectable invoices where the buyer cannot clear the balance in
#: one payment. Dunning these harder achieves nothing -- the money is not
#: there in one piece -- and only a part-payment offer unlocks anything.
#: Rises with age, because the buyers who are still not paying at ninety days
#: are disproportionately the ones who cannot.
CASHFLOW_CONSTRAINED: dict[str, float] = {
    "1-30": 0.12,
    "31-60": 0.20,
    "61-90": 0.30,
    "90+": 0.35,
}

#: What share of the balance a constrained buyer can actually find.
PART_PAYMENT_SHARE: tuple[float, float] = (0.35, 0.75)

#: How long an overdue invoice stays collectable, in days from the scan.
#:
#: Weeks, not hours. Receivables are the most patient of the three surfaces --
#: an AP department has a cycle, not a deadline -- and a dunning cadence tuned
#: to a failed checkout would burn a buyer relationship to save nothing.
COLLECTION_WINDOW_DAYS: tuple[float, float] = (10.0, 75.0)


@dataclass(frozen=True)
class InvoiceRecovery:
    """What would actually collect one overdue invoice. Drawn once per invoice."""

    invoice_id: str
    would_pay_if_chased: bool
    pays_unprompted: bool
    """Whether the AP cycle would have produced the money with no contact."""
    needs_part_payment: bool
    """The buyer has the intent but not the balance. Only an offer unlocks it."""
    part_payment_share: float
    """Largest fraction of the outstanding a constrained buyer can find. Only
    consulted when `needs_part_payment`; a buyer who can clear the balance
    pays whatever was asked instead."""
    responsive_until: datetime | None

    @property
    def is_incremental(self) -> bool:
        """Whether chasing this invoice collects anything the merchant did not
        already have coming. Recovery is a delta, not a total."""
        return self.would_pay_if_chased and not self.pays_unprompted

    def collects_at(self, at: datetime, *, offered_share: float | None) -> float:
        """Share of the outstanding balance collected by acting at `at`.

        `offered_share` is None for a plain reminder, which asks for the whole
        balance, and the fraction proposed for a part-payment offer.

        Nothing is collected from an invoice that was going to arrive anyway,
        one that was never going to arrive, or one chased after the buyer
        stopped paying attention. Past those, what lands depends on whether
        the blocker is attention or money.

        A buyer who *can* clear the balance pays whatever was asked -- the
        whole of it to a reminder, the instalment to an offer. Offering a
        split to one of them therefore collects less than a reminder would
        have, which is the cost of using the instrument in the wrong place
        and it should be visible rather than modelled away.

        A buyer who *cannot* pay a demand for the whole balance returns
        nothing to a reminder however firmly it is written, and to an offer
        returns the lesser of what was asked and what they can find.
        """
        if not self.is_incremental or self.responsive_until is None:
            return 0.0
        if at > self.responsive_until:
            return 0.0
        if self.needs_part_payment:
            if offered_share is None:
                return 0.0
            return min(self.part_payment_share, offered_share)
        return 1.0 if offered_share is None else offered_share


def build_invoices(
    invoices: list[Invoice], reference: datetime, seed: int
) -> dict[str, InvoiceRecovery]:
    """Assign latent collectability to every invoice that is overdue at `reference`.

    Disputed invoices are given no latent outcome at all. That is not an
    oversight: a dispute is a disagreement about whether the money is owed,
    and its resolution is a conversation rather than a collections outcome.
    Modelling a probability that dunning settles one would invent a reward for
    exactly the behaviour `dispute_freeze` exists to refuse.

    Its own RNG stream, for the same reason the other two builders have one.
    """
    rng = random.Random(seed ^ 0x2EC1E5)
    out: dict[str, InvoiceRecovery] = {}

    for inv in invoices:
        if inv.is_settled or inv.disputed_at is not None:
            continue
        if not inv.is_overdue(reference):
            continue
        bucket = _bucket(inv.days_overdue(reference))
        # One uniform draw against two nested thresholds, so that self-curing
        # invoices are a subset of collectable ones rather than an independent
        # coin. The same trick the generator uses for incident attribution,
        # and for the same reason: it makes the counterfactual exact.
        u = rng.random()
        chased = u < COLLECTION_TRUTH[bucket]
        unprompted = u < SELF_CURE_TRUTH[bucket]
        constrained = rng.random() < CASHFLOW_CONSTRAINED[bucket]
        share = rng.uniform(*PART_PAYMENT_SHARE)
        patience = timedelta(days=rng.uniform(*COLLECTION_WINDOW_DAYS))
        out[inv.id] = InvoiceRecovery(
            invoice_id=inv.id,
            would_pay_if_chased=chased,
            pays_unprompted=unprompted,
            needs_part_payment=constrained,
            part_payment_share=share,
            responsive_until=reference + patience if chased else None,
        )
    return out


def _bucket(days_overdue: int) -> str:
    """Aging bracket, duplicated from `detect.receivables` on purpose.

    Ground truth must not import the agent's own detection module. If the two
    ever needed to disagree about where a boundary sits, that disagreement
    should show up as a measurable error rather than be made impossible by a
    shared constant -- and a truth table that depends on the agent's code is
    one refactor away from being defined by it.
    """
    if days_overdue <= 30:
        return "1-30"
    if days_overdue <= 60:
        return "31-60"
    if days_overdue <= 90:
        return "61-90"
    return "90+"
