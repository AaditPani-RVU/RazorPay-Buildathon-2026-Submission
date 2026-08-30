"""Ground truth about what would actually recover each failed order.

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
from backstop.domain.entities import Order

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
