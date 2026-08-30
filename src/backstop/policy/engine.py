"""The policy engine. The model proposes; this disposes.

Everything upstream of here is advisory. A detector can be wrong about scope, a
diagnoser can be wrong about cause, a planner can propose something that would
be actively harmful -- and none of that reaches a customer or an issuer unless
this file permits it. That inversion is the whole design: recovery is only safe
if its guarantees hold *regardless* of model quality, which is exactly what
lets a 20b model drive it.

Three properties this is built to give:

*   **Provable, not asserted.** Every ruling names the rule that produced it.
    "Zero policy violations" is then a claim about a log, not a hope.
*   **Deny wins.** Rules cannot negotiate. Any denial is final, whatever else
    voted, so adding a rule can only ever make the system more conservative.
*   **Stopping rules are first-class.** Knowing when to give up is a feature.
    A recovery agent with no stopping condition is a harassment engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import ClassVar, Protocol
from zoneinfo import ZoneInfo

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, RetryClass, RootCause
from backstop.domain.entities import (
    Channel,
    ContactRecord,
    Customer,
    Invoice,
    MandateStatus,
    Order,
    Subscription,
    utc,
)
from backstop.domain.money import Money

IST = ZoneInfo("Asia/Kolkata")


class Disposition(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    RESCHEDULE = "reschedule"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class Verdict:
    rule_id: str
    disposition: Disposition
    reason: str
    reschedule_to: datetime | None = None


@dataclass
class PolicyConfig:
    max_contacts_per_subject: int = 3
    max_contacts_per_customer: int = 6
    """A ceiling on the *person*, not the debt. Higher than the per-subject cap
    because somebody with two unrelated problems may hear about both, and far
    below the sum of their subjects, because they are still one person."""
    contact_window_days: int = 14
    quiet_hours_start_ist: int = 21
    quiet_hours_end_ist: int = 9
    approval_threshold: Money = field(default_factory=lambda: Money.rupees(25000))
    """Above this, a person signs off. Automation earns trust in small amounts."""
    cost_per_contact: Money = field(default_factory=lambda: Money.rupees(1.5))
    cost_per_retry: Money = field(default_factory=lambda: Money.rupees(0.4))
    min_expected_margin: Money = field(default_factory=lambda: Money.rupees(5))
    """Chasing money at a loss is not recovery, it is activity."""


@dataclass
class PolicyContext:
    """Everything a rule may consider. Rules read; they never mutate."""

    now: datetime
    customer: Customer | None = None
    order: Order | None = None
    invoice: Invoice | None = None
    subscription: Subscription | None = None
    contacts: list[ContactRecord] = field(default_factory=list)
    """Everything already sent about *this subject*."""
    customer_contacts: list[ContactRecord] = field(default_factory=list)
    """Everything already sent to this person, about any subject. Separate from
    `contacts` because the two answer different questions and conflating them
    would make one of the rules below silently unenforceable."""
    diagnosis: RootCause | None = None
    outage_until: datetime | None = None
    config: PolicyConfig = field(default_factory=PolicyConfig)

    @property
    def last_decline(self) -> DeclineCode | None:
        return self.order.last_decline if self.order else None

    @property
    def subject_amount(self) -> Money:
        """What is at stake in the action's subject, for the value rules.

        A subscription is priced over a year rather than per charge, because
        re-registering a lapsed mandate does not recover one billing period,
        it restores a stream. Two rules read this and both consequences are
        intended: chasing a mandate almost always clears the cost floor, and a
        large mandate is large enough that a person signs off on it.

        Order first: a mandate *presentation* that failed is an order, and the
        money at stake in retrying that charge is the charge, not the stream.
        """
        if self.order:
            return self.order.amount
        if self.invoice:
            return self.invoice.outstanding
        if self.subscription:
            return self.subscription.annual_value
        return Money.zero()


class Rule(Protocol):
    id: str

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        """Return a verdict, or None to abstain."""
        ...


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


@dataclass
class FraudBlockRule:
    """Retrying a stolen card is not recovery; contacting its victim is worse.

    This is the one rule that blocks contact as well as charging. Everything
    else here throttles; this one refuses.
    """

    id: str = "fraud_block"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        code = ctx.last_decline
        if code is None or code.retry_class is not RetryClass.FRAUD_BLOCK:
            return None
        if action.type is ActionType.ESCALATE_TO_RISK or action.is_inert:
            return None
        return Verdict(self.id, Disposition.DENY,
                       f"{code.value} is a fraud block; escalate to risk instead")


@dataclass
class NonRetryableDeclineRule:
    """Some failures cannot be fixed by asking again, only by the customer."""

    id: str = "non_retryable_decline"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_charging:
            return None
        code = ctx.last_decline
        if code is None:
            return None
        if code.spec.is_retryable:
            return None
        return Verdict(self.id, Disposition.DENY,
                       f"{code.value} is {code.retry_class.value}; re-presenting cannot succeed")


@dataclass
class RetryBudgetRule:
    """A ceiling on attempts, set by what actually failed."""

    id: str = "retry_budget"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_charging or ctx.order is None:
            return None
        code = ctx.last_decline
        if code is None:
            return None
        used = ctx.order.recovery_attempts
        if used >= code.spec.max_retries:
            return Verdict(self.id, Disposition.DENY,
                           f"retry budget exhausted: {used}/{code.spec.max_retries} for {code.value}")
        return None


@dataclass
class RetrySpacingRule:
    """Re-presenting an NSF decline in thirty seconds burns the merchant's
    decline ratio and cannot work. Spacing is set by the failure, not taste."""

    id: str = "retry_spacing"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_charging or ctx.order is None:
            return None
        code, last = ctx.last_decline, ctx.order.last_attempt
        if code is None or last is None:
            return None
        earliest = last.at + timedelta(seconds=code.spec.min_retry_delay_s)
        if utc(action.scheduled_at) < earliest:
            return Verdict(self.id, Disposition.RESCHEDULE,
                           f"{code.value} requires {code.spec.min_retry_delay_s // 3600}h spacing",
                           reschedule_to=earliest)
        return None


@dataclass
class ConsentRule:
    """Absence of consent is not permission, and an opt-out is permanent."""

    id: str = "contact_consent"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_contact:
            return None
        if ctx.customer is None or action.channel is None:
            return Verdict(self.id, Disposition.DENY, "contact action without a customer or channel")
        if not ctx.customer.reachable_on(action.channel):
            why = (
                "opted out" if ctx.customer.opted_out_at
                else "on the DND registry" if ctx.customer.dnd_registered
                and action.channel in (Channel.SMS, Channel.VOICE)
                else "no consent on record"
            )
            return Verdict(self.id, Disposition.DENY,
                           f"cannot contact on {action.channel.value}: {why}")
        return None


@dataclass
class QuietHoursRule:
    """Nobody's overdue invoice justifies a 3am SMS. Email is exempt: it waits
    in an inbox rather than making a phone light up."""

    id: str = "quiet_hours"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_contact or action.channel in (None, Channel.EMAIL):
            return None
        cfg = ctx.config
        local = utc(action.scheduled_at).astimezone(IST)
        if cfg.quiet_hours_end_ist <= local.hour < cfg.quiet_hours_start_ist:
            return None
        next_ok = local.replace(hour=cfg.quiet_hours_end_ist, minute=0, second=0, microsecond=0)
        if local.hour >= cfg.quiet_hours_start_ist:
            next_ok += timedelta(days=1)
        return Verdict(self.id, Disposition.RESCHEDULE,
                       f"{local:%H:%M} IST is inside quiet hours",
                       reschedule_to=next_ok.astimezone(UTC))


@dataclass
class ContactFrequencyRule:
    """The stopping rule that stops recovery becoming harassment."""

    id: str = "contact_frequency"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_contact:
            return None
        cfg = ctx.config
        since = ctx.now - timedelta(days=cfg.contact_window_days)
        recent = [c for c in ctx.contacts
                  if c.subject_ref == action.subject_id and utc(c.at) >= since]
        if len(recent) >= cfg.max_contacts_per_subject:
            return Verdict(self.id, Disposition.DENY,
                           f"{len(recent)} contacts in {cfg.contact_window_days}d "
                           f"reaches the cap of {cfg.max_contacts_per_subject}")
        return None


@dataclass
class ContactFatigueRule:
    """The stopping rule for the *person*, where the frequency cap stops the debt.

    A per-subject cap is the obvious one and it is not enough, because a
    subject is not a person. One buyer sits behind four overdue invoices; one
    customer has a failed order, a lapsed mandate and a second failed order
    the following week. Every per-subject cap can be scrupulously observed
    while that person is contacted a dozen times in a fortnight, and each
    individual message passes review.

    This is the receivables surface's characteristic hazard rather than a
    general nicety -- invoices cluster on buyers far harder than orders
    cluster on customers -- but the rule is written on the customer because
    the harm is, and it protects all three surfaces for the same reason.
    """

    id: str = "contact_fatigue"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_contact or ctx.customer is None:
            return None
        cfg = ctx.config
        since = ctx.now - timedelta(days=cfg.contact_window_days)
        recent = [c for c in ctx.customer_contacts if utc(c.at) >= since]
        if len(recent) >= cfg.max_contacts_per_customer:
            subjects = len({c.subject_ref for c in recent})
            return Verdict(self.id, Disposition.DENY,
                           f"{len(recent)} contacts to this customer across {subjects} "
                           f"subject(s) in {cfg.contact_window_days}d reaches the "
                           f"per-person cap of {cfg.max_contacts_per_customer}")
        return None


@dataclass
class DisputeFreezeRule:
    """A contested invoice is a conversation, not a collection."""

    id: str = "dispute_freeze"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        inv = ctx.invoice
        if inv is None or inv.disputed_at is None or action.is_inert:
            return None
        return Verdict(self.id, Disposition.DENY,
                       "invoice is disputed; a human owns this, not automation")


@dataclass
class PromiseToPayRule:
    """A buyer who committed to a date has earned silence until it passes."""

    id: str = "promise_to_pay"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        inv = ctx.invoice
        if inv is None or inv.promise is None or action.is_inert:
            return None
        if inv.promise.is_live(ctx.now):
            return Verdict(self.id, Disposition.DENY,
                           f"promise to pay is live until {inv.promise.promised_for:%Y-%m-%d}")
        return None


@dataclass
class RevokedMandateRule:
    """A revoked mandate is a decision, not a lapse.

    Expiry is administrative and asking again is a courtesy. Revocation is the
    customer switching the payments off on purpose, and an agent that responds
    by re-requesting authorisation has not recovered revenue -- it has ignored
    a cancellation. Roughly one in twelve ever comes back, so the expected
    value does not justify overriding somebody's stated decision either.

    A person may still decide to ask. This rule only refuses to let automation
    make that call.
    """

    id: str = "revoked_mandate"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if action.is_inert:
            return None
        revoked = (
            ctx.subscription is not None
            and ctx.subscription.mandate_status is MandateStatus.REVOKED
        ) or ctx.last_decline is DeclineCode.MANDATE_REVOKED
        if not revoked:
            return None
        return Verdict(self.id, Disposition.DENY,
                       "mandate was revoked by the customer; a person decides whether to ask again")


@dataclass
class CancelledSubscriptionRule:
    """Nothing is owed on a subscription the customer already ended."""

    id: str = "cancelled_subscription"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if action.is_inert or ctx.subscription is None:
            return None
        if ctx.subscription.cancelled_at is None:
            return None
        return Verdict(self.id, Disposition.DENY,
                       "subscription is cancelled; there is nothing left to collect")


@dataclass
class DiagnosisGateRule:
    """Some causes make re-presenting pointless however healthy the instrument.

    An authentication drop-off means the customer left at the OTP screen. The
    card is fine; nobody is there. Retrying spends the merchant's decline ratio
    to achieve nothing, so the only route is re-engagement.
    """

    id: str = "diagnosis_gate"

    BLOCKS_CHARGING: ClassVar[set[RootCause]] = {
        RootCause.AUTHENTICATION_DROPOFF,
        RootCause.FRAUD_PRESSURE,
        RootCause.INVOICE_DISPUTE,
        RootCause.MANDATE_LIFECYCLE_FAILURE,
    }

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_charging or ctx.diagnosis is None:
            return None
        if ctx.diagnosis in self.BLOCKS_CHARGING:
            return Verdict(self.id, Disposition.DENY,
                           f"{ctx.diagnosis.value} cannot be resolved by re-presenting")
        return None


@dataclass
class OutageHoldRule:
    """Retrying into a known outage manufactures failures and wastes budget."""

    id: str = "outage_hold"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if not action.is_charging or ctx.outage_until is None:
            return None
        if utc(action.scheduled_at) < utc(ctx.outage_until):
            return Verdict(self.id, Disposition.RESCHEDULE,
                           "issuer or rail is still degraded",
                           reschedule_to=utc(ctx.outage_until))
        return None


@dataclass
class HighValueApprovalRule:
    """Automation earns trust in small amounts. Large sums get a person."""

    id: str = "high_value_approval"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if action.is_inert:
            return None
        if ctx.subject_amount >= ctx.config.approval_threshold:
            return Verdict(self.id, Disposition.REQUIRE_APPROVAL,
                           f"{ctx.subject_amount} is at or above the "
                           f"{ctx.config.approval_threshold} approval threshold")
        return None


#: Rough recovery odds by failure class, used only to decide whether an attempt
#: is worth its cost. Conservative on purpose: over-estimating here spends real
#: money chasing value that is not there.
RECOVERY_ODDS: dict[RetryClass, float] = {
    RetryClass.TRANSIENT: 0.62,
    RetryClass.SOFT_RETRYABLE: 0.34,
    RetryClass.USER_ACTION_REQUIRED: 0.18,
    RetryClass.HARD_DECLINE: 0.02,
    RetryClass.FRAUD_BLOCK: 0.0,
}


@dataclass
class CostOfRecoveryRule:
    """Chasing money at a loss is activity, not recovery.

    Small balances are where automated dunning quietly destroys value: three
    SMS to recover ninety rupees is a worse outcome than writing it off.
    """

    id: str = "cost_of_recovery"

    def check(self, action: Action, ctx: PolicyContext) -> Verdict | None:
        if action.is_inert:
            return None
        cfg = ctx.config
        cost = cfg.cost_per_contact if action.is_contact else cfg.cost_per_retry
        code = ctx.last_decline
        odds = RECOVERY_ODDS.get(code.retry_class, 0.2) if code else 0.25
        expected = ctx.subject_amount * odds
        if expected - cost < cfg.min_expected_margin:
            return Verdict(self.id, Disposition.DENY,
                           f"expected recovery {expected} does not clear cost {cost} "
                           f"plus margin {cfg.min_expected_margin}")
        return None


DEFAULT_RULES: list[Rule] = [
    FraudBlockRule(),
    DisputeFreezeRule(),
    PromiseToPayRule(),
    RevokedMandateRule(),
    CancelledSubscriptionRule(),
    NonRetryableDeclineRule(),
    DiagnosisGateRule(),
    RetryBudgetRule(),
    ConsentRule(),
    ContactFrequencyRule(),
    ContactFatigueRule(),
    CostOfRecoveryRule(),
    HighValueApprovalRule(),
    RetrySpacingRule(),
    QuietHoursRule(),
    OutageHoldRule(),
]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class Ruling:
    """What the engine decided about one proposed action, and why."""

    proposed: Action
    disposition: Disposition
    verdicts: list[Verdict]
    final: Action | None = None
    """The action as permitted, possibly rescheduled. None when denied."""

    @property
    def allowed(self) -> bool:
        return self.disposition in (Disposition.ALLOW, Disposition.REQUIRE_APPROVAL)

    @property
    def blocking_rule(self) -> str | None:
        for v in self.verdicts:
            if v.disposition is Disposition.DENY:
                return v.rule_id
        return None

    def describe(self) -> str:
        # Show the action as permitted, not as proposed: an audit trail that
        # prints the requested time for a rescheduled action is misleading
        # about what the system actually did.
        shown = self.final or self.proposed
        head = f"{self.disposition.value.upper():<16} {shown.describe()}"
        why = "; ".join(f"[{v.rule_id}] {v.reason}" for v in self.verdicts) or "no rule objected"
        return f"{head}\n                 {why}"


class PolicyEngine:
    """Evaluates every rule against every action. No rule may be skipped.

    Rules are not consulted until one objects: all of them speak, so the audit
    trail records every reason an action was constrained, not merely the first.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = list(rules if rules is not None else DEFAULT_RULES)

    def evaluate(self, action: Action, ctx: PolicyContext) -> Ruling:
        verdicts = [v for v in (r.check(action, ctx) for r in self.rules) if v is not None]

        denials = [v for v in verdicts if v.disposition is Disposition.DENY]
        if denials:
            return Ruling(action, Disposition.DENY, verdicts, final=None)

        # Reschedules compose: take the latest constraint so every rule that
        # asked for a delay is satisfied at once.
        moves = [v.reschedule_to for v in verdicts
                 if v.disposition is Disposition.RESCHEDULE and v.reschedule_to]
        final = action
        disposition = Disposition.ALLOW
        if moves:
            final = action.model_copy(update={"scheduled_at": max(moves)})
            disposition = Disposition.RESCHEDULE
            # A delay can push an action into quiet hours, so settle the
            # schedule before deciding it is permitted.
            for _ in range(3):
                again = [r.check(final, ctx) for r in self.rules]
                more = [v.reschedule_to for v in again
                        if v and v.disposition is Disposition.RESCHEDULE and v.reschedule_to]
                if not more or max(more) <= utc(final.scheduled_at):
                    break
                final = final.model_copy(update={"scheduled_at": max(more)})
                verdicts.extend(v for v in again if v is not None)

        if any(v.disposition is Disposition.REQUIRE_APPROVAL for v in verdicts):
            disposition = Disposition.REQUIRE_APPROVAL

        return Ruling(action, disposition, verdicts, final=final)

    def evaluate_all(self, actions: list[Action], ctx: PolicyContext) -> list[Ruling]:
        return [self.evaluate(a, ctx) for a in actions]
